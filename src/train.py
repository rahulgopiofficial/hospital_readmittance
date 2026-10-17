"""
train.py — Stage 3: train a baseline + advanced model, track with MLflow,
register and promote the best. Imported by Model_Development_and_Tracking.ipynb.

Run:  python -m src.train
"""
from matplotlib import category
import config as cfg
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import src.evaluate as ev
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
import joblib
import json
import pandas as pd


def train_and_log():
    """TODO (Stage 3):
    - Get splits + feature lists from src.data_prep; compute scale_pos_weight = neg / pos.
    - Build two FULL sklearn Pipelines (preprocessor + classifier):
        * LogisticRegression(class_weight='balanced')           (baseline)
        * XGBClassifier(scale_pos_weight=..., **cfg.XGB_PARAMS)  (advanced)
    - For each model: fit on train, evaluate on validation (src.evaluate.compute_metrics),
      and log params + metrics + the model to the MLflow experiment cfg.MLFLOW_EXPERIMENT.
    - Pick the best by validation ROC-AUC, evaluate on the test split, then register it
      (cfg.REGISTERED_MODEL) and promote to the 'production' alias.
    - Save artifacts: best_model.pkl, metrics.json, reference_sample.csv, input_columns.json.
    Return (best_model_name, test_metrics). Expected validation ROC-AUC ~0.65."""

    # Importing the data_prep python file for splits and feature list function
    from src.data_prep import get_splits, get_feature_lists, build_preprocessor

    # Get the split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    # Get the feature list
    numeric_cols, categorical_cols = get_feature_lists(X_train)

    # Compute scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # LogisticRegression(class_weight='balanced') (baseline) 
    logistic_pipeline = Pipeline([
        ('preprocessor', build_preprocessor(numeric_cols, categorical_cols)),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=cfg.RANDOM_STATE))
    ])

    # XGBClassifier(scale_pos_weight=..., **cfg.XGB_PARAMS)  (advanced)
    xgb_pipeline = Pipeline([
        ('preprocessor', build_preprocessor(numeric_cols, categorical_cols)),
        ('classifier', XGBClassifier(scale_pos_weight=scale_pos_weight, **cfg.XGB_PARAMS))
    ])
    
    # Set the tracking experiment before starting runs
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    # Fit, Evaluate, and Log the Baseline Model
    with mlflow.start_run(run_name="Baseline_LogReg") as run_base:
        logistic_pipeline.fit(X_train, y_train)
        
        # Evaluate using predict_proba for the positive class (1)
        y_val_prob_base = logistic_pipeline.predict_proba(X_val)[:, 1]
        base_metrics = ev.compute_metrics(y_val, y_val_prob_base)
        
        # Log to MLflow
        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced"})
        mlflow.log_metrics(base_metrics)
        mlflow.sklearn.log_model(
            sk_model=logistic_pipeline, 
            name="model", 
            serialization_format="cloudpickle"
        )
        
        run_id_base = run_base.info.run_id

    # Fit, Evaluate, and Log the Advanced Model
    with mlflow.start_run(run_name="Advanced_XGBoost") as run_xgb:
        xgb_pipeline.fit(X_train, y_train)
        
        y_val_prob_xgb = xgb_pipeline.predict_proba(X_val)[:, 1]
        xgb_metrics = ev.compute_metrics(y_val, y_val_prob_xgb)
        
        # Log to MLflow
        mlflow.log_params({"model_type": "XGBoost", "scale_pos_weight": scale_pos_weight})
        mlflow.log_params(cfg.XGB_PARAMS)
        mlflow.log_metrics(xgb_metrics)
        mlflow.sklearn.log_model(
            sk_model=xgb_pipeline, 
            name="model", 
            serialization_format="cloudpickle"
        )
        
        run_id_xgb = run_xgb.info.run_id

    # Model Selection based on Validation ROC-AUC
    if xgb_metrics['roc_auc'] > base_metrics['roc_auc']:
        best_model = xgb_pipeline
        best_model_name = "Advanced_XGBoost"
        best_run_id = run_id_xgb
    else:
        best_model = logistic_pipeline
        best_model_name = "Baseline_LogReg"
        best_run_id = run_id_base

    # Evaluate the best model on the unseen Test split
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = ev.compute_metrics(y_test, y_test_prob)

    # Model Registry and Promotion to 'production' alias
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, cfg.REGISTERED_MODEL)
    
    client = MlflowClient()
    client.set_registered_model_alias(
        name=cfg.REGISTERED_MODEL, 
        alias="production", 
        version=registered_model.version
    )

    # Save Local Artifacts
    joblib.dump(best_model, cfg.ARTIFACT_DIR / "best_model.pkl")
    
    with (cfg.ARTIFACT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
        
    # Save a small reference sample and feature columns for future inference mapping
    reference_sample = X_train.sample(
        n=min(1000, len(X_train)),
        random_state=cfg.RANDOM_STATE,
    )
    
    reference_sample.to_csv(
        cfg.ARTIFACT_DIR / "reference_sample.csv",
        index=False,
    )

    with (cfg.ARTIFACT_DIR / "input_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f, indent=4)

    # 7. Return required output
    return best_model_name, test_metrics

if __name__ == "__main__":
    train_and_log()
