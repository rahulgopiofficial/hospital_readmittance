"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:  python -m src.monitoring
"""
import config as cfg
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import joblib
import numpy as np
import json

def load_scored_batches():
    """Load the reference/current batches and their predicted readmission probabilities."""
    artifacts = cfg.ARTIFACT_DIR
    input_columns = json.loads(
        (artifacts / "input_columns.json").read_text(encoding="utf-8")
    )
    reference = pd.read_csv(artifacts / "reference_sample.csv")
    current = pd.read_csv(artifacts / "current_batch.csv")
    model = joblib.load(artifacts / "best_model.pkl")

    reference_scores = model.predict_proba(reference[input_columns])[:, 1]
    current_scores = model.predict_proba(current[input_columns])[:, 1]
    return reference, current, reference_scores, current_scores


def run_monitoring():
    """TODO (Stage 4.4):
    - Load artifacts/reference_sample.csv (training baseline) and artifacts/current_batch.csv.
    - Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
      read number_of_drifted_columns / number_of_columns / dataset_drift.
    - Prediction drift: load artifacts/best_model.pkl and compute the PSI between the reference and
      current predicted probabilities.
    - Write artifacts/drift_summary.json and return the summary dict.
    Expected (reference run): ~6/39 features drifted, dataset_drift False, prediction PSI ~0.46."""

    reference, current, reference_scores, current_scores = load_scored_batches()
    artifacts = cfg.ARTIFACT_DIR
    input_columns = json.loads(
        (artifacts / "input_columns.json").read_text(encoding="utf-8")
      )

    reference_features = reference[input_columns].copy()
    current_features = current[input_columns].copy()

    for column in input_columns:
        if reference_features[column].isna().all():
            reference_features[column] = (
                reference_features[column].astype("object").fillna("__MISSING__")
            )
            current_features[column] = (
                current_features[column].astype("object").fillna("__MISSING__")
            )

    
    
    # Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
    # read number_of_drifted_columns / number_of_columns / dataset_drift.

    report = Report([DataDriftPreset(columns=input_columns, drift_share=0.5)])
    snapshot = report.run(
        current_data=current_features,
        reference_data=reference_features,
      )
    snapshot.save_html(str(artifacts / "drift_report.html"))

    drift_count_metric = next(
        metric for metric in snapshot.dict()["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount(")
      )

    number_of_drifted_columns = int(drift_count_metric["value"]["count"])
    number_of_columns = len(input_columns)
    dataset_drift = number_of_drifted_columns / number_of_columns >= 0.5

    
    
    # Prediction drift: load artifacts/best_model.pkl and compute the PSI between the reference and
    # current predicted probabilities.

    # Use identical bins for both score distributions.
    bins = np.linspace(0.0, 1.0, 11)
    reference_counts = np.histogram(reference_scores, bins=bins)[0].astype(float)
    current_counts = np.histogram(current_scores, bins=bins)[0].astype(float)

    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    # Smooth empty bins to avoid taking the logarithm of zero.
    epsilon = 0.01
    reference_pct = np.maximum(reference_pct, epsilon)
    current_pct = np.maximum(current_pct, epsilon)
    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    prediction_psi = float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
          )
      )


    # Write artifacts/drift_summary.json and return the summary dict.
    # Expected (reference run): ~6/39 features drifted, dataset_drift False, prediction PSI ~0.46.

    summary = {
        "number_of_drifted_columns": int(number_of_drifted_columns),
        "number_of_columns": int(number_of_columns),
        "dataset_drift": bool(dataset_drift),
        "prediction_psi": float(prediction_psi),
      }

    summary_path = artifacts / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    
    return summary

if __name__ == "__main__":
    run_monitoring()
