"""
retrain.py (STARTER) — Stage 4.5: multi-signal retraining trigger + workflow.

Run:  python -m src.retrain
"""
import config as cfg
import json

PSI_THRESHOLD = 0.20            # representative example for instruction
DRIFT_SHARE_THRESHOLD = 0.30   # representative example for instruction


def decide(summary):
    """TODO: return a list of human-readable reasons if any trigger fires —
    prediction_psi > PSI_THRESHOLD, share_drifted > DRIFT_SHARE_THRESHOLD, or dataset_drift is True.
    An empty list means no retraining is needed."""
    psi = float(summary["prediction_psi"])
    drifted = int(summary["number_of_drifted_columns"])
    total = int(summary["number_of_columns"])

    if total <= 0:
        raise ValueError("number_of_columns must be greater than zero")

    drift_share = drifted / total
    reasons = []

    if psi > PSI_THRESHOLD:
        reasons.append(f"Prediction PSI {psi:.3f} exceeds {PSI_THRESHOLD:.2f}")
    if drift_share > DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"Drifted feature share {drift_share:.1%} exceeds "
            f"{DRIFT_SHARE_THRESHOLD:.0%}"
        )
    if summary["dataset_drift"]:
        reasons.append("Evidently detected dataset drift")

    return reasons


def run_retraining_workflow():
    """TODO (Stage 4.5): read artifacts/drift_summary.json; if decide() returns reasons, retrain
    (call src.train.train_and_log, which registers + promotes a new version) and record
    artifacts/retraining_decision.json with the reasons + new metrics. Return the decision dict.
    In production the retrain would use freshly LABELLED data; here it re-runs the pipeline to
    demonstrate the automated workflow + versioning."""

    summary_path = cfg.ARTIFACT_DIR / "drift_summary.json"
    
    if not summary_path.is_file():
        raise FileNotFoundError(f"Drift summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reasons = decide(summary)

    decision = {
        "retrained": bool(reasons),
        "reasons": reasons,
        "model_name": None,
        "metrics": None,
    }

    if reasons:
        from src.train import train_and_log

        model_name, metrics = train_and_log()
        decision["model_name"] = model_name
        decision["metrics"] = metrics

    decision_path = cfg.ARTIFACT_DIR / "retraining_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    return decision


if __name__ == "__main__":
    run_retraining_workflow()
