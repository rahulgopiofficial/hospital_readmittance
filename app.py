"""
app.py — Stage 4.1: FastAPI inference service.
Implement GET /health and POST /predict, then demonstrate it from the operations notebook.

Run:  uvicorn app:app --reload
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5

app = FastAPI(title="Hospital Readmission Predictor", version="1.0")


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw encounter features (column -> value)")

# - Lazily load artifacts/best_model.pkl and artifacts/input_columns.json.
@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(ARTIFACTS / "best_model.pkl")
    input_columns = json.loads((ARTIFACTS / "input_columns.json").read_text(encoding="utf-8"))
    return model, input_columns

# - GET  /health  -> {"status": ..., "model_loaded": bool}.
@app.get("/health")
def health():
    load_artifacts()
    return {"status": "ok", "model_loaded": True}

# - POST /predict -> align the incoming features to the training columns (missing -> NaN; the pipeline imputes), 
#   return {"readmission_probability", "readmitted_30d", "threshold"}.
#   Validate the payload (reject an empty features dict with 422) and log every prediction to
#   artifacts/predictions.log (governance). Note: JSON has no NaN, so clients send null / omit fields.
@app.post("/predict")
def predict(request: PredictRequest):
    if not request.features:
        raise HTTPException(status_code=422, detail="'features' must not be empty")

    model, input_columns = load_artifacts()
    row = {
        column: (
            float("nan")
            if request.features.get(column) is None
            else request.features.get(column, float("nan"))
        )
        for column in input_columns
    }
    frame = pd.DataFrame([row], columns=input_columns)
    probability = float(model.predict_proba(frame)[0, 1])
    label = int(probability >= THRESHOLD)

    result = {
        "readmission_probability": probability,
        "readmitted_30d": label,
        "threshold": THRESHOLD,
    }
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    with (ARTIFACTS / "predictions.log").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return result


