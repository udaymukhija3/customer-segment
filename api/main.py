from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
KMEANS_PATH = ARTIFACTS_DIR / "kmeans_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


class CustomerInput(BaseModel):
    age: int
    annual_income: int
    spending_score: int


app = FastAPI(title="Customer Segmentation API", version="1.0.0")


SEGMENT_NAME: Dict[int, str] = {
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "High Spender, Low Income",
    4: "Segment 4",
}


@app.on_event("startup")
def load_models() -> None:
    global KMEANS_MODEL, SCALER
    if not KMEANS_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            f"Artifacts not found. Expected '{KMEANS_PATH}' and '{SCALER_PATH}'. Run training first."
        )
    KMEANS_MODEL = load(KMEANS_PATH)
    SCALER = load(SCALER_PATH)


@app.post("/get_segment")
def get_segment(payload: CustomerInput) -> dict:
    try:
        X = np.array([[payload.age, payload.annual_income, payload.spending_score]], dtype=float)
        X_scaled = SCALER.transform(X)
        seg_id = int(KMEANS_MODEL.predict(X_scaled)[0])
        return {
            "segment_id": seg_id,
            "segment_name": SEGMENT_NAME.get(seg_id, f"Segment {seg_id}"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


