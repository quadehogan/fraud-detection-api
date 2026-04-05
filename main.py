"""
main.py - FastAPI fraud-detection REST API.

Endpoints:
    GET  /health  -> liveness check
    POST /predict -> score one order, return fraud probability + label

Fix: model lazy-loaded on first /predict so Railway passes health checks
before the heavy model.pkl load completes.
"""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import FEATURE_COLS, engineer_features

MODEL_PATH = Path(__file__).parent / "model.pkl"
_model = None  # lazy-loaded on first /predict request


def _get_model():
    """Load and cache the model on first call."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            pass
    return _model


class OrderInput(BaseModel):
    customer_id: int = Field(..., example=42)
    order_datetime: str = Field(..., example="2025-11-29 14:23:00")
    billing_zip: str = Field(..., example="10001")
    shipping_zip: str = Field(..., example="90210")
    shipping_state: str = Field(..., example="CA")
    payment_method: str = Field(..., example="card")
    device_type: str = Field(..., example="mobile")
    ip_country: str = Field(..., example="US")
    promo_used: int = Field(..., example=0)
    order_subtotal: float = Field(..., example=299.99)
    shipping_fee: float = Field(..., example=14.99)
    tax_amount: float = Field(..., example=24.00)
    order_total: float = Field(..., example=338.98)
    risk_score: float = Field(..., example=22.5)
    gender: str = Field(..., example="Female")
    birthdate: str = Field(..., example="1990-06-15")
    customer_segment: str = Field(..., example="standard")
    loyalty_tier: str = Field(..., example="silver")
    customer_created_at: str = Field(..., example="2025-10-01 09:00:00")
    customer_zip: str = Field(..., example="10001")
    item_count: int = Field(1, example=3)
    total_quantity: int = Field(1, example=4)
    unique_products: int = Field(1, example=2)


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_fraud: bool
    risk_level: str = Field(..., description="low | medium | high")
    model_version: str = "1.0.0"
    scored_at: str


app = FastAPI(
    title="Fraud Detection API",
    description="GradientBoosting fraud risk scorer.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    raw = pd.DataFrame([order.model_dump()])
    features = engineer_features(raw)

    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = np.nan

    X = features[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0, 1])
    pred = bool(model.predict(X)[0])
    risk_level = "low" if prob < 0.30 else ("medium" if prob < 0.70 else "high")

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        predicted_fraud=pred,
        risk_level=risk_level,
        scored_at=datetime.utcnow().isoformat(),
    )
