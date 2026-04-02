"""
main.py — FastAPI fraud-detection REST API.

Endpoints:
    GET  /health   → liveness check
    POST /predict  → score one order, return fraud probability + label
"""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import FEATURE_COLS, engineer_features

# ── Load model at startup ─────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"

try:
    _model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    _model = None


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class OrderInput(BaseModel):
    """Raw order + customer fields — identical to what the DB join produces."""

    # Order fields
    customer_id:     int   = Field(..., example=42)
    order_datetime:  str   = Field(..., example="2025-11-29 14:23:00",
                                   description="ISO datetime of order placement")
    billing_zip:     str   = Field(..., example="10001")
    shipping_zip:    str   = Field(..., example="90210")
    shipping_state:  str   = Field(..., example="CA")
    payment_method:  str   = Field(..., example="card",
                                   description="card | paypal | bank | crypto")
    device_type:     str   = Field(..., example="mobile",
                                   description="desktop | mobile | tablet")
    ip_country:      str   = Field(..., example="US")
    promo_used:      int   = Field(..., example=0, description="0 or 1")
    order_subtotal:  float = Field(..., example=299.99)
    shipping_fee:    float = Field(..., example=14.99)
    tax_amount:      float = Field(..., example=24.00)
    order_total:     float = Field(..., example=338.98)
    risk_score:      float = Field(..., example=22.5)

    # Customer fields
    gender:              str = Field(..., example="Female")
    birthdate:           str = Field(..., example="1990-06-15",
                                     description="YYYY-MM-DD")
    customer_segment:    str = Field(..., example="standard")
    loyalty_tier:        str = Field(..., example="silver")
    customer_created_at: str = Field(..., example="2025-10-01 09:00:00",
                                     description="ISO datetime account was created")
    customer_zip:        str = Field(..., example="10001")

    # Order-item aggregates (default to 1 if unknown)
    item_count:       int = Field(1, example=3)
    total_quantity:   int = Field(1, example=4)
    unique_products:  int = Field(1, example=2)


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_fraud:   bool
    risk_level:        str   = Field(..., description="low | medium | high")
    model_version:     str   = "1.0.0"
    scored_at:         str


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Scores e-commerce orders for fraud risk using a GradientBoosting pipeline.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python model.py` to train and serialize it first.",
        )

    # Build a one-row DataFrame from the raw input
    raw = pd.DataFrame([order.model_dump()])

    # Apply the same feature engineering used during training
    features = engineer_features(raw)

    # Ensure every expected column is present (fill missing with NaN for imputer)
    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = np.nan

    X = features[FEATURE_COLS]

    prob = float(_model.predict_proba(X)[0, 1])
    pred = bool(_model.predict(X)[0])

    if prob < 0.30:
        risk_level = "low"
    elif prob < 0.70:
        risk_level = "medium"
    else:
        risk_level = "high"

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        predicted_fraud=pred,
        risk_level=risk_level,
        scored_at=datetime.utcnow().isoformat(),
    )
