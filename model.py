"""
model.py — Training script and shared inference utilities.

Run once to produce model.pkl:
    python model.py [path/to/shop.db]

The engineer_features() function and FEATURE_COLS / NUM_COLS / CAT_COLS
are imported by main.py to ensure training and inference use identical logic.
"""
import os
import sys
import warnings

import psycopg2
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED      = 27
LABEL     = "is_fraud"
REF_YEAR  = 2026          # fixed reference year for customer_age calculation
MODEL_PATH = Path(__file__).parent / "model.pkl"

# Feature columns — must stay in sync with engineer_features() output.
# Numeric first, then categorical (matches ColumnTransformer order).
NUM_COLS = [
    "customer_id", "promo_used", "order_subtotal", "shipping_fee",
    "tax_amount", "order_total", "risk_score", "item_count",
    "total_quantity", "unique_products", "order_month", "is_weekend",
    "account_age_days", "zip_mismatch", "ip_foreign",
]
CAT_COLS = [
    "billing_zip", "shipping_zip", "shipping_state", "payment_method",
    "device_type", "ip_country", "gender", "customer_segment",
    "loyalty_tier", "customer_zip",
]
FEATURE_COLS = NUM_COLS + CAT_COLS


# ── Feature engineering (identical in training AND inference) ─────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw order+customer fields into model-ready features.
    This function is the single source of truth — imported by main.py.
    Any change here must be reflected in training and re-serialised.
    """
    df = df.copy()

    # Parse datetimes
    df["order_datetime"]      = pd.to_datetime(df["order_datetime"],      errors="coerce")
    df["birthdate"]           = pd.to_datetime(df["birthdate"],           errors="coerce")
    df["customer_created_at"] = pd.to_datetime(df["customer_created_at"], errors="coerce")

    # Time-based signals
    df["order_month"]      = df["order_datetime"].dt.month
    df["order_weekday"]    = df["order_datetime"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]       = (df["order_weekday"] >= 5).astype(int)

    # Customer tenure at time of order (negative if dates are swapped → imputer handles it)
    df["account_age_days"] = (df["order_datetime"] - df["customer_created_at"]).dt.days

    # Binary flags
    df["zip_mismatch"] = (df["billing_zip"] != df["shipping_zip"]).astype(int)
    df["ip_foreign"]   = (df["ip_country"] != "US").astype(int)

    # Drop raw datetime columns — not directly usable by the model
    df.drop(
        columns=["order_datetime", "birthdate", "customer_created_at", "order_weekday"],
        errors="ignore",
        inplace=True,
    )
    return df


# ── Data loading ──────────────────────────────────────────────────────────────
def load_training_data(db_url: str) -> pd.DataFrame:
    conn = psycopg2.connect(db_url)
    query = """
    SELECT
        o.order_id,
        o.customer_id,
        o.order_datetime,
        o.billing_zip,
        o.shipping_zip,
        o.shipping_state,
        o.payment_method,
        o.device_type,
        o.ip_country,
        o.promo_used,
        o.order_subtotal,
        o.shipping_fee,
        o.tax_amount,
        o.order_total,
        o.risk_score,
        o.is_fraud,
        c.gender,
        c.birthdate,
        c.customer_segment,
        c.loyalty_tier,
        c.created_at  AS customer_created_at,
        c.zip_code    AS customer_zip,
        COALESCE(oi.item_count,    0) AS item_count,
        COALESCE(oi.total_qty,     0) AS total_quantity,
        COALESCE(oi.unique_prods,  0) AS unique_products
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN (
        SELECT order_id,
               COUNT(*)                   AS item_count,
               SUM(quantity)              AS total_qty,
               COUNT(DISTINCT product_id) AS unique_prods
        FROM   order_items
        GROUP  BY order_id
    ) oi ON o.order_id = oi.order_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ── Preprocessor factory (used in training; recreated identically each run) ───
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,     NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ],
        remainder="drop",
    )


# ── Training entry point ──────────────────────────────────────────────────────
def train(db_path: str) -> Pipeline:
    print(f"Loading data from {db_path} …")
    df_raw = load_training_data(db_path)
    df     = engineer_features(df_raw)

    y = df[LABEL].astype(int)
    X = df[FEATURE_COLS]

    print(f"  {len(X):,} rows | fraud rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = Pipeline([
        ("prep", build_preprocessor()),
        ("gbdt", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=SEED,
        )),
    ])

    # Class weights to handle 6.4% fraud imbalance — GradientBoostingClassifier
    # doesn't accept class_weight, so we use sample_weight in fit() instead.
    fraud_rate = y_train.mean()
    sample_weight = y_train.map({0: fraud_rate, 1: 1 - fraud_rate}).values

    print("Training …")
    model.fit(X_train, y_train, gbdt__sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  F1      : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall  : {recall_score(y_test, y_pred, zero_division=0):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return model


if __name__ == "__main__":
    db_url = sys.argv[1] if len(sys.argv) > 1 else os.environ["DATABASE_URL"]
    train(db_url)
