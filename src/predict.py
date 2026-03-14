import joblib
import numpy as np
import pandas as pd

MODEL_PATH  = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_transaction(features: dict) -> dict:
    df = pd.DataFrame([features])
    df['Amount'] = scaler.transform(df[['Amount']])
    df['Time']   = scaler.transform(df[['Time']])
    prediction   = model.predict(df)[0]
    probability  = model.predict_proba(df)[0][1]
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }