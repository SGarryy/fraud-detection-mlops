import joblib
import pandas as pd
from sklearn import set_config

# Disable sklearn's strict feature name validation
set_config(assume_finite=True)

MODEL_PATH = "models/fraud_model.pkl"
SCALER_AMOUNT_PATH = "models/scaler_amount.pkl"
SCALER_TIME_PATH   = "models/scaler_time.pkl"


def predict_transaction(features: dict) -> dict:
    # Load models lazily to allow API to start before training
    model = joblib.load(MODEL_PATH)
    scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
    scaler_time = joblib.load(SCALER_TIME_PATH)
    
    # Define the expected column order (same as during training)
    expected_columns = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    
    # Create DataFrame with ordered columns
    df = pd.DataFrame([{col: features[col] for col in expected_columns}])
    
    # Scale the features
    df[['Amount']] = scaler_amount.transform(df[['Amount']])
    df[['Time']]   = scaler_time.transform(df[['Time']])
    
    # Convert to numpy to avoid feature name validation
    X = df[expected_columns].values.astype('float64')
    
    prediction     = model.predict(X)[0]
    probability    = model.predict_proba(X)[0][1]
    return {
        "is_fraud":          bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level":        "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }