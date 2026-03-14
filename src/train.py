import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ────────────────────────────────────────────────
DATA_PATH   = "data/creditcard.csv"
MODEL_PATH  = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
REPORTS_DIR = "reports"


def load_and_preprocess(path: str):
    """Load CSV, scale Amount+Time, return X and y."""
    df = pd.read_csv(path)

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time']   = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y, scaler


def apply_smote(X_train, y_train):
    """Fix class imbalance using SMOTE oversampling."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — fraud: {y_resampled.sum()}, "
          f"legit: {(y_resampled == 0).sum()}")
    return X_resampled, y_resampled


def plot_confusion_matrix(y_test, y_pred):
    """Save confusion matrix plot to reports/."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues',
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved.")


def train():
    """Full training pipeline with MLflow tracking."""

    print("Loading data...")
    X, y, scaler = load_and_preprocess(DATA_PATH)

    # Train/test split BEFORE SMOTE (important — never SMOTE test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("Applying SMOTE...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── MLflow experiment ────────────────────────────────
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="random-forest-v1"):

        # Model config
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1
        }
        mlflow.log_params(params)

        print("Training Random Forest...")
        model = RandomForestClassifier(**params)
        model.fit(X_train_sm, y_train_sm)

        # Evaluate
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        f1      = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_prob)

        print(f"\nF1 Score  : {f1:.4f}")
        print(f"AUC-ROC   : {auc_roc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Legit', 'Fraud']))

        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)

        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(f"{REPORTS_DIR}/confusion_matrix.png")

        # Save model + scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        mlflow.log_artifact(MODEL_PATH)

        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train()