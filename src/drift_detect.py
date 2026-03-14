import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

DATA_PATH   = "data/creditcard.csv"
REPORTS_DIR = "reports"


def generate_drift_report():
    df = pd.read_csv(DATA_PATH)
    reference = df.sample(n=5000, random_state=42)
    current   = df.sample(n=5000, random_state=99)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output = f"{REPORTS_DIR}/drift_report.html"
    report.save_html(output)
    print(f"Drift report saved to {output}")
    return output


if __name__ == "__main__":
    generate_drift_report()