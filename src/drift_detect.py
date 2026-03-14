import pandas as pd
import numpy as np
import json
import os

DATA_PATH   = "data/creditcard.csv"
REPORTS_DIR = "reports"


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    """Population Stability Index — measures data drift."""
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_perc = np.histogram(expected, np.percentile(expected, breakpoints))[0]
    actual_perc   = np.histogram(actual,   np.percentile(expected, breakpoints))[0]
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc) / len(expected)
    actual_perc   = np.where(actual_perc   == 0, 0.0001, actual_perc)   / len(actual)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return round(float(psi), 4)


def generate_drift_report():
    df        = pd.read_csv(DATA_PATH)
    reference = df.sample(n=5000, random_state=42)
    current   = df.sample(n=5000, random_state=99)

    features      = [c for c in df.columns if c != "Class"]
    drift_results = {}
    drift_detected = 0

    for col in features:
        psi = compute_psi(reference[col].values, current[col].values)
        status = "DRIFT" if psi > 0.2 else "STABLE"
        if status == "DRIFT":
            drift_detected += 1
        drift_results[col] = {"psi": psi, "status": status}

    summary = {
        "total_features": len(features),
        "drifted_features": drift_detected,
        "drift_detected": drift_detected > 0,
        "results": drift_results
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(f"{REPORTS_DIR}/drift_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Drift detected in {drift_detected}/{len(features)} features")
    print(f"Report saved to {REPORTS_DIR}/drift_report.json")
    return summary


if __name__ == "__main__":
    generate_drift_report()