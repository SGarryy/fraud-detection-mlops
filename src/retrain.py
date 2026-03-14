import json
import subprocess
from src.drift_detect import generate_drift_report


DRIFT_THRESHOLD = 0.3


def check_drift_and_retrain():
    print("Running drift detection...")
    drift_report = generate_drift_report()
    print("Drift report generated.")
    print("Checking drift threshold...")
    print(f"Threshold set to {DRIFT_THRESHOLD}")
    
    drifted_features = drift_report.get("drifted_features", 0)
    total_features = drift_report.get("total_features", 1)
    drift_ratio = drifted_features / total_features
    
    if drift_ratio > DRIFT_THRESHOLD:
        print(f"Drift detected ({drift_ratio:.2%} features drifted). Triggering retrain...")
        subprocess.run(["python", "src/train.py"])
        print("Retrain complete.")
    else:
        print(f"No significant drift detected ({drift_ratio:.2%} < {DRIFT_THRESHOLD}). Skipping retrain.")


if __name__ == "__main__":
    check_drift_and_retrain()