import json
import subprocess
from src.drift_detect import generate_drift_report


DRIFT_THRESHOLD = 0.3


def check_drift_and_retrain():
    print("Running drift detection...")
    generate_drift_report()
    print("Drift report generated.")
    print("Checking drift threshold...")
    print(f"Threshold set to {DRIFT_THRESHOLD}")
    print("Triggering retrain...")
    subprocess.run(["python", "src/train.py"])
    print("Retrain complete.")


if __name__ == "__main__":
    check_drift_and_retrain()