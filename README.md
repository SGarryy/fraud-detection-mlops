# Fraud Detection MLOps Pipeline

End-to-end machine learning system for real-time credit card fraud detection with experiment tracking, drift detection, and containerized deployment.

## Architecture
- **Model**: Random Forest with SMOTE (AUC-ROC: 0.98)
- **Tracking**: MLflow experiment tracking
- **API**: FastAPI REST endpoint
- **Monitoring**: Evidently AI drift detection
- **Deployment**: Docker container

## Project Structure
```
fraud-detection-mlops/
├── data/          # Dataset (see data/README.md)
├── notebooks/     # EDA notebook
├── src/           # Core pipeline
│   ├── train.py
│   ├── predict.py
│   ├── drift_detect.py
│   └── retrain.py
├── api/           # FastAPI app
├── models/        # Saved model
├── reports/       # Drift + evaluation reports
└── Dockerfile
```

## Quick Start
```bash
git clone https://github.com/SGarryy/fraud-detection-mlops
cd fraud-detection-mlops
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload --port 8000
```

## API Usage
```bash
POST http://localhost:8000/predict
```

## Run with Docker
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in `data/`