import os

DATA_PATH = "data"
MODEL_PATH = "models"

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
