import os
import pickle

import loguru
import mlflow
from sklearn.base import BaseEstimator

from diamonds.params import MODEL_PATH

logger = loguru.logger

def save_model(model: BaseEstimator, name: str, model_registry: str = "local") -> None:
    """Save a trained model to disk and return its path."""
    model_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    if model_registry == "mlflow":
        mlflow.sklearn.log_model(model, name)
        logger.info(f"Model saved to MLflow registry with name: {name}")


def load_model(name: str, model_registry: str = "local") -> BaseEstimator:
    """Load a previously saved model from disk."""
    model_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    if model_registry == "mlflow":
        model_uri = f"models:/{name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from MLflow registry with name: {name}")
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from local with name: {name}")

    return model
