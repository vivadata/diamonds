import loguru
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diamonds.registry import save_model

logger = loguru.logger

def create_model(
    model_name: str, estimators: int = 200, max_depth: int = 10, random_state: int = 42
) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=estimators, max_depth=max_depth, random_state=random_state
        ),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "linear": LinearRegression(fit_intercept=True),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]


def create_preproc():

    num_pipeline = Pipeline(
        [
            ("num_imp", KNNImputer()),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("cat_imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", num_pipeline, make_column_selector(dtype_include="number")),
            ("categorical", cat_pipeline, make_column_selector(dtype_exclude="number")),
        ]
    ).set_output(transform="pandas")

    return preprocessor


def create_training_pipeline(
    model_name: str,
    estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = create_preproc()
    model = create_model(
        model_name=model_name,
        estimators=estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(model, X_train, y_train):
    """
    Train model and save it.

    Parameters
    ----------
    model : any
        The model to train
    X_train : pd.DataFrame
        The training data
    y_train : pd.Series
        The target variable
    """
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model trained. Saving...")
    save_model(model, "model")
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """
    Evaluate the model on the test set.

    Parameters
    ----------
    model : any
        The model to evaluate
    X_test : pd.DataFrame
        The test data
    y_test : pd.Series
        The target variable

    Returns
    -------
    dict[str, float]
        The metrics for the model.
    """
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {"mae": mae, "mse": mse, "r2": r2, "mape": mape}

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


def predict(model, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    y_pred = model.predict(X)

    return pd.Series(y_pred, index=X.index, name="prediction")
