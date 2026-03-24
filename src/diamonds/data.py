# Import other necessary libraries here
import os

import loguru
import pandas as pd
import seaborn as sns

from diamonds.model import create_preproc
from diamonds.params import DATA_PATH
from diamonds.registry import load_model, save_model

df_diamonds = sns.load_dataset("diamonds")

logger = loguru.logger

def load_data() -> pd.DataFrame:
    df_diamonds = sns.load_dataset("diamonds")
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    logger.info("Loading diamonds dataset...")
    csv_path = os.path.join(DATA_PATH, "raw", "diamonds.csv")
    if not os.path.exists(csv_path):
        logger.info("Caching the diamonds dataset...")
        df_diamonds = sns.load_dataset("diamonds")
        df_diamonds.to_csv(csv_path, index=False)
    else:
        logger.info("Loading diamonds dataset from cache...")
        df_diamonds = pd.read_csv(csv_path)
    return df_diamonds


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    rows = len(df)

    def keep_not_allow(rows):
        if 0 in rows.values:
            return False
        return True

    df_clean = df[df.apply(keep_not_allow, axis=1)]
    logger.info(f"Cleaned the diamonds dataset: {rows} rows -> {len(df_clean)} rows")

    return df_clean


def preprocess_data(df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """

    if train:
        preprocessor = create_preproc()
        preprocessor.fit(df)
        save_model(preprocessor, "preprocessor")

    else:
        preprocessor = load_model("preprocessor")

    df_preprocessed = preprocessor.transform(df)
    logger.info(f"Preprocessed the diamonds dataset: {df.shape} -> {df_preprocessed.shape}")
    return df_preprocessed


def create_X_y(df: pd.DataFrame, predict_value: str = "price") -> tuple[pd.DataFrame, pd.Series]:
    # Split target first so preprocessing columns only reference feature columns
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """

    X = df.drop(columns=predict_value)
    y = df[predict_value]

    return X, y


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
