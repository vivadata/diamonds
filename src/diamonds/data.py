import pandas as pd
import loguru
import os

# For loading data
import seaborn as sns

from diamonds.params import DATA_PATH
from diamonds.model import create_preproc
from diamonds.registry import save_model, load_model

from sklearn.model_selection import train_test_split

logger = loguru.logger

# Import other necessary libraries here


def load_data() -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    
    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    logger.info("Loading the diamonds dataset...")
    csv_path = os.path.join(DATA_PATH,"raw", "diamonds.csv")
    if not os.path.exists(csv_path):
        logger.info("Caching the diamonds dataset...")
        df = sns.load_dataset("diamonds")
        df.to_csv(csv_path, index=False)
        logger.info("✅ Diamonds dataset cached successfully.")
    else:
        logger.info("Loading the diamonds dataset from cache...")
        df = pd.read_csv(csv_path)
    return df
            
    
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
    def keep_not_null(row) :
        if 0 in row.values : return False
        return True
    df_clean = df[df.apply(keep_not_null,axis=1)]
    logger.info(f"Cleaned the diamonds dataset: {rows} rows -> {len(df_clean)} rows")
    return df_clean


def preprocess_data( X: pd.DataFrame
                    , train: bool = True) -> pd.DataFrame:
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
    # Instantier la pipeline 
    if train : 
        preprocessor = create_preproc()
        preprocessor.fit(X)
        save_model(preprocessor, "preprocessor")
    else :
        preprocessor = load_model("preprocessor")
        df_preprocessed = preprocessor.transform(X)
        logger.info(f"Preprocessed the diamonds dataset: {X.shape} -> {df_preprocessed.shape}") 
    return df_preprocessed

def create_X_y(df: pd.DataFrame) ->tuple[pd.DataFrame, pd.Series]:
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
    
    X = df.drop(columns="price")
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

    return X, y



if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_preprocessed)