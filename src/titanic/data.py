"""Load the Kaggle Titanic competition data."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
TARGET = "Survived"


def load_train(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return the training features and the survival target."""
    df = pd.read_csv(raw_dir / "train.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return the Kaggle test features (no target column)."""
    return pd.read_csv(raw_dir / "test.csv")