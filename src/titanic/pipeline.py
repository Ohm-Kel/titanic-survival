"""Leakage-safe preprocessing and model pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Pclass", "Embarked"]


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and encode the raw columns; drop everything else."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Return the full preprocessing + logistic regression pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )