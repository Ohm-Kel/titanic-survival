"""Leakage-safe preprocessing and model pipeline."""

from sklearn.base import ClassifierMixin
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


def build_pipeline(model: ClassifierMixin | None = None) -> Pipeline:
    """Return the preprocessing pipeline followed by the given model.

    Defaults to logistic regression when no model is passed.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])