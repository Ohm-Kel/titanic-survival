"""Leakage-safe feature engineering, preprocessing and model pipeline."""

from collections.abc import Sequence

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from titanic.features import add_features

RAW_NUMERIC = ("Age", "Fare", "SibSp", "Parch")
RAW_CATEGORICAL = ("Sex", "Pclass", "Embarked")

# Chosen by ablation: Title improved CV accuracy and stability; FamilySize,
# IsAlone and HasCabin added nothing on top of Title.
NUMERIC_FEATURES = RAW_NUMERIC
CATEGORICAL_FEATURES = (*RAW_CATEGORICAL, "Title")


def build_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str]
) -> ColumnTransformer:
    """Impute, scale and encode the chosen columns; drop everything else."""
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_steps, list(numeric)),
            ("cat", categorical_steps, list(categorical)),
        ]
    )


def build_pipeline(
    model: ClassifierMixin | None = None,
    numeric: Sequence[str] = NUMERIC_FEATURES,
    categorical: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Return feature engineering, preprocessing and the given model.

    Defaults to gradient boosting on the full feature set.
    """
    if model is None:
        model = GradientBoostingClassifier(random_state=42)
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", model),
        ]
    )