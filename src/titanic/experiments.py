"""Measure each engineered feature, then compare models on the full set."""

import logging

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from titanic.data import load_train
from titanic.groups import WomanChildGroupClassifier
from titanic.pipeline import RAW_CATEGORICAL, RAW_NUMERIC, build_pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5

# Each experiment adds one engineered feature group to the raw columns.
FEATURE_EXPERIMENTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Raw columns": (RAW_NUMERIC, RAW_CATEGORICAL),
    "+ Title": (RAW_NUMERIC, (*RAW_CATEGORICAL, "Title")),
    "+ FamilySize/IsAlone": (
        (*RAW_NUMERIC, "FamilySize", "IsAlone"),
        RAW_CATEGORICAL,
    ),
    "+ HasCabin": ((*RAW_NUMERIC, "HasCabin"), RAW_CATEGORICAL),
    "All features": (
        (*RAW_NUMERIC, "FamilySize", "IsAlone", "HasCabin"),
        (*RAW_CATEGORICAL, "Title"),
    ),
}


def get_models() -> dict[str, ClassifierMixin]:
    """Return the candidate models to compare, keyed by display name."""
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_group_models() -> dict[str, ClassifierMixin]:
    """Return models that take raw passenger rows, without the pipeline."""
    return {"Woman-child-group": WomanChildGroupClassifier()}


def evaluate_cv(
    pipeline: Pipeline | ClassifierMixin, X: pd.DataFrame, y: pd.Series
) -> tuple[float, float]:
    """Return mean and standard deviation of accuracy across stratified folds."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def women_survive_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    """Return the accuracy of predicting survival for women only."""
    return ((X["Sex"] == "female").astype(int) == y).mean()


def main() -> None:
    """Run the feature experiments and the model comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X, y = load_train()
    logger.info("%-22s %.3f", "Baseline (women)", women_survive_baseline(X, y))

    logger.info("Feature experiments (gradient boosting):")
    for name, (numeric, categorical) in FEATURE_EXPERIMENTS.items():
        pipeline = build_pipeline(numeric=numeric, categorical=categorical)
        mean, std = evaluate_cv(pipeline, X, y)
        logger.info("  %-22s %.3f ± %.3f", name, mean, std)

    logger.info("Model comparison (selected features: raw + Title):")
    for name, model in get_models().items():
        mean, std = evaluate_cv(build_pipeline(model), X, y)
        logger.info("  %-22s %.3f ± %.3f", name, mean, std)
    for name, model in get_group_models().items():
        mean, std = evaluate_cv(model, X, y)
        logger.info("  %-22s %.3f ± %.3f", name, mean, std)


if __name__ == "__main__":
    main()