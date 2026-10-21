"""Compare models on the same pipeline with stratified cross-validation."""

import logging

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from titanic.data import load_train
from titanic.pipeline import build_pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5


def get_models() -> dict[str, ClassifierMixin]:
    """Return the candidate models to compare, keyed by display name."""
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_cv(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[float, float]:
    """Return mean and standard deviation of accuracy across stratified folds."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def women_survive_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    """Return the accuracy of predicting survival for women only."""
    return ((X["Sex"] == "female").astype(int) == y).mean()


def main() -> None:
    """Compare every candidate model against the women-survive baseline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X, y = load_train()

    logger.info("%-20s %.3f", "Baseline (women)", women_survive_baseline(X, y))
    for name, model in get_models().items():
        mean, std = evaluate_cv(build_pipeline(model), X, y)
        logger.info("%-20s %.3f ± %.3f", name, mean, std)


if __name__ == "__main__":
    main()