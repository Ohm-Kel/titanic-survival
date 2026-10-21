"""Evaluate the baseline pipeline with stratified cross-validation."""

import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from titanic.data import load_train
from titanic.pipeline import build_pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5


def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """Return mean and standard deviation of accuracy across stratified folds."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def main() -> None:
    """Load the training data and report the baseline CV accuracy."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X, y = load_train()
    mean, std = evaluate_cv(build_pipeline(), X, y)
    logger.info("Baseline logistic regression CV accuracy: %.3f ± %.3f", mean, std)


if __name__ == "__main__":
    main()