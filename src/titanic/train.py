"""Evaluate the woman-child-group model out-of-fold and save it."""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from titanic.data import load_train
from titanic.groups import WomanChildGroupClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
MODEL_PATH = Path("models/model.joblib")
FIGURE_PATH = Path("reports/figures/confusion_matrix.png")
CLASS_NAMES = ["Died", "Survived"]

def make_cv() -> StratifiedKFold:
    """Return the stratified 5-fold splitter used everywhere in the project."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_repeated_cv() -> RepeatedStratifiedKFold:
    """Return repeated 5-fold CV, for tighter estimates on 891 passengers."""
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )


def save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, path: Path = FIGURE_PATH
) -> None:
    """Plot the out-of-fold confusion matrix and save it as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, cmap="Greys"
    )
    plt.title("Out-of-fold confusion matrix (5-fold CV)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    """Evaluate, fit and save the final woman-child-group model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X, y = load_train()
    model = WomanChildGroupClassifier()

    scores = cross_val_score(model, X, y, cv=make_repeated_cv(), scoring="accuracy")
    logger.info("Repeated CV accuracy: %.3f ± %.3f", scores.mean(), scores.std())

    y_pred = cross_val_predict(model, X, y, cv=make_cv())
    report = classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3)
    logger.info("Out-of-fold classification report:\n%s", report)
    save_confusion_matrix(y, y_pred)
    logger.info("Saved confusion matrix to %s", FIGURE_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.fit(X, y), MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()