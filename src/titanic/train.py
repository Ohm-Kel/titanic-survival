"""Tune gradient boosting, evaluate it out-of-fold and save the final model."""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from titanic.data import load_train
from titanic.pipeline import build_pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5
MODEL_PATH = Path("models/model.joblib")
FIGURE_PATH = Path("reports/figures/confusion_matrix.png")
CLASS_NAMES = ["Died", "Survived"]

# "model__" targets the pipeline step named "model".
PARAM_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
}


def make_cv() -> StratifiedKFold:
    """Return the stratified 5-fold splitter used everywhere in the project."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def tune(X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """Try every parameter combination with cross-validation and keep the best."""
    search = GridSearchCV(build_pipeline(), PARAM_GRID, cv=make_cv(), scoring="accuracy")
    search.fit(X, y)
    return search


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
    """Tune, evaluate and save the final gradient boosting pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X, y = load_train()

    search = tune(X, y)
    std = search.cv_results_["std_test_score"][search.best_index_]
    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Tuned CV accuracy: %.3f ± %.3f", search.best_score_, std)

    y_pred = cross_val_predict(search.best_estimator_, X, y, cv=make_cv())
    report = classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3)
    logger.info("Out-of-fold classification report:\n%s", report)
    save_confusion_matrix(y, y_pred)
    logger.info("Saved confusion matrix to %s", FIGURE_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()