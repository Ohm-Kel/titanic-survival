"""Create a Kaggle submission file from the saved model."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from titanic.data import load_test
from titanic.train import MODEL_PATH

logger = logging.getLogger(__name__)

SUBMISSION_PATH = Path("data/submissions/submission.csv")


def make_submission(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame in Kaggle's format: PassengerId, Survived (0 or 1)."""
    return pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Survived": model.predict(test).astype(int),
        }
    )


def main() -> None:
    """Load the saved model, predict test.csv and write the submission file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run titanic-train first.")

    model = joblib.load(MODEL_PATH)
    submission = make_submission(model, load_test())

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info(
        "Wrote %d predictions to %s (predicted survival rate %.3f)",
        len(submission),
        SUBMISSION_PATH,
        submission["Survived"].mean(),
    )


if __name__ == "__main__":
    main()