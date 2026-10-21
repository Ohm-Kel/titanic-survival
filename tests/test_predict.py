"""Tests for the Kaggle submission format."""

from titanic.pipeline import build_pipeline
from titanic.predict import make_submission


def test_submission_matches_kaggle_format(sample):
    X, y = sample
    model = build_pipeline().fit(X, y)
    submission = make_submission(model, X)
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == len(X)
    assert submission["Survived"].isin([0, 1]).all()