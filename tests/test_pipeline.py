"""Tests for the end-to-end pipeline."""

import numpy as np

from titanic.pipeline import build_pipeline


def test_pipeline_predicts_zero_or_one_for_every_row(sample):
    X, y = sample
    predictions = build_pipeline().fit(X, y).predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_pipeline_handles_missing_values_at_prediction(sample):
    X, y = sample
    model = build_pipeline().fit(X, y)
    unseen = X.head(3).copy()
    unseen[["Age", "Fare", "Embarked"]] = np.nan
    assert len(model.predict(unseen)) == 3


def test_pipeline_handles_unseen_category(sample):
    X, y = sample
    model = build_pipeline().fit(X, y)
    unseen = X.head(1).copy()
    unseen["Embarked"] = "X"
    assert len(model.predict(unseen)) == 1