"""Tests for data loading."""

from titanic.data import TARGET


def test_load_train_separates_target(sample):
    X, y = sample
    assert TARGET not in X.columns
    assert len(X) == len(y) == 12
    assert set(y.unique()) == {0, 1}