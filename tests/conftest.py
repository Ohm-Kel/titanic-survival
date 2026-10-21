"""Shared pytest fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from titanic.data import load_train

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample() -> tuple[pd.DataFrame, pd.Series]:
    """Return 12 hand-written passengers in Kaggle's train.csv format."""
    return load_train(FIXTURES)