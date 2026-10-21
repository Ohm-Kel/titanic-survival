"""Tests for the woman-child-group model."""

import pandas as pd

from titanic.groups import WomanChildGroupClassifier, group_keys


def passengers(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Return passengers from (name, sex, ticket) tuples, all in third class."""
    return pd.DataFrame(rows, columns=["Name", "Sex", "Ticket"]).assign(Pclass=3)


TRAIN = passengers(
    [
        ("Doe, Mrs. Jane", "female", "111"),
        ("Doe, Master. Tom", "male", "111"),
        ("Roe, Mrs. Ann", "female", "222"),
        ("Roe, Master. Sam", "male", "222"),
        ("Loe, Mr. Mark", "male", "333"),
    ]
)
SURVIVED = pd.Series([0, 0, 1, 1, 1])


def test_family_key_matches_consecutive_tickets():
    keys = group_keys(
        passengers([("Doe, Mr. A", "male", "3101"), ("Doe, Mrs. B", "female", "3102")])
    )
    assert keys["family"].nunique() == 1
    assert keys["ticket"].nunique() == 2


def test_predicts_by_sex_without_a_known_group():
    model = WomanChildGroupClassifier().fit(TRAIN, SURVIVED)
    test = passengers(
        [("New, Miss. Eve", "female", "999"), ("New, Master. Ed", "male", "999")]
    )
    assert model.predict(test).tolist() == [1, 0]


def test_woman_follows_a_group_that_died():
    model = WomanChildGroupClassifier().fit(TRAIN, SURVIVED)
    test = passengers([("Doe, Miss. Kim", "female", "111")])
    assert model.predict(test).tolist() == [0]


def test_boy_follows_a_group_that_survived():
    model = WomanChildGroupClassifier().fit(TRAIN, SURVIVED)
    test = passengers([("Roe, Master. Ben", "male", "222")])
    assert model.predict(test).tolist() == [1]


def test_adult_men_are_ignored_in_group_rates():
    model = WomanChildGroupClassifier().fit(TRAIN, SURVIVED)
    test = passengers([("Loe, Master. Joe", "male", "333")])
    assert model.predict(test).tolist() == [0]
