"""Tests for engineered features."""

import pandas as pd

from titanic.features import add_features, extract_title


def test_extract_title_keeps_common_titles():
    names = pd.Series(["Doe, Mr. John", "Roe, Mrs. Jane", "Noe, Master. Tom"])
    assert extract_title(names).tolist() == ["Mr", "Mrs", "Master"]


def test_extract_title_maps_french_titles():
    names = pd.Series(["Zoe, Mlle. Anne", "Doe, Mme. Marie"])
    assert extract_title(names).tolist() == ["Miss", "Mrs"]


def test_extract_title_groups_rare_titles():
    names = pd.Series(["Boe, Rev. Luke", "Yoe, the Countess. of (Rose)"])
    assert extract_title(names).tolist() == ["Rare", "Rare"]


def test_add_features_computes_family_and_cabin(sample):
    X, _ = sample
    out = add_features(X)
    noe = out.loc[out["Name"] == "Noe, Master. Tom"].iloc[0]
    assert noe["FamilySize"] == 5
    assert noe["IsAlone"] == 0
    assert noe["HasCabin"] == 0
    assert out.loc[out["Name"] == "Loe, Mr. Mark", "HasCabin"].iloc[0] == 1


def test_add_features_does_not_modify_input(sample):
    X, _ = sample
    columns_before = list(X.columns)
    add_features(X)
    assert list(X.columns) == columns_before