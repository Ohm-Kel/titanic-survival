"""Engineered features derived from the raw Titanic columns."""

import pandas as pd

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def extract_title(names: pd.Series) -> pd.Series:
    """Return the title from names like 'Braund, Mr. Owen Harris'.

    French and variant spellings are mapped to their English equivalent, and
    uncommon titles (Dr, Rev, Col, Countess...) are grouped as 'Rare'.
    """
    titles = (
        names.str.extract(r",\s*([^.]+)\.", expand=False)
        .str.strip()
        .replace(TITLE_ALIASES)
    )
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with Title, FamilySize, IsAlone and HasCabin added."""
    out = df.copy()
    out["Title"] = extract_title(out["Name"])
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    return out