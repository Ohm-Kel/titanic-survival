"""Woman-child-group model: passengers who travelled together shared a fate.

The baseline rule is that women survive and men do not. Two exceptions are
learned from the training labels of each passenger's travel group:

- a woman whose group's women and boys all died is predicted to die;
- a boy (title Master) whose group's women and boys all survived is predicted
  to survive.

Adult men are left out of the group statistics because almost all of them
died whatever their group did. Groups are matched by ticket number and by
surname, so they link test passengers to their relatives in the training set.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from titanic.features import extract_title


def group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ticket and family keys used to match travel groups.

    The family key combines surname, class and all but the last ticket digit,
    so relatives on consecutive tickets match while unrelated passengers who
    share a common surname do not.
    """
    surname = df["Name"].str.split(",").str[0].str.strip()
    ticket = df["Ticket"].astype(str)
    return pd.DataFrame(
        {
            "ticket": ticket,
            "family": surname + "_" + df["Pclass"].astype(str) + "_" + ticket.str[:-1],
        },
        index=df.index,
    )


def is_woman_or_boy(df: pd.DataFrame) -> pd.Series:
    """Return True for women and for boys (title Master)."""
    return (df["Sex"] == "female") | (extract_title(df["Name"]) == "Master")


class WomanChildGroupClassifier(ClassifierMixin, BaseEstimator):
    """Predict survival from sex, then correct it with group survival."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WomanChildGroupClassifier":
        """Record how the women and boys of each training group fared."""
        members = is_woman_or_boy(X).to_numpy()
        keys = group_keys(X)[members].assign(Survived=np.asarray(y)[members])
        self.group_rates_ = {
            key: keys.groupby(key)["Survived"].mean() for key in ("ticket", "family")
        }
        self.classes_ = np.array([0, 1])
        return self

    def group_rate(self, X: pd.DataFrame) -> pd.Series:
        """Return each passenger's group survival rate (NaN if no known group)."""
        keys = group_keys(X)
        rates = pd.concat(
            [keys[key].map(rates) for key, rates in self.group_rates_.items()], axis=1
        )
        return rates.mean(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1 for predicted survivors and 0 otherwise."""
        rate = self.group_rate(X)
        female = X["Sex"] == "female"
        boy = extract_title(X["Name"]) == "Master"
        survived = female & (rate != 0)
        survived |= boy & (rate == 1)
        return survived.astype(int).to_numpy()
