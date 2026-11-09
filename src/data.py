"""Load and clean the UCI Adult Income dataset.

Target: income > $50K/year (1) vs <= $50K (0).
Sensitive attributes: sex and race (kept in the frame for the audit,
never silently dropped — the audit compares models with and without them).

Cleaning: strip whitespace, "?" -> NaN, drop rows missing workclass /
occupation / native-country (the standard treatment), one-hot encode
categoricals, standard-scale numerics inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain",
           "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "native-country"]
SENSITIVE = ["sex", "race"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS, header=None if "test" not in path.name else 0,
                     skipinitialspace=True, na_values="?", engine="python")
    # adult.test has a header row and labels end with "."
    df = df[df["age"].notna()].copy()
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return df


def load_adult(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) using the official UCI split."""
    train = _read(Path(data_dir) / "adult.data")
    test = _read(Path(data_dir) / "adult.test")
    return clean(train), clean(test)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["workclass", "occupation", "native-country"])
    df["income"] = (df["income"] == ">50K").astype(int)
    return df.reset_index(drop=True)


def race_group(race: str) -> str:
    """Collapse race to White / Black / Other for readable group tables."""
    if race == "White":
        return "White"
    if race == "Black":
        return "Black"
    return "Other"
