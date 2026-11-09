"""Fairness metrics, computed per group and as gaps.

Per group (sex, race): accuracy, selection rate (share predicted
positive), true positive rate, false positive rate.

Gap metrics (the audit's headline numbers):
- Demographic parity difference: max |selection rate gap| between groups.
- Equal opportunity difference: max |TPR gap| between groups.
- Equalized odds difference: max over groups of max(|TPR gap|, |FPR gap|).
All gaps are 0 = perfectly fair, 1 = worst. A common working rule is
that gaps under 0.10 are small; the audit reports the numbers and lets
the reader judge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per group with accuracy / selection rate / TPR / FPR."""
    rows = []
    for group, g in df.groupby(group_col):
        y, p = g["income"].to_numpy(), g["y_pred"].to_numpy()
        tp = int(((y == 1) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        rows.append({
            "group": group,
            "n": len(g),
            "base_rate": round(float(y.mean()), 4),
            "accuracy": round(float((tp + tn) / max(len(g), 1)), 4),
            "selection_rate": round(float(p.mean()), 4),
            "tpr": round(float(tp / max(tp + fn, 1)), 4),
            "fpr": round(float(fp / max(fp + tn, 1)), 4),
        })
    return pd.DataFrame(rows)


def gap_metrics(group_df: pd.DataFrame) -> dict[str, float]:
    """Max pairwise gaps across groups for the three headline metrics."""
    def max_gap(col: str) -> float:
        vals = group_df[col].to_numpy()
        return round(float(vals.max() - vals.min()), 4) if len(vals) > 1 else 0.0
    dp = max_gap("selection_rate")
    eo = max_gap("tpr")
    eod = round(max(max_gap("tpr"), max_gap("fpr")), 4)
    return {
        "demographic_parity_diff": dp,
        "equal_opportunity_diff": eo,
        "equalized_odds_diff": eod,
    }


def overall_accuracy(df: pd.DataFrame) -> float:
    return round(float((df["income"] == df["y_pred"]).mean()), 4)


def fairness_report(df: pd.DataFrame) -> dict:
    """Full report: overall accuracy + per-group tables + gaps, for sex and race."""
    df = df.copy()
    df["race_group"] = df["race"].apply(
        lambda r: "White" if r == "White" else ("Black" if r == "Black" else "Other"))
    sex = group_metrics(df, "sex")
    race = group_metrics(df, "race_group")
    return {
        "accuracy": overall_accuracy(df),
        "by_sex": sex.to_dict("records"),
        "by_race": race.to_dict("records"),
        "gaps_sex": gap_metrics(sex),
        "gaps_race": gap_metrics(race),
    }
