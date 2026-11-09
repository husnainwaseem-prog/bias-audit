"""Mitigation: reweighing (pre-processing).

Reweighing (Kamiran & Calders, 2012) gives each training example a
weight so that the protected attribute and the label become
independent in the weighted training set:

    w(group, label) = P(group) * P(label) / P(group, label)

Groups that are under-represented for a label get up-weighted, and
over-represented ones get down-weighted. The model is then trained
with these sample weights — no change to the model itself, no
threshold hacking, and the sensitive attribute stays in the features
(removing it alone does not work; see the audit).

Also provided: `drop_sensitive_weights` is not needed — the ablation
model simply drops the columns (see model.build_pipeline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reweighing_weights(train: pd.DataFrame, sensitive_col: str = "sex") -> np.ndarray:
    """Sample weights that make `sensitive_col` independent of the label."""
    n = len(train)
    weights = np.ones(n, dtype=float)
    groups = train[sensitive_col].unique()
    labels = [0, 1]
    for g in groups:
        p_group = (train[sensitive_col] == g).mean()
        for y in labels:
            mask = (train[sensitive_col] == g) & (train["income"] == y)
            p_joint = mask.mean()
            if p_joint > 0:
                weights[mask.to_numpy()] = p_group * (train["income"] == y).mean() / p_joint
    return weights


def combined_weights(train: pd.DataFrame) -> np.ndarray:
    """Reweighing on sex AND race: multiply the two weight sets, then
    rescale so the mean weight is 1 (keeps the loss scale stable)."""
    w = reweighing_weights(train, "sex") * reweighing_weights(train, "race")
    return w / w.mean()
