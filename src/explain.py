"""SHAP explainability: global and local.

Global: mean |SHAP value| per feature (bar plot) + a beeswarm plot —
which features drive the model overall, and in which direction.
Local: SHAP waterfall plots for individual predictions — which
features pushed this one person over or under the threshold.

Uses shap.TreeExplainer on the fitted RandomForest (fast, exact).
SHAP values are computed on a sample of the test set (2,000 rows) to
keep the plots readable and the runtime short.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .model import feature_names


def compute_shap(pipe, X: pd.DataFrame, sample_n: int = 2000,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Returns (shap_values, X_encoded, feature_names, expected_value)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    X_sample = X.iloc[idx]
    X_enc = pipe.named_steps["pre"].transform(X_sample)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    X_enc = np.asarray(X_enc, dtype=float)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(X_enc)
    # shap 0.54 returns (n_rows, n_features, n_classes) for classifiers;
    # older versions returned a list per class — keep class 1 either way
    if isinstance(sv, list):
        sv = sv[1]
    elif np.asarray(sv).ndim == 3:
        sv = np.asarray(sv)[:, :, 1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(expected[1])
    return np.asarray(sv), X_enc, feature_names(pipe), float(expected)


def plot_global_importance(shap_values: np.ndarray, names: list[str],
                           out: str | Path) -> Path:
    """Mean |SHAP| bar chart, top 15 features."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:15][::-1]
    plt.figure(figsize=(9, 6))
    plt.barh([names[i] for i in order], mean_abs[order])
    plt.xlabel("mean |SHAP value| (impact on log-odds of >50K)")
    plt.title("Global feature importance (SHAP)")
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return Path(out)


def plot_beeswarm(shap_values: np.ndarray, X_enc: np.ndarray,
                  names: list[str], out: str | Path) -> Path:
    """SHAP beeswarm: each dot is one person, colour = feature value."""
    X_df = pd.DataFrame(X_enc, columns=names)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_df, feature_names=names,
                      max_display=15, show=False, plot_size=None)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return Path(out)


def plot_waterfall(shap_values: np.ndarray, X_enc: np.ndarray,
                   names: list[str], row: int, out: str | Path,
                   base_value: float = 0.0, title: str = "") -> Path:
    """Waterfall for one person (row index into the SHAP sample)."""
    plt.figure(figsize=(10, 6))
    exp = shap.Explanation(
        values=shap_values[row],
        base_values=base_value,
        data=X_enc[row],
        feature_names=names,
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return Path(out)
