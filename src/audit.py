"""Run the full audit: baseline -> ablation -> mitigation, with SHAP
explanations and fairness reports at each step.

Outputs (results/):
  audit.json          — accuracy + fairness gaps for all three models
  plots/shap_importance.png
  plots/shap_beeswarm.png
  plots/shap_waterfall_pos.png / _neg.png  (one local example each way)
  plots/fairness_gaps.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_adult
from .explain import compute_shap, plot_beeswarm, plot_global_importance, plot_waterfall
from .fairness import fairness_report
from .mitigate import combined_weights
from .model import build_pipeline, fit_predict

RESULTS = Path(__file__).resolve().parent.parent / "results"


def plot_fairness_gaps(audit: dict, out: str | Path) -> Path:
    """Grouped bar chart: the three gap metrics for each model."""
    models = list(audit["models"].keys())
    metrics = ["demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_diff"]
    labels = ["Demographic\nparity diff", "Equal opportunity\ndiff", "Equalized odds\ndiff"]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(10, 5))
    for i, m in enumerate(models):
        gaps = audit["models"][m]["gaps_sex"]
        vals = [gaps[metric] for metric in metrics]
        plt.bar(x + (i - 1) * width, vals, width, label=m.replace("_", " "))
    plt.axhline(0.10, color="red", linestyle="--", linewidth=1, label="0.10 rule-of-thumb")
    plt.xticks(x, labels)
    plt.ylabel("Gap (0 = fair)")
    plt.title("Fairness gaps by sex — before vs after mitigation")
    plt.legend()
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return Path(out)


def run_audit(data_dir: str = "data", out_dir: str | Path = RESULTS,
              shap_n: int = 100) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    train, test = load_adult(data_dir)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | "
          f"positive rate: {train['income'].mean():.3f} train / {test['income'].mean():.3f} test")

    audit: dict = {"models": {}}

    # 1. Baseline — sensitive attributes included
    pipe = build_pipeline()
    _, test_base = fit_predict(pipe, train, test)
    audit["models"]["baseline"] = fairness_report(test_base)
    print(f"Baseline accuracy: {audit['models']['baseline']['accuracy']:.4f}")

    # 2. Ablation — sensitive attributes dropped entirely
    pipe_no_sens = build_pipeline(drop_sensitive=True)
    _, test_abl = fit_predict(pipe_no_sens, train, test)
    audit["models"]["no_sensitive_attrs"] = fairness_report(test_abl)
    print(f"No-sensitive-attrs accuracy: {audit['models']['no_sensitive_attrs']['accuracy']:.4f}")

    # 3. Mitigation — reweighing on sex + race
    weights = combined_weights(train)
    pipe_mit = build_pipeline()
    _, test_mit = fit_predict(pipe_mit, train, test, sample_weight=weights)
    audit["models"]["reweighed"] = fairness_report(test_mit)
    print(f"Reweighed accuracy: {audit['models']['reweighed']['accuracy']:.4f}")

    # SHAP on the baseline model (global + two local examples)
    X_test = test.drop(columns=["income"])
    sv, X_enc, names, expected = compute_shap(pipe, X_test, sample_n=shap_n)
    plot_global_importance(sv, names, out_dir / "plots" / "shap_importance.png")
    plot_beeswarm(sv, X_enc, names, out_dir / "plots" / "shap_beeswarm.png")
    # two local examples: the first two people in the SHAP sample
    plot_waterfall(sv, X_enc, names, 0, out_dir / "plots" / "shap_waterfall_0.png",
                   base_value=expected, title="Local explanation — sample person 0")
    plot_waterfall(sv, X_enc, names, 1, out_dir / "plots" / "shap_waterfall_1.png",
                   base_value=expected, title="Local explanation — sample person 1")
    audit["shap_top_features"] = [
        {"feature": names[i], "mean_abs_shap": round(float(np.abs(sv).mean(axis=0)[i]), 4)}
        for i in np.argsort(-np.abs(sv).mean(axis=0))[:10]
    ]
    plot_fairness_gaps(audit, out_dir / "plots" / "fairness_gaps.png")

    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nAudit written: {out_dir / 'audit.json'}")
    for name, m in audit["models"].items():
        g = m["gaps_sex"]
        print(f"  {name:20s} acc {m['accuracy']:.4f}  DP {g['demographic_parity_diff']:.4f}  "
              f"EOpp {g['equal_opportunity_diff']:.4f}  EOdds {g['equalized_odds_diff']:.4f}")
    return audit
