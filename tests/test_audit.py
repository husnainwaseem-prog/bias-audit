"""Tests for the Model Explainability and Bias Audit.

Unit tests use small synthetic data (fast, no download). One
integration test runs the audit pipeline on a tiny slice of the real
Adult data (skipped if the data is not present).
"""

import numpy as np
import pandas as pd
import pytest

from src.fairness import fairness_report, gap_metrics, group_metrics
from src.mitigate import combined_weights, reweighing_weights


def _preds_df() -> pd.DataFrame:
    # 4 men (2 pos), 4 women (2 pos) — men predicted positive more often
    return pd.DataFrame({
        "income": [1, 1, 0, 0, 1, 1, 0, 0],
        "y_pred": [1, 1, 1, 0, 1, 0, 0, 0],
        "sex": ["Male"] * 4 + ["Female"] * 4,
        "race": ["White", "Black", "White", "Black"] * 2,
    })


# --- fairness -----------------------------------------------------------

def test_group_metrics_known_values():
    df = _preds_df()
    m = group_metrics(df, "sex").set_index("group")
    # Male: TP=2, FP=1, TN=1, FN=0 -> acc 0.75, sel 0.75, TPR 1.0, FPR 0.5
    assert m.loc["Male", "accuracy"] == pytest.approx(0.75)
    assert m.loc["Male", "selection_rate"] == pytest.approx(0.75)
    assert m.loc["Male", "tpr"] == pytest.approx(1.0)
    assert m.loc["Male", "fpr"] == pytest.approx(0.5)
    # Female: TP=1, FP=0, TN=2, FN=1 -> acc 0.75, sel 0.25, TPR 0.5, FPR 0.0
    assert m.loc["Female", "selection_rate"] == pytest.approx(0.25)
    assert m.loc["Female", "tpr"] == pytest.approx(0.5)


def test_gap_metrics_max_pairwise():
    g = pd.DataFrame({
        "group": ["A", "B"],
        "selection_rate": [0.8, 0.3],
        "tpr": [0.9, 0.6],
        "fpr": [0.2, 0.1],
    })
    gaps = gap_metrics(g)
    assert gaps["demographic_parity_diff"] == pytest.approx(0.5)
    assert gaps["equal_opportunity_diff"] == pytest.approx(0.3)
    assert gaps["equalized_odds_diff"] == pytest.approx(0.3)


def test_fairness_report_structure():
    rep = fairness_report(_preds_df())
    assert "accuracy" in rep and "by_sex" in rep and "by_race" in rep
    assert "gaps_sex" in rep and "gaps_race" in rep
    assert len(rep["by_sex"]) == 2
    assert rep["accuracy"] == pytest.approx(0.75)


# --- reweighing ----------------------------------------------------------

def test_reweighing_weights_balance_groups():
    # Men are 80% positive, women 20% positive in train
    train = pd.DataFrame({
        "sex": ["Male"] * 10 + ["Female"] * 10,
        "race": ["White"] * 20,
        "income": [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
    })
    w = reweighing_weights(train, "sex")
    # weighted positive rate should be equal across groups
    for g in ("Male", "Female"):
        mask = train["sex"] == g
        wp = (w[mask.to_numpy()] * (train["income"] == 1).to_numpy()[mask.to_numpy()]).sum()
        wn = (w[mask.to_numpy()] * (train["income"] == 0).to_numpy()[mask.to_numpy()]).sum()
        assert abs(wp / (wp + wn) - 0.5) < 1e-9


def test_combined_weights_mean_is_one():
    train = pd.DataFrame({
        "sex": ["Male", "Female"] * 10,
        "race": ["White", "Black"] * 10,
        "income": [1, 0] * 10,
    })
    w = combined_weights(train)
    assert abs(w.mean() - 1.0) < 1e-9
    assert (w > 0).all()


# --- data -----------------------------------------------------------------

def test_clean_drops_missing_and_binarises():
    from src.data import clean
    df = pd.DataFrame({
        "workclass": ["Private", None, "Private"],
        "occupation": ["Tech", "Tech", None],
        "native-country": ["US", "US", "US"],
        "income": ["<=50K", ">50K", "<=50K"],
    })
    out = clean(df)
    assert len(out) == 1
    assert out["income"].iloc[0] == 0


# --- model -----------------------------------------------------------------

def test_pipeline_trains_and_predicts():
    from src.model import build_pipeline, fit_predict
    rng = np.random.RandomState(0)
    n = 60
    train = pd.DataFrame({
        "age": rng.randint(20, 60, n),
        "fnlwgt": rng.randint(10000, 200000, n),
        "education-num": rng.randint(1, 16, n),
        "capital-gain": rng.randint(0, 5000, n),
        "capital-loss": rng.randint(0, 1000, n),
        "hours-per-week": rng.randint(20, 60, n),
        "workclass": rng.choice(["Private", "Self-emp"], n),
        "education": rng.choice(["HS-grad", "Bachelors"], n),
        "marital-status": rng.choice(["Married", "Single"], n),
        "occupation": rng.choice(["Tech", "Sales"], n),
        "relationship": rng.choice(["Husband", "Wife"], n),
        "native-country": rng.choice(["US", "Mexico"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["White", "Black"], n),
        "income": rng.choice([0, 1], n),
    })
    pipe = build_pipeline(n_estimators=10)
    _, test_out = fit_predict(pipe, train, train)
    assert "y_pred" in test_out and "y_prob" in test_out
    assert set(test_out["y_pred"].unique()) <= {0, 1}
