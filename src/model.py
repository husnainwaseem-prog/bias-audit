"""Classifier + preprocessing pipeline.

Model: RandomForest (200 trees) — strong on tabular data, and SHAP's
TreeExplainer gives fast exact SHAP values for it. Preprocessing is a
sklearn Pipeline (one-hot for categoricals, passthrough for numerics),
so the same object fits, predicts, and feeds the explainer.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, NUMERIC


def build_pipeline(drop_sensitive: bool = False,
                   n_estimators: int = 100, seed: int = 42) -> Pipeline:
    """Features = numerics + categoricals (+ sex/race unless dropped).

    `drop_sensitive` builds the ablation model: the sensitive columns
    are removed from the features entirely (see the audit — proxies
    mean the gap usually survives anyway).
    """
    cats = list(CATEGORICAL)
    if not drop_sensitive:
        cats = cats + ["sex", "race"]
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                 random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def feature_names(pipe: Pipeline) -> list[str]:
    """Human-readable feature names after one-hot encoding."""
    pre = pipe.named_steps["pre"]
    names = list(NUMERIC)
    cat_encoder = pre.named_transformers_["cat"]
    cats = pre.transformers_[1][2]
    for col, cats_in_col in zip(cats, cat_encoder.categories_):
        names += [f"{col}={c}" for c in cats_in_col]
    return names


def fit_predict(pipe: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
                sample_weight=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train, return (train, test) with y_true / y_pred / y_prob."""
    X_train = train.drop(columns=["income"])
    y_train = train["income"]
    X_test = test.drop(columns=["income"])
    y_test = test["income"]
    fit_params = {}
    if sample_weight is not None:
        fit_params["clf__sample_weight"] = sample_weight
    pipe.fit(X_train, y_train, **fit_params)
    for df, X, y in ((train, X_train, y_train), (test, X_test, y_test)):
        pass
    train_out = train.copy()
    train_out["y_pred"] = pipe.predict(X_train)
    train_out["y_prob"] = pipe.predict_proba(X_train)[:, 1]
    test_out = test.copy()
    test_out["y_pred"] = pipe.predict(X_test)
    test_out["y_prob"] = pipe.predict_proba(X_test)[:, 1]
    return train_out, test_out
