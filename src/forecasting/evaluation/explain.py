"""SHAP explanations for the fitted LightGBM model (TreeSHAP on a row sample)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.models.lgbm import LGBMModel


def shap_sample(
    model: LGBMModel, frame: pd.DataFrame, n_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """TreeSHAP values for a random sample of open-day rows of ``frame``.

    Values are in the model's raw score space (log of the mean for tweedie/poisson),
    so ``base + values.sum(axis=1)`` equals the raw prediction of each row.

    Returns:
        ``(shap_values, X, base_value)`` with one column per model feature.
    """
    import shap

    rows = frame.loc[frame["is_closed"] == 0]
    sample = rows.sample(n=min(n_rows, len(rows)), random_state=seed)
    X = model.feature_matrix(sample)
    explainer = shap.TreeExplainer(model.booster_)
    values = np.asarray(explainer.shap_values(X), dtype="float64")
    base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(values, columns=X.columns, index=X.index), X, base


def mean_abs_shap(values: pd.DataFrame) -> pd.Series:
    """Global importance: mean |SHAP| per feature, largest first."""
    return values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
