"""Forecast accuracy metrics. Closed days (``is_closed == 1``) are always excluded.

Sign convention: error ``e = y_pred - y_true`` so a positive bias means over-forecast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


# ---------------------------------------------------------------- point metrics
def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted absolute percentage error: sum|e| / sum(y)."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = y.sum()
    return float(np.abs(p - y).sum() / denom) if denom > 0 else float("nan")


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Relative bias: sum(e) / sum(y). Positive = over-forecast (stock piles up)."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = y.sum()
    return float((p - y).sum() / denom) if denom > 0 else float("nan")


def pinball(y_true: ArrayLike, q_pred: ArrayLike, q: float) -> float:
    """Mean pinball (quantile) loss for quantile level ``q``."""
    diff = np.asarray(y_true, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of observations inside ``[lower, upper]``."""
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))


# ---------------------------------------------------------------- scale & weights
def compute_scales(train: pd.DataFrame, cutoff_d: int) -> pd.DataFrame:
    """In-sample 1-step naive error per series, from open days with ``d <= cutoff_d``.

    Returns:
        DataFrame indexed by ``series_id`` with ``mse`` (RMSSE scale) and ``mae``
        (MASE scale). Series with fewer than two open days get NaN.
    """
    hist = train.loc[(train["d"] <= cutoff_d) & (train["is_closed"] == 0)]
    hist = hist.assign(series_id=hist["series_id"].astype(str)).sort_values(["series_id", "d"])
    diff = hist.groupby("series_id")["sales"].diff()
    stats = pd.DataFrame({"series_id": hist["series_id"], "sq": diff**2, "abs": diff.abs()})
    out = (
        stats.groupby("series_id")[["sq", "abs"]].mean().rename(columns={"sq": "mse", "abs": "mae"})
    )
    return out.astype("float64")


def revenue_weights(train: pd.DataFrame, cutoff_d: int, window: int) -> pd.Series:
    """Revenue share (units x sell_price) per series over the last ``window`` days."""
    recent = train.loc[train["d"].between(cutoff_d - window + 1, cutoff_d)]
    revenue = (
        (recent["sales"] * recent["sell_price"]).groupby(recent["series_id"].astype(str)).sum()
    )
    total = revenue.sum()
    weights = revenue / total if total > 0 else revenue * np.nan
    weights.index.name = "series_id"
    return weights.rename("weight").astype("float64")


# ---------------------------------------------------------------- scaled errors
def _per_series(preds: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    e = preds["y_pred"].astype(float) - preds["y_true"].astype(float)
    frame = pd.DataFrame({"series_id": preds["series_id"].astype(str), "sq": e**2, "abs": e.abs()})
    return frame.groupby("series_id")


def rmsse_per_series(preds: pd.DataFrame, scales: pd.DataFrame) -> pd.Series:
    """RMSSE = sqrt(MSE_forecast / MSE_naive_in_sample). NaN where the scale is 0 or missing."""
    mse = _per_series(preds)["sq"].mean()
    scale = scales["mse"].reindex(mse.index).replace(0, np.nan)
    return np.sqrt(mse / scale).rename("rmsse")


def mase_per_series(preds: pd.DataFrame, scales: pd.DataFrame) -> pd.Series:
    """MASE = MAE_forecast / MAE_naive_in_sample. NaN where the scale is 0 or missing."""
    mae = _per_series(preds)["abs"].mean()
    scale = scales["mae"].reindex(mae.index).replace(0, np.nan)
    return (mae / scale).rename("mase")


def wrmsse_lite(rmsse: pd.Series, weights: pd.Series) -> float:
    """Revenue-weighted mean RMSSE over scored series (weights re-normalised)."""
    scored = rmsse.dropna()
    w = weights.reindex(scored.index).fillna(0.0)
    if w.sum() <= 0:
        return float("nan")
    return float((scored * w).sum() / w.sum())


# ---------------------------------------------------------------- fold summary
def evaluate_fold(
    preds: pd.DataFrame,
    train: pd.DataFrame,
    cutoff_d: int,
    weight_window: int,
) -> dict[str, Any]:
    """All point metrics for one fold. Closed validation days are dropped first.

    ``wrmsse`` is NaN when ``train`` has no ``sell_price`` (aggregate level).
    """
    p = preds.loc[preds["is_closed"] == 0] if "is_closed" in preds else preds
    scales = compute_scales(train, cutoff_d)
    rmsse = rmsse_per_series(p, scales)
    mase = mase_per_series(p, scales)

    if "sell_price" in train.columns:
        wrmsse = wrmsse_lite(rmsse, revenue_weights(train, cutoff_d, weight_window))
    else:
        wrmsse = float("nan")

    return {
        "wrmsse": wrmsse,
        "rmsse": float(rmsse.mean()),
        "mase": float(mase.mean()),
        "wape": wape(p["y_true"], p["y_pred"]),
        "bias": bias(p["y_true"], p["y_pred"]),
        "n_series": int(p["series_id"].nunique()),
        "n_scored": int(rmsse.notna().sum()),
    }
