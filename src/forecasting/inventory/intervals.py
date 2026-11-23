"""Model-agnostic prediction intervals for cumulative demand over the protection period.

Residuals of earlier backtest folds give the error of the cumulative forecast over
``R + L`` days. Errors are divided by an item scale (in-sample naive MAE) so items of
very different volume can be pooled per demand class, the class quantile is taken,
and it is multiplied back by the item scale. Only folds *before* the evaluated
period are used for calibration (split-conformal style, no leakage).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calibration_folds(period: str, ordered_names: Sequence[str]) -> list[str]:
    """Folds whose residuals may calibrate ``period``: all folds strictly before it."""
    names = list(ordered_names)
    idx = names.index(period)
    if idx == 0:
        raise ValueError(f"{period} is the first fold: no earlier residuals to calibrate on")
    return names[:idx]


def cumulative_windows(preds: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sum actuals and forecasts over every run of ``window`` consecutive days within a fold.

    Closed days count as zero demand and zero forecast.

    Returns:
        DataFrame with ``fold, series_id, start_d, y_cum, f_cum``.
    """
    p = preds.assign(
        y=np.where(preds["is_closed"] == 1, 0.0, preds["y_true"].astype(float)),
        f=np.where(preds["is_closed"] == 1, 0.0, preds["y_pred"].astype(float)),
        series_id=preds["series_id"].astype(str),
    )
    parts = []
    for fold, g in p.groupby("fold", sort=False):
        y = g.pivot(index="series_id", columns="d", values="y").sort_index(axis=1)
        f = g.pivot(index="series_id", columns="d", values="f").reindex_like(y)
        if y.shape[1] < window:
            continue
        cy = np.nancumsum(np.c_[np.zeros(len(y)), y.to_numpy()], axis=1)
        cf = np.nancumsum(np.c_[np.zeros(len(f)), f.to_numpy()], axis=1)
        y_cum = cy[:, window:] - cy[:, :-window]
        f_cum = cf[:, window:] - cf[:, :-window]
        starts = y.columns.to_numpy()[: y_cum.shape[1]]
        parts.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "series_id": np.repeat(y.index.to_numpy(), len(starts)),
                    "start_d": np.tile(starts, len(y)),
                    "y_cum": y_cum.ravel(),
                    "f_cum": f_cum.ravel(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def normalized_errors(windows: pd.DataFrame, scales: pd.Series, classes: pd.Series) -> pd.DataFrame:
    """Add ``e_norm = (y_cum - f_cum) / scale`` and ``demand_class``; drops zero/NaN scales."""
    sid = windows["series_id"].astype(str)
    scale = sid.map(scales).astype(float)
    out = windows.assign(
        demand_class=sid.map(classes).to_numpy(),
        e_norm=(windows["y_cum"] - windows["f_cum"]) / scale.replace(0, np.nan),
    )
    return out.dropna(subset=["e_norm", "demand_class"])


def calibrate_quantiles(errors: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    """Quantile of the normalised cumulative error per demand class (rows) and level (cols)."""
    q = errors.groupby("demand_class", observed=True)["e_norm"].quantile(list(quantiles))
    return q.unstack()


def safety_stock(
    quantiles: pd.DataFrame, level: float, scales: pd.Series, classes: pd.Series
) -> pd.Series:
    """Safety stock per item (units) = class quantile at ``level`` x item scale."""
    q = classes.map(quantiles[level]).astype(float)
    return (q * scales.reindex(classes.index).astype(float)).fillna(0.0).rename("safety_stock")


def interval_coverage(
    windows: pd.DataFrame,
    quantiles: pd.DataFrame,
    scales: pd.Series,
    classes: pd.Series,
    levels: Sequence[float],
) -> pd.DataFrame:
    """Share of windows whose actual cumulative demand is <= forecast + safety stock."""
    sid = windows["series_id"].astype(str)
    w = windows.assign(demand_class=sid.map(classes).to_numpy())
    rows = []
    for level in levels:
        ss = sid.map(safety_stock(quantiles, level, scales, classes)).fillna(0.0)
        hit = (w["y_cum"] <= w["f_cum"] + ss).astype(float)
        by = hit.groupby(w["demand_class"]).mean()
        rows.append(pd.DataFrame({"demand_class": by.index, "level": level, "coverage": by.values}))
        rows.append(
            pd.DataFrame(
                {"demand_class": ["semua"], "level": level, "coverage": [float(hit.mean())]}
            )
        )
    return pd.concat(rows, ignore_index=True)
