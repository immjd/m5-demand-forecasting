"""Glue: out-of-sample forecasts -> calibrated safety stock -> (R, S) simulation.

Simulation periods are the CV folds from ``inventory.sim_start_fold`` onwards plus
the test window, each planned with the forecast vintage made at its own cutoff and
safety stock calibrated only on residuals of earlier folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forecasting.config import resolve_path
from forecasting.evaluation.metrics import compute_scales
from forecasting.inventory.intervals import (
    calibrate_quantiles,
    calibration_folds,
    cumulative_windows,
    interval_coverage,
    normalized_errors,
    safety_stock,
)
from forecasting.inventory.policy import order_up_to_levels
from forecasting.inventory.simulate import simulate_lost_sales, summarize_items
from forecasting.validation.backtest import Fold, generate_folds, get_test_fold

ITEM_CLASSES_FILE = "item_classes.parquet"


@dataclass
class PolicyRun:
    """Daily arrays (items x days) of one model at one service level."""

    items: pd.Index
    days: np.ndarray
    demand: np.ndarray
    forecast: np.ndarray
    order_up_to: np.ndarray
    review: np.ndarray
    sim: dict[str, np.ndarray]
    safety_stock: pd.DataFrame  # period x item


def sim_periods(config: dict[str, Any]) -> tuple[list[Fold], list[str]]:
    """Folds simulated (start fold .. last CV fold, then TEST) and all fold names in order."""
    folds = [*generate_folds(config), get_test_fold(config)]
    names = [f.name for f in folds]
    start = names.index(config["inventory"]["sim_start_fold"])
    return folds[start:], names


def item_classes(config: dict[str, Any]) -> pd.Series:
    """Demand class per item; the 2 ``insufficient_history`` items are pooled as intermittent."""
    c = pd.read_parquet(resolve_path(config, "processed") / ITEM_CLASSES_FILE)
    cls = c.set_index(c["item_id"].astype(str))["demand_class"].astype(str)
    return cls.where(cls != "insufficient_history", "intermittent").rename("demand_class")


def load_forecasts(config: dict[str, Any], model: str) -> pd.DataFrame:
    """Out-of-sample item forecasts of every CV fold plus the one-time test run."""
    preds_dir = resolve_path(config, "results") / "preds"
    cv = pd.read_parquet(preds_dir / f"{model}_item.parquet")
    test = pd.read_parquet(preds_dir / f"test_{model}_item.parquet")
    cols = ["fold", "series_id", "d", "y_true", "y_pred", "is_closed"]
    out = pd.concat([cv[cols], test[cols]], ignore_index=True)
    out["series_id"] = out["series_id"].astype(str)
    return out


def _matrix(df: pd.DataFrame, value: str, items: pd.Index, days: np.ndarray) -> np.ndarray:
    m = df.pivot_table(index="series_id", columns="d", values=value, aggfunc="first")
    return m.reindex(index=items, columns=days).to_numpy(dtype=float)


def run_policy(
    config: dict[str, Any],
    preds: pd.DataFrame,
    history: pd.DataFrame,
    levels: list[float],
    items: pd.Index,
) -> tuple[dict[float, PolicyRun], pd.DataFrame]:
    """Plan and simulate every service level for one model on ``items``.

    Args:
        preds: output of ``load_forecasts`` (any superset of ``items``).
        history: item-level long table with ``series_id, d, sales, is_closed, sell_price``
            covering the whole simulation window (actuals are read from it).

    Returns:
        ``(runs, coverage)``: one ``PolicyRun`` per level, and interval coverage per
        period / class / level.
    """
    inv = config["inventory"]
    R, L = inv["review_period_days"], inv["lead_time_days"]
    classes = item_classes(config).reindex(items).fillna("intermittent")
    periods, names = sim_periods(config)
    all_folds = {f.name: f for f in [*generate_folds(config), get_test_fold(config)]}
    preds = preds[preds["series_id"].isin(items)]
    hist = history.assign(series_id=history["series_id"].astype(str))
    hist = hist[hist["series_id"].isin(items)]

    windows = cumulative_windows(preds, R + L)
    scale_cache: dict[str, pd.Series] = {}

    def scales_at(fold_name: str) -> pd.Series:
        if fold_name not in scale_cache:
            cutoff = all_folds[fold_name].train_end
            s = compute_scales(hist[hist["d"] <= cutoff], cutoff)["mae"]
            scale_cache[fold_name] = s.reindex(items)
        return scale_cache[fold_name]

    q_levels = sorted(set(levels) | set(inv["interval_levels"]))
    days = np.arange(periods[0].valid_start, periods[-1].valid_end + 1)
    S = {lv: np.full((len(items), len(days)), np.nan) for lv in levels}
    review = np.zeros(len(days), bool)
    ss_rows, cov_rows = [], []
    for period in periods:
        calib = calibration_folds(period.name, names)
        err = pd.concat(
            [
                normalized_errors(windows[windows["fold"] == f], scales_at(f), classes)
                for f in calib
            ],
            ignore_index=True,
        )
        q = calibrate_quantiles(err, q_levels)
        scales = scales_at(period.name)
        cov = interval_coverage(
            windows[windows["fold"] == period.name], q, scales, classes, inv["interval_levels"]
        )
        cov_rows.append(cov.assign(period=period.name))

        p_days = np.arange(period.valid_start, period.valid_end + 1)
        fc = np.nan_to_num(_matrix(preds[preds["fold"] == period.name], "y_pred", items, p_days))
        sl = slice(int(p_days[0] - days[0]), int(p_days[-1] - days[0]) + 1)
        for lv in levels:
            ss = safety_stock(q, lv, scales, classes).to_numpy()
            S_p, review_p = order_up_to_levels(fc, ss, R, L)
            S[lv][:, sl] = S_p
            review[sl] = review_p
            ss_rows.append(
                pd.DataFrame(
                    {"period": period.name, "level": lv, "series_id": items, "safety_stock": ss}
                )
            )

    actual = hist[hist["d"].between(days[0], days[-1])]
    demand = np.nan_to_num(_matrix(actual, "sales", items, days))
    closed = np.nan_to_num(_matrix(actual, "is_closed", items, days))
    demand = np.where(closed == 1, 0.0, demand)
    forecast = np.nan_to_num(
        _matrix(preds[preds["d"].between(days[0], days[-1])], "y_pred", items, days)
    )
    ss_all = pd.concat(ss_rows, ignore_index=True)
    runs = {}
    for lv in levels:
        sim = simulate_lost_sales(demand, S[lv], review, lead_time=L, initial=S[lv][:, 0])
        runs[lv] = PolicyRun(
            items, days, demand, forecast, S[lv], review, sim, ss_all[ss_all["level"] == lv]
        )
    return runs, pd.concat(cov_rows, ignore_index=True)


def price_matrix(history: pd.DataFrame, items: pd.Index, days: np.ndarray) -> np.ndarray:
    """Daily sell price (items x days); gaps filled forward/backward within each item."""
    h = history.assign(series_id=history["series_id"].astype(str))
    m = h[h["d"].between(days[0], days[-1])].pivot_table(
        index="series_id", columns="d", values="sell_price", aggfunc="first"
    )
    m = m.reindex(index=items, columns=days).ffill(axis=1).bfill(axis=1)
    return m.fillna(0.0).to_numpy(dtype=float)


def summarize_run(
    run: PolicyRun, prices: np.ndarray, classes: pd.Series, holding_rate: float
) -> pd.DataFrame:
    """Per-item KPIs of one run, with the demand class attached."""
    items = summarize_items(run.sim, run.demand, prices, holding_rate)
    items.insert(0, "series_id", run.items.to_numpy())
    items.insert(1, "demand_class", classes.reindex(run.items).fillna("intermittent").to_numpy())
    return items


def aggregate_kpis(items: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Portfolio KPIs: unit fill rate, total average inventory value, holding cost, stockouts."""
    g = items.groupby(by, observed=True)
    out = g[
        ["demand", "served", "lost", "avg_inventory_value", "holding_cost", "stockout_days"]
    ].sum()
    out["fill_rate"] = out["served"] / out["demand"]
    out["n_items"] = g.size()
    return out.reset_index()


def inventory_at_fill_rate(
    summary: pd.DataFrame, targets: list[float], by: list[str] | None = None
) -> pd.DataFrame:
    """Average inventory value needed to reach each fill-rate target (linear interpolation).

    ``summary`` holds one row per service level with ``fill_rate`` and
    ``avg_inventory_value``; targets outside the simulated range give NaN.
    """
    by = by or ["model"]
    rows = []
    for key, g in summary.groupby(by, observed=True):
        g = g.sort_values("fill_rate")
        vals = np.interp(
            targets, g["fill_rate"], g["avg_inventory_value"], left=np.nan, right=np.nan
        )
        key = key if isinstance(key, tuple) else (key,)
        rows += [
            dict(zip(by, key, strict=True), fill_rate=t, avg_inventory_value=v)
            for t, v in zip(targets, vals, strict=True)
        ]
    return pd.DataFrame(rows)
