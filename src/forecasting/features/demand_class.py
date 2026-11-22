"""Syntetos-Boylan demand classification (ADI / CV^2) and stratified item sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEMAND_CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]
INSUFFICIENT = "insufficient_history"


def classification_cutoff(config: dict[str, Any]) -> int:
    """Return the last ``d`` usable for classification: train_end of the first fold."""
    return int(config["backtest"]["folds"][0][0])


def compute_adi_cv2(df: pd.DataFrame, cutoff_d: int) -> pd.DataFrame:
    """Compute ADI and CV^2 per item using only open days with ``d <= cutoff_d``.

    ADI = number of periods / number of non-zero periods (average inter-demand
    interval). CV^2 = (std / mean)^2 of the non-zero demand sizes.

    Args:
        df: Long table with ``item_id, d, sales, is_closed`` (already launch-trimmed).
        cutoff_d: Last day included. Data after it is never read.

    Returns:
        One row per item seen before the cutoff:
        ``item_id, n_periods, n_nonzero, adi, cv2``.
    """
    hist = df.loc[(df["d"] <= cutoff_d) & (df["is_closed"] == 0), ["item_id", "sales"]]
    grouped = hist.groupby("item_id", observed=True)["sales"]
    n_periods = grouped.size()

    nonzero = hist.loc[hist["sales"] > 0]
    nz = nonzero.groupby("item_id", observed=True)["sales"]
    n_nonzero = nz.size().reindex(n_periods.index, fill_value=0)
    mean = nz.mean().reindex(n_periods.index)
    std = nz.std(ddof=0).reindex(n_periods.index)

    out = pd.DataFrame(
        {
            "n_periods": n_periods.astype("int32"),
            "n_nonzero": n_nonzero.astype("int32"),
            "adi": (n_periods / n_nonzero.replace(0, np.nan)).astype("float64"),
            "cv2": ((std / mean) ** 2).astype("float64"),
        }
    )
    out.index.name = "item_id"
    return out.reset_index()


def classify_demand(
    stats: pd.DataFrame,
    adi_threshold: float,
    cv2_threshold: float,
    min_nonzero_days: int,
) -> pd.Series:
    """Map ADI / CV^2 to smooth, erratic, intermittent, lumpy or insufficient_history."""
    frequent = stats["adi"] < adi_threshold
    stable = stats["cv2"] < cv2_threshold
    labels = np.select(
        [frequent & stable, frequent & ~stable, ~frequent & stable, ~frequent & ~stable],
        DEMAND_CLASSES,
        default=INSUFFICIENT,
    )
    labels = np.where(stats["n_nonzero"] < min_nonzero_days, INSUFFICIENT, labels)
    return pd.Series(labels, index=stats.index, name="demand_class")


def build_item_classes(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Classify every item in ``df``; items with no history before the cutoff are insufficient.

    Returns:
        ``item_id, dept_id, cat_id, n_periods, n_nonzero, adi, cv2, demand_class``.
    """
    params = config["demand_class"]
    cutoff = classification_cutoff(config)
    items = df[["item_id", "dept_id", "cat_id"]].drop_duplicates("item_id")
    stats = compute_adi_cv2(df, cutoff)
    out = items.merge(stats, on="item_id", how="left")
    out[["n_periods", "n_nonzero"]] = out[["n_periods", "n_nonzero"]].fillna(0).astype("int32")
    out["demand_class"] = classify_demand(
        out, params["adi_threshold"], params["cv2_threshold"], params["min_nonzero_days"]
    )
    out["demand_class"] = pd.Categorical(
        out["demand_class"], categories=[*DEMAND_CLASSES, INSUFFICIENT]
    )
    for col in ["item_id", "dept_id", "cat_id"]:
        out[col] = out[col].astype(str)
    return out.sort_values("item_id").reset_index(drop=True)


def stratified_sample(classes: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Draw ``n_total / 4`` items per demand class (all items if a class is smaller).

    ``insufficient_history`` items are never sampled.

    Returns:
        ``item_id, demand_class`` sorted by class then item.
    """
    n_per_class = n_total // len(DEMAND_CLASSES)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in DEMAND_CLASSES:
        pool = np.sort(classes.loc[classes["demand_class"] == cls, "item_id"].to_numpy())
        take = min(n_per_class, len(pool))
        chosen = rng.choice(pool, size=take, replace=False) if take else np.array([], dtype=str)
        parts.append(pd.DataFrame({"item_id": np.sort(chosen), "demand_class": cls}))
    return pd.concat(parts, ignore_index=True)
