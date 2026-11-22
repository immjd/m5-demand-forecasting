"""Assemble the LightGBM feature table (fold-independent part)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forecasting.features.calendar import CALENDAR_FEATURES, calendar_features
from forecasting.features.demand_class import DEMAND_CLASSES, INSUFFICIENT
from forecasting.features.lags import (
    add_lag_features,
    add_rolling_features,
    add_shifted,
    check_contiguous,
    lag_feature_names,
)
from forecasting.features.price import FOLD_PRICE_FEATURES, PRICE_FEATURES, add_price_features

ID_FEATURES = ["series_id", "dept_id", "cat_id"]
INTERMITTENCY_FEATURES = ["zero_run_len_lag", "demand_class"]
CATEGORICAL = ["series_id", "dept_id", "cat_id", "event_name_1", "event_type_1", "demand_class"]
BASE_COLS = ["series_id", "dept_id", "cat_id", "d", "date", "sales", "sell_price", "is_closed"]


def feature_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """Feature names per group (used for training and for ablations)."""
    p = config["lgbm"]
    return {
        "lag": lag_feature_names(p["lags"], p["rolling_windows"]),
        "calendar": CALENDAR_FEATURES,
        "price": PRICE_FEATURES + FOLD_PRICE_FEATURES,
        "identity": ID_FEATURES,
        "intermittency": INTERMITTENCY_FEATURES,
    }


def feature_columns(config: dict[str, Any], drop_groups: tuple[str, ...] = ()) -> list[str]:
    """All model features, optionally without some groups (ablation)."""
    groups = feature_groups(config)
    return [f for name, cols in groups.items() if name not in drop_groups for f in cols]


def build_features(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    classes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build every fold-independent feature for the item-level table.

    Args:
        sales: Item-level long table with ``series_id`` (from ``load_level_data``),
            including ``zero_run_len`` from Phase 1.
        calendar: Full M5 calendar (``read_calendar``).
        classes: ``item_classes.parquet`` (demand classes computed on data <= d_1801).
        config: Project config (lags, windows, min_lag, momentum window).

    Returns:
        Base columns + features, sorted by ``series_id, d``. ``price_rel_item_mean``
        is added later per fold by the model.
    """
    p = config["lgbm"]
    min_lag = config["forecast"]["min_lag"]

    df = sales[[*BASE_COLS, "zero_run_len"]].sort_values(["series_id", "d"]).reset_index(drop=True)
    check_contiguous(df)
    df = add_lag_features(df, p["lags"], min_lag)
    df = add_rolling_features(df, p["rolling_windows"], shift=min_lag)
    df = add_shifted(df, "zero_run_len", min_lag, "zero_run_len_lag")
    df = add_price_features(df, p["price_momentum_days"])
    df = df.drop(columns="zero_run_len")

    df = df.merge(calendar_features(calendar), on="d", how="left", validate="many_to_one")

    cls = classes.assign(series_id=classes["item_id"].astype(str))[["series_id", "demand_class"]]
    df["series_id"] = df["series_id"].astype(str)
    df = df.merge(cls, on="series_id", how="left")
    df["demand_class"] = pd.Categorical(
        df["demand_class"].fillna(INSUFFICIENT), categories=[*DEMAND_CLASSES, INSUFFICIENT]
    )
    for col in ["series_id", "dept_id", "cat_id"]:
        df[col] = df[col].astype("category")
    return df.sort_values(["series_id", "d"]).reset_index(drop=True)
