"""Read raw M5 CSVs and build a long-format sales table for one store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.config import resolve_path

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id"]
CATEGORY_COLS = [
    *ID_COLS,
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


def store_state(store_id: str) -> str:
    """Return the state code of a store, e.g. ``'CA_1' -> 'CA'``."""
    return store_id.split("_")[0]


def read_sales_wide(raw_dir: Path, store_id: str) -> pd.DataFrame:
    """Read the wide sales CSV (one row per item) and keep a single store."""
    path = raw_dir / SALES_FILE
    header = pd.read_csv(path, nrows=0).columns
    d_cols = [c for c in header if c.startswith("d_")]
    wide = pd.read_csv(path, usecols=ID_COLS + d_cols, dtype=dict.fromkeys(d_cols, "int16"))
    wide = wide.loc[wide["store_id"] == store_id]
    if wide.empty:
        raise ValueError(f"store_id {store_id!r} not found in {path}")
    return wide.reset_index(drop=True)


def melt_sales(wide: pd.DataFrame, history_start_d: int = 1) -> pd.DataFrame:
    """Convert wide ``d_1..d_N`` columns to long rows with an integer ``d``."""
    long = wide.melt(id_vars=ID_COLS, var_name="d", value_name="sales")
    long["d"] = long["d"].str.removeprefix("d_").astype("int16")
    long["sales"] = long["sales"].astype("int16")
    long = long.loc[long["d"] >= history_start_d]
    return long.reset_index(drop=True)


def read_calendar(raw_dir: Path, state: str) -> pd.DataFrame:
    """Read the calendar and keep only the SNAP column of ``state`` as ``snap``."""
    snap_col = f"snap_{state}"
    cols = [
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "d",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        snap_col,
    ]
    cal = pd.read_csv(raw_dir / CALENDAR_FILE, usecols=cols, parse_dates=["date"])
    cal = cal.rename(columns={snap_col: "snap"})
    cal["d"] = cal["d"].str.removeprefix("d_").astype("int16")
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int16")
    cal["wday"] = cal["wday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    cal["snap"] = cal["snap"].astype("int8")
    return cal


def read_prices(raw_dir: Path, store_id: str) -> pd.DataFrame:
    """Read weekly sell prices for one store."""
    prices = pd.read_csv(
        raw_dir / PRICES_FILE,
        dtype={"wm_yr_wk": "int16", "sell_price": "float32"},
    )
    prices = prices.loc[prices["store_id"] == store_id, ["item_id", "wm_yr_wk", "sell_price"]]
    return prices.reset_index(drop=True)


def store_overview(config: dict[str, Any], last_d: int, window_days: int = 364) -> pd.DataFrame:
    """Summarise every store for store selection, using only days ``<= last_d``.

    Metrics per store: number of items, mean daily units over the full history
    and over the last ``window_days``, year-over-year growth of the last window,
    share of zero item-days in the last window, and the number of days on which
    the whole store sold nothing.
    """
    path = resolve_path(config, "raw") / SALES_FILE
    header = pd.read_csv(path, nrows=0).columns
    d_cols = [c for c in header if c.startswith("d_") and int(c[2:]) <= last_d]
    wide = pd.read_csv(path, usecols=["store_id", *d_cols], dtype=dict.fromkeys(d_cols, "int16"))

    last = d_cols[-window_days:]
    prev = d_cols[-2 * window_days : -window_days]
    rows = []
    for store_id, g in wide.groupby("store_id"):
        daily = g[d_cols].sum(axis=0)
        rows.append(
            {
                "store_id": store_id,
                "n_items": len(g),
                "mean_daily_units": float(daily.mean()),
                "mean_daily_units_last": float(daily[last].mean()),
                "yoy_growth_last": float(daily[last].sum() / daily[prev].sum() - 1),
                "pct_zero_item_days_last": float((g[last] == 0).to_numpy().mean() * 100),
                "zero_sales_days": int((daily == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("store_id").reset_index(drop=True)


def load_sales_long(config: dict[str, Any]) -> pd.DataFrame:
    """Build the long sales table for ``config['scope']['store_id']``.

    Joins calendar features (date, events, the store's SNAP flag) and the
    weekly sell price. ``sell_price`` is NaN for weeks where the item was not
    on sale yet.

    Returns:
        One row per (item, day), sorted by ``item_id`` then ``d``.
    """
    raw_dir = resolve_path(config, "raw")
    store_id = config["scope"]["store_id"]
    history_start_d = config["scope"]["history_start_d"]

    wide = read_sales_wide(raw_dir, store_id)
    long = melt_sales(wide, history_start_d)
    del wide

    calendar = read_calendar(raw_dir, store_state(store_id))
    prices = read_prices(raw_dir, store_id)

    for col in ID_COLS:
        long[col] = long[col].astype(str)
    df = long.merge(calendar, on="d", how="left", validate="many_to_one")
    df = df.merge(prices, on=["item_id", "wm_yr_wk"], how="left", validate="many_to_one")

    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    df = df.sort_values(["item_id", "d"]).reset_index(drop=True)
    return df
