"""Phase 1: build the long-format sales table and the 11 aggregate series.

Usage (from the project root):
    python -m scripts.make_dataset [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.data.aggregate import build_aggregates
from forecasting.data.load import load_sales_long
from forecasting.data.preprocess import add_zero_run_len, flag_closed_days, trim_leading_zeros

SALES_LONG_FILE = "sales_long.parquet"
AGGREGATES_FILE = "aggregates.parquet"

log = logging.getLogger("make_dataset")


def _mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    t0 = time.perf_counter()
    config = load_config(args.config)
    store_id = config["scope"]["store_id"]

    log.info("Loading raw CSVs for store %s ...", store_id)
    df = load_sales_long(config)
    rows_raw, sales_raw = len(df), int(df["sales"].sum())
    log.info(
        "Loaded %s rows (%.0f MB) in %.1fs", f"{rows_raw:,}", _mem_mb(df), time.perf_counter() - t0
    )

    log.info("Flagging closed days ...")
    df = flag_closed_days(df)
    closed_dates = sorted(df.loc[df["is_closed"] == 1, "date"].dt.date.unique())

    log.info("Trimming rows before each item's first sell price ...")
    items_raw = df["item_id"].nunique()
    df = trim_leading_zeros(df)
    sales_dropped = sales_raw - int(df["sales"].sum())

    log.info("Computing zero_run_len ...")
    df = add_zero_run_len(df)

    log.info("Building aggregate series ...")
    agg = build_aggregates(df)

    interim_dir = resolve_path(config, "interim")
    processed_dir = resolve_path(config, "processed")
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_dir / SALES_LONG_FILE, index=False)
    agg.to_parquet(processed_dir / AGGREGATES_FILE, index=False)

    open_days = df["is_closed"] == 0
    zero_pct = 100 * (df.loc[open_days, "sales"] == 0).mean()
    long_zero_runs = df.loc[df["zero_run_len"] >= 28, "item_id"].nunique()

    log.info("=" * 60)
    log.info("SUMMARY (store %s)", store_id)
    log.info("Items: %d (raw %d)", df["item_id"].nunique(), items_raw)
    log.info(
        "Date range: %s -> %s (d_%d..d_%d)",
        df["date"].min().date(),
        df["date"].max().date(),
        df["d"].min(),
        df["d"].max(),
    )
    log.info(
        "Rows: %s raw -> %s after trim (-%.1f%%)",
        f"{rows_raw:,}",
        f"{len(df):,}",
        100 * (1 - len(df) / rows_raw),
    )
    log.info("Units sold dropped by trim: %s (should be 0)", f"{sales_dropped:,}")
    log.info("Zero-sales rows on open days: %.1f%%", zero_pct)
    log.info("Closed days: %d -> %s", len(closed_dates), ", ".join(map(str, closed_dates)))
    log.info("Items with a zero run >= 28 days at some point: %d", long_zero_runs)
    log.info("Missing sell_price after trim: %s rows", f"{int(df['sell_price'].isna().sum()):,}")
    log.info("Aggregates: %d series x %d days", agg["series_id"].nunique(), agg["d"].nunique())
    log.info("Memory: sales_long %.0f MB, aggregates %.1f MB", _mem_mb(df), _mem_mb(agg))
    log.info("Saved: %s", interim_dir / SALES_LONG_FILE)
    log.info("Saved: %s", processed_dir / AGGREGATES_FILE)
    log.info("Total runtime: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
