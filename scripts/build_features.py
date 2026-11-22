"""Phase 5: build the fold-independent LightGBM feature table.

Usage (from the project root):
    python -m scripts.build_features            # days up to the last validation day (d_1913)
    python -m scripts.build_features --include-test   # Phase 6 only: also d_1914..d_1941
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.data.load import read_calendar, store_state
from forecasting.features.build import build_features as build
from forecasting.features.build import feature_columns
from forecasting.validation.backtest import load_level_data

FEATURES_FILE = "features.parquet"
FEATURES_TEST_FILE = "features_with_test.parquet"
log = logging.getLogger("build_features")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Phase 6 only: include the test window (d_1914..d_1941)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.perf_counter()
    config = load_config(args.config)
    sales = load_level_data(config, "item", include_test=args.include_test)
    calendar = read_calendar(resolve_path(config, "raw"), store_state(config["scope"]["store_id"]))
    classes = pd.read_parquet(resolve_path(config, "processed") / "item_classes.parquet")
    log.info(
        "Building features for %d items, d_%d..d_%d (%s rows)",
        sales["series_id"].nunique(),
        sales["d"].min(),
        sales["d"].max(),
        f"{len(sales):,}",
    )

    feats = build(sales, calendar, classes, config)
    cols = [c for c in feature_columns(config) if c in feats.columns]
    path = resolve_path(config, "processed") / (
        FEATURES_TEST_FILE if args.include_test else FEATURES_FILE
    )
    feats.to_parquet(path, index=False)

    nan_share = feats[cols].isna().mean().sort_values(ascending=False)
    log.info(
        "Features: %d (+ price_rel_item_mean per fold) | rows %s | memory %.0f MB",
        len(cols),
        f"{len(feats):,}",
        feats.memory_usage(deep=True).sum() / 1024**2,
    )
    log.info(
        "Highest NaN share (expected for long lags / sparse events):\n%s",
        nan_share.head(6).round(3).to_string(),
    )
    log.info("Saved: %s (%.1fs)", path, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
