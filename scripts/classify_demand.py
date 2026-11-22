"""Phase 2: classify items by demand pattern and draw the stratified statistical sample.

Usage (from the project root):
    python -m scripts.classify_demand [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.features.demand_class import (
    build_item_classes,
    classification_cutoff,
    stratified_sample,
)

SALES_LONG_FILE = "sales_long.parquet"
ITEM_CLASSES_FILE = "item_classes.parquet"
SAMPLE_FILE = "stat_sample_items.csv"

log = logging.getLogger("classify_demand")


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

    config = load_config(args.config)
    cutoff = classification_cutoff(config)
    sales = pd.read_parquet(
        resolve_path(config, "interim") / SALES_LONG_FILE,
        columns=["item_id", "dept_id", "cat_id", "d", "sales", "is_closed"],
    )
    log.info("Classifying %d items on data up to d_%d ...", sales["item_id"].nunique(), cutoff)
    classes = build_item_classes(sales, config)

    sample = stratified_sample(
        classes, n_total=config["statistical"]["item_sample_size"], seed=config["seed"]
    )

    out_dir = resolve_path(config, "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    classes.to_parquet(out_dir / ITEM_CLASSES_FILE, index=False)
    sample.to_csv(out_dir / SAMPLE_FILE, index=False)

    counts = classes["demand_class"].value_counts().reindex(classes["demand_class"].cat.categories)
    log.info("=" * 60)
    log.info("Demand classes (cutoff d_%d):", cutoff)
    for cls, n in counts.items():
        log.info("  %-22s %5d  (%.1f%%)", cls, n, 100 * n / len(classes))
    log.info("Sample per class: %s", sample["demand_class"].value_counts().to_dict())
    log.info("Saved: %s", out_dir / ITEM_CLASSES_FILE)
    log.info("Saved: %s", out_dir / SAMPLE_FILE)


if __name__ == "__main__":
    main()
