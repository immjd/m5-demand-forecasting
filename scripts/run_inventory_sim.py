"""Phase 7: turn out-of-sample forecasts into (R, S) stock policies and simulate them.

For every model x service level: safety stock from earlier-fold residuals per demand
class, order-up-to levels on weekly reviews, and a daily lost-sales simulation over
the periods from ``inventory.sim_start_fold`` to the end of the test window.

Usage (from the project root):
    python -m scripts.run_inventory_sim
    python -m scripts.run_inventory_sim --sample 50     # smoke test, nothing saved
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.inventory.run import (
    aggregate_kpis,
    item_classes,
    load_forecasts,
    price_matrix,
    run_policy,
    summarize_run,
)
from forecasting.validation.backtest import load_level_data

ITEMS_FILE = "inventory_items.parquet"
SUMMARY_FILE = "inventory_summary.csv"
COVERAGE_FILE = "interval_coverage.csv"
log = logging.getLogger("run_inventory_sim")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample", type=int, default=None, help="Smoke test: N random items")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = load_config(args.config)
    inv = config["inventory"]
    levels = list(inv["service_levels"])
    history = load_level_data(config, "item", include_test=True)
    classes = item_classes(config)
    all_items = pd.Index(sorted(history["series_id"].astype(str).unique()))
    sample = pd.read_csv(resolve_path(config, "processed") / "stat_sample_items.csv")
    sample_items = pd.Index(sorted(sample["item_id"].astype(str)))
    if args.sample:
        rng = np.random.default_rng(config["seed"])
        all_items = pd.Index(sorted(rng.choice(all_items, args.sample, replace=False)))
        sample_items = sample_items[: max(4, args.sample // 5)]
        log.info(
            "SMOKE TEST: %d items (+%d sample items), nothing saved",
            len(all_items),
            len(sample_items),
        )

    jobs = [("all", m, all_items) for m in inv["models_all_items"]]
    jobs += [("stat_sample", m, sample_items) for m in inv["models_sample"]]
    item_rows, cov_rows = [], []
    for subset, model, items in jobs:
        t0 = time.perf_counter()
        runs, cov = run_policy(config, load_forecasts(config, model), history, levels, items)
        first = next(iter(runs.values()))
        prices = price_matrix(history, items, first.days)
        for lv, run in runs.items():
            item_rows.append(
                summarize_run(run, prices, classes, inv["holding_cost_rate_annual"]).assign(
                    model=model, subset=subset, level=lv
                )
            )
        cov_rows.append(cov.assign(model=model, subset=subset))
        log.info(
            "%s / %s: %d items x %d days x %d levels in %.1fs",
            subset,
            model,
            len(items),
            len(first.days),
            len(levels),
            time.perf_counter() - t0,
        )

    items_df = pd.concat(item_rows, ignore_index=True)
    total = aggregate_kpis(items_df, ["subset", "model", "level"]).assign(demand_class="semua")
    per_class = aggregate_kpis(items_df, ["subset", "model", "level", "demand_class"])
    summary = pd.concat([total, per_class], ignore_index=True)
    coverage = pd.concat(cov_rows, ignore_index=True)

    view = total[
        [
            "subset",
            "model",
            "level",
            "fill_rate",
            "avg_inventory_value",
            "holding_cost",
            "stockout_days",
        ]
    ]
    with pd.option_context("display.width", 160, "display.float_format", "{:,.3f}".format):
        log.info("Portfolio KPIs:\n%s", view.to_string(index=False))
        cov_all = coverage[coverage["demand_class"] == "semua"]
        log.info(
            "Interval coverage (all classes):\n%s",
            cov_all.pivot_table(
                index=["subset", "model", "level"], columns="period", values="coverage"
            ).to_string(),
        )
    if args.sample:
        return
    results = resolve_path(config, "results")
    items_df.to_parquet(results / ITEMS_FILE, index=False)
    summary.to_csv(results / SUMMARY_FILE, index=False)
    coverage.to_csv(results / COVERAGE_FILE, index=False)
    log.info("Saved: %s, %s, %s", ITEMS_FILE, SUMMARY_FILE, COVERAGE_FILE)


if __name__ == "__main__":
    main()
