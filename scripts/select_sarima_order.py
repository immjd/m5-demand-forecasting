"""Phase 4: choose SARIMA orders by AIC on fold F1 training data only.

* Aggregate level: one order per series (11 series).
* Item level: one order per demand class, by summed AIC over a few items per
  class drawn (with the config seed) from the 200-item statistical sample.

Usage (from the project root):
    python -m scripts.select_sarima_order               # full grid
    python -m scripts.select_sarima_order --max-orders 8 --no-save   # timing smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.features.demand_class import DEMAND_CLASSES
from forecasting.models.registry import SARIMA_ORDERS_FILE
from forecasting.models.statistical import (
    aic_table,
    best_order,
    order_to_str,
    regular_history,
    sarima_grid,
)
from forecasting.validation.backtest import generate_folds, load_level_data

SAMPLE_FILE = "stat_sample_items.csv"
log = logging.getLogger("select_sarima_order")


def histories(data: pd.DataFrame, ids: list[str], cutoff: int, window: int, season: int) -> dict:
    """Regular training histories (last ``window`` days up to ``cutoff``) for ``ids``."""
    recent = data.loc[(data["d"] <= cutoff) & (data["d"] > cutoff - window)]
    recent = recent.assign(series_id=recent["series_id"].astype(str))
    recent = recent.loc[recent["series_id"].isin(ids)]
    return {
        sid: regular_history(g, cutoff, season).to_numpy()
        for sid, g in recent.groupby("series_id", sort=True)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=None,
        help="Only try the first N grid orders (timing smoke test)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write the JSON file")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = load_config(args.config)
    s = config["statistical"]
    f1 = generate_folds(config)[0]
    cutoff, window, season = f1.train_end, s["train_window_days"], s["season"]
    orders = sarima_grid(s["sarima_grid"])[: args.max_orders]
    log.info(
        "Grid: %d orders | data: d_%d..d_%d (F1 train, last %d days)",
        len(orders),
        cutoff - window + 1,
        cutoff,
        window,
    )

    # ---- aggregate level: one order per series
    agg = load_level_data(config, "agg")
    agg_hist = histories(agg, sorted(agg["series_id"].astype(str).unique()), cutoff, window, season)
    t0 = time.perf_counter()
    agg_table = aic_table(agg_hist, orders, season, s["sarima_maxiter"], s["n_jobs"])
    t_agg = time.perf_counter() - t0
    agg_orders = {sid: order_to_str(best_order(agg_table, [sid])) for sid in agg_hist}
    log.info(
        "Aggregate: %d series x %d orders in %.1fs (%.3fs per fit, %d failed fits)",
        len(agg_hist),
        len(orders),
        t_agg,
        t_agg / agg_table.size,
        int(np.isinf(agg_table.to_numpy()).sum()),
    )
    for sid, o in agg_orders.items():
        log.info("  %-12s %s", sid, o)

    # ---- item level: one order per demand class
    rng = np.random.default_rng(config["seed"])
    sample = pd.read_csv(resolve_path(config, "processed") / SAMPLE_FILE)
    k = s["order_selection_items_per_class"]
    picked = {
        cls: sorted(
            rng.choice(
                np.sort(sample.loc[sample["demand_class"] == cls, "item_id"]),
                size=min(k, int((sample["demand_class"] == cls).sum())),
                replace=False,
            ).tolist()
        )
        for cls in DEMAND_CLASSES
    }
    all_ids = sorted(i for ids in picked.values() for i in ids)
    items = load_level_data(config, "item")
    item_hist = histories(items, all_ids, cutoff, window, season)
    t0 = time.perf_counter()
    item_table = aic_table(item_hist, orders, season, s["sarima_maxiter"], s["n_jobs"])
    t_item = time.perf_counter() - t0
    class_orders = {
        cls: order_to_str(best_order(item_table, [i for i in ids if i in item_hist]))
        for cls, ids in picked.items()
    }
    log.info(
        "Items: %d series x %d orders in %.1fs (%.3fs per fit, %d failed fits)",
        len(item_hist),
        len(orders),
        t_item,
        t_item / item_table.size,
        int(np.isinf(item_table.to_numpy()).sum()),
    )
    for cls, o in class_orders.items():
        log.info("  %-12s %s", cls, o)

    if args.no_save:
        full = len(sarima_grid(s["sarima_grid"]))
        est = (t_agg + t_item) * full / len(orders)
        log.info(
            "Not saved. Estimated time for the full grid (%d orders): %.1f min", full, est / 60
        )
        return

    out = {
        "agg": agg_orders,
        "item_class": class_orders,
        "meta": {
            "fold": f1.name,
            "train_d": [cutoff - window + 1, cutoff],
            "n_orders": len(orders),
            "selection_items": picked,
            "runtime_sec": round(t_agg + t_item, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }
    path = resolve_path(config, "results") / SARIMA_ORDERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved: %s", path)


if __name__ == "__main__":
    main()
