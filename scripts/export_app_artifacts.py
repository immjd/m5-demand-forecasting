"""Phase 8: precompute the small, aggregate-only tables the Streamlit app reads.

The app never trains a model and never sees item-level data: the M5 rules (7B) do
not allow the competition data to be published (docs D-033). Every table goes
through ``assert_aggregate_only`` before it is written, and daily series are
exported as an index (mean of the history window = 100) instead of units.

Usage (from the project root):
    python -m scripts.export_app_artifacts
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.config import PROJECT_ROOT, load_config, resolve_path
from forecasting.evaluation.metrics import metrics_by_horizon, per_series_rmsse
from forecasting.inventory.intervals import daily_intervals
from forecasting.inventory.run import (
    PolicyRun,
    aggregate_kpis,
    item_classes,
    load_forecasts,
    price_matrix,
    run_policy,
    summarize_run,
)
from forecasting.validation.backtest import generate_folds, get_test_fold, load_level_data

ARTIFACTS_DIR = PROJECT_ROOT / "app" / "artifacts"
ITEM_NAME = re.compile(r"(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}")
FORBIDDEN_COLUMNS = {"id", "item_id", "sales", "sell_price", "y_true", "units"}
log = logging.getLogger("export_app_artifacts")


# ---------------------------------------------------------------- guard
def assert_aggregate_only(df: pd.DataFrame, allowed_series: set[str]) -> None:
    """Raise if a table could expose item-level or raw-unit competition data."""
    bad = FORBIDDEN_COLUMNS & {str(c).lower() for c in df.columns}
    if bad:
        raise ValueError(f"forbidden columns in app artifact: {sorted(bad)}")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        values = df[col].dropna().astype(str).unique()
        hits = [v for v in values if ITEM_NAME.search(v)]
        if hits:
            raise ValueError(f"item names in app artifact column {col!r}: {hits[:3]}")
    if "series_id" in df.columns:
        extra = set(df["series_id"].astype(str)) - allowed_series
        if extra:
            raise ValueError(f"series outside the aggregate levels: {sorted(extra)[:3]}")


# ---------------------------------------------------------------- tables
def fold_table(config: dict[str, Any], agg: pd.DataFrame) -> pd.DataFrame:
    """Calendar dates of every fold for the backtest diagram."""
    day = agg.drop_duplicates("d").set_index("d")["date"]
    rows = [
        {
            "fold": f.name,
            "train_start": day.iloc[0],
            "train_end": day[f.train_end],
            "valid_start": day[f.valid_start],
            "valid_end": day[f.valid_end],
        }
        for f in [*generate_folds(config), get_test_fold(config)]
    ]
    return pd.DataFrame(rows)


def series_tables(
    config: dict[str, Any], agg: pd.DataFrame, preds_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Indexed history, indexed test forecasts with daily intervals, and interval coverage."""
    app, test = config["app"], get_test_fold(config)
    levels = app["interval_levels"]
    agg = agg.assign(series_id=agg["series_id"].astype(str))
    start = test.train_end - app["history_days"] + 1
    window = agg[agg["d"].between(start, test.train_end)]
    base = window[window["is_closed"] == 0].groupby("series_id")["sales"].mean().rename("base")

    hist = agg[agg["d"].between(start, test.valid_end)].join(base, on="series_id")
    hist["index"] = (100.0 * hist["sales"] / hist["base"]).round(1)  # units not recoverable exactly
    hist["part"] = np.where(hist["d"] <= test.train_end, "history", "test")
    history = hist[["series_id", "agg_level", "date", "part", "is_closed", "index"]]

    dates = agg.drop_duplicates("d").set_index("d")["date"]
    fc_rows, cov_rows = [], []
    for model in app["forecast_models"]:
        cv = pd.read_parquet(preds_dir / f"{model}_agg.parquet")
        te = pd.read_parquet(preds_dir / f"test_{model}_agg.parquet")
        out = daily_intervals(cv, te, levels).join(base, on="series_id")
        is_open = out["is_closed"] == 0
        for lv in levels:
            lo, hi = out[f"lo{round(lv * 100)}"], out[f"hi{round(lv * 100)}"]
            hit = (out["y_true"].between(lo, hi))[is_open]
            cov_rows.append({"model": model, "level": lv, "coverage": float(hit.mean())})
        bounds = [f"{b}{round(lv * 100)}" for lv in levels for b in ("lo", "hi")]
        table = pd.DataFrame(
            {
                "model": model,
                "series_id": out["series_id"],
                "date": out["d"].map(dates),
                "h": out["d"].astype(int) - test.train_end,
            }
        )
        for col in ["y_pred", *bounds]:
            table["forecast" if col == "y_pred" else col] = (100.0 * out[col] / out["base"]).round(
                1
            )
        fc_rows.append(table)
    return history, pd.concat(fc_rows, ignore_index=True), pd.DataFrame(cov_rows)


def error_tables(
    config: dict[str, Any], items: pd.DataFrame, preds_dir: Path
) -> dict[str, pd.DataFrame]:
    """CV errors per horizon day, demand class and department (aggregated over items)."""
    app, folds = config["app"], generate_folds(config)
    sample = pd.read_csv(resolve_path(config, "processed") / "stat_sample_items.csv")
    sample_ids = set(sample["item_id"].astype(str))
    sid = items["series_id"].astype(str)
    classes = item_classes(config)
    dept = items.assign(sid=sid).drop_duplicates("sid").set_index("sid")["dept_id"].astype(str)
    sample_items = items[sid.isin(sample_ids)]

    horizon, by_class, by_dept = [], [], []
    for subset, models, data in [
        ("all", app["item_models"], items),
        ("stat_sample", app["sample_models"], sample_items),
    ]:
        for model in models:
            p = pd.read_parquet(preds_dir / f"{model}_item.parquet")
            if subset == "stat_sample":
                p = p[p["series_id"].astype(str).isin(sample_ids)]
            tag = {"model": model, "level": "item", "subset": subset}
            horizon.append(metrics_by_horizon(p, folds).assign(**tag))
            r = per_series_rmsse(p, data, folds)
            cls = r.groupby(r["series_id"].map(classes))["rmsse"].mean()
            by_class.append(cls.rename_axis("demand_class").reset_index().assign(**tag))
            if subset == "all":
                dep = r.groupby(r["series_id"].map(dept))["rmsse"].mean()
                by_dept.append(dep.rename_axis("dept").reset_index().assign(**tag))
    for model in app["agg_models"]:
        p = pd.read_parquet(preds_dir / f"{model}_agg.parquet")
        tag = {"model": model, "level": "agg", "subset": "all"}
        horizon.append(metrics_by_horizon(p, folds).assign(**tag))
    return {
        "error_by_horizon": pd.concat(horizon, ignore_index=True),
        "error_by_class": pd.concat(by_class, ignore_index=True),
        "error_by_dept": pd.concat(by_dept, ignore_index=True),
    }


# ---------------------------------------------------------------- inventory grid
def with_policy(config: dict[str, Any], review: int, lead: int) -> dict[str, Any]:
    """Copy of the config with another review period / lead time."""
    out = copy.deepcopy(config)
    out["inventory"]["review_period_days"] = review
    out["inventory"]["lead_time_days"] = lead
    return out


def stock_composition(run: PolicyRun, classes: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Safety stock as a share of the order-up-to level, per demand class and overall."""
    n_reviews = int(run.review.sum()) // run.safety_stock["period"].nunique()
    ss = run.safety_stock.groupby("series_id")["safety_stock"].sum() * n_reviews
    frame = pd.DataFrame(
        {
            "ss": ss.reindex(run.items).to_numpy(),
            "target": np.nansum(run.order_up_to, axis=1),
            "demand_class": classes.reindex(run.items).to_numpy(),
        }
    )
    g = frame.groupby("demand_class")[["ss", "target"]].sum()
    g.loc["semua"] = g.sum()
    return (g["ss"] / g["target"]).rename("safety_stock_share").reset_index()


def inventory_grid(config: dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    """Portfolio KPIs for every review period x lead time x model x service level x class."""
    app, inv = config["app"], config["inventory"]
    levels = list(inv["service_levels"])
    classes = item_classes(config)
    items = pd.Index(sorted(history["series_id"].astype(str).unique()))
    rows = []
    for review in app["planner_review_days"]:
        for lead in app["planner_lead_days"]:
            cfg = with_policy(config, review, lead)
            for model in inv["models_all_items"]:
                t0 = time.perf_counter()
                runs, _ = run_policy(cfg, load_forecasts(cfg, model), history, levels, items)
                first = next(iter(runs.values()))
                prices, n_days = price_matrix(history, items, first.days), len(first.days)
                for lv, run in runs.items():
                    kpi = summarize_run(run, prices, classes, inv["holding_cost_rate_annual"])
                    per = aggregate_kpis(kpi, ["demand_class"])
                    tot = aggregate_kpis(kpi.assign(demand_class="semua"), ["demand_class"])
                    cover = kpi.groupby("demand_class")["avg_on_hand"].sum()
                    cover.loc["semua"] = cover.sum()
                    t = pd.concat([tot, per], ignore_index=True)
                    t = t.merge(stock_composition(run, classes, cfg), on="demand_class")
                    t["cover_days"] = t["demand_class"].map(cover) / (t["demand"] / n_days)
                    t["stockout_rate"] = t["stockout_days"] / (t["n_items"] * n_days)
                    rows.append(t.assign(review=review, lead=lead, model=model, level=lv))
                log.info("R=%d L=%d %s: %.1fs", review, lead, model, time.perf_counter() - t0)
    keep = ["review", "lead", "model", "level", "demand_class", "n_items", "fill_rate",
            "avg_inventory_value", "holding_cost", "stockout_rate", "cover_days",
            "safety_stock_share"]  # fmt: skip
    return pd.concat(rows, ignore_index=True)[keep]


def check_against_summary(grid: pd.DataFrame, config: dict[str, Any]) -> None:
    """The default R, L cell must reproduce ``inventory_summary.csv`` (Phase 7)."""
    inv = config["inventory"]
    s = pd.read_csv(resolve_path(config, "results") / "inventory_summary.csv")
    s = s[s["subset"] == "all"]
    g = grid[
        (grid["review"] == inv["review_period_days"]) & (grid["lead"] == inv["lead_time_days"])
    ]
    key = ["model", "level", "demand_class"]
    m = g.merge(s[[*key, "fill_rate", "avg_inventory_value"]], on=key, suffixes=("", "_ref"))
    if len(m) != len(g) or m.empty:
        raise ValueError("default planner cell does not match inventory_summary.csv rows")
    for col in ["fill_rate", "avg_inventory_value"]:
        np.testing.assert_allclose(m[col], m[f"{col}_ref"], rtol=1e-6)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    config = load_config(args.config)
    results = resolve_path(config, "results")
    preds_dir = results / "preds"
    # Only reads predictions that already exist; nothing is trained or re-evaluated here.
    agg = load_level_data(config, "agg", include_test=True)
    history = load_level_data(config, "item", include_test=True)
    last_cv = generate_folds(config)[-1].valid_end
    allowed = set(agg["series_id"].astype(str))

    hist, forecast, daily_cov = series_tables(config, agg, preds_dir)
    grid = inventory_grid(config, history)
    check_against_summary(grid, config)
    drop = ["n_folds", "runtime_sec_total"]
    tables = {
        "folds": fold_table(config, agg),
        "series_history": hist,
        "series_forecast": forecast,
        "daily_coverage": daily_cov,
        "leaderboard_cv": pd.read_csv(results / "leaderboard_cv.csv").drop(columns=drop),
        "test_results": pd.read_csv(results / "test_results.csv")[
            ["model", "level", "subset", "wrmsse", "rmsse", "mase", "wape", "bias"]
        ],
        "interval_coverage": pd.read_csv(results / "interval_coverage.csv"),
        "inventory_grid": grid,
        **error_tables(config, history[history["d"] <= last_cv], preds_dir),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, df in tables.items():
        assert_aggregate_only(df, allowed)
        path = ARTIFACTS_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        total += path.stat().st_size
        log.info("%-18s %6d rows %8.1f KB", name, len(df), path.stat().st_size / 1024)
    classes = item_classes(config).value_counts()
    inv = config["inventory"]
    meta = {
        "store": config["scope"]["store_id"],
        "n_items": int(classes.sum()),
        "items_per_class": {k: int(v) for k, v in classes.items()},
        "history_days": config["app"]["history_days"],
        "horizon": config["forecast"]["horizon"],
        "service_levels": inv["service_levels"],
        "default_review": inv["review_period_days"],
        "default_lead": inv["lead_time_days"],
        "review_days": config["app"]["planner_review_days"],
        "lead_days": config["app"]["planner_lead_days"],
        "interval_levels": config["app"]["interval_levels"],
        "holding_rate": inv["holding_cost_rate_annual"],
        "sim_days": int(
            get_test_fold(config).valid_end
            - next(f for f in generate_folds(config) if f.name == inv["sim_start_fold"]).valid_start
            + 1
        ),
        "exported": datetime.now().isoformat(timespec="seconds"),
    }
    text = json.dumps(meta, indent=2)
    if ITEM_NAME.search(text):
        raise ValueError("item names in meta.json")
    (ARTIFACTS_DIR / "meta.json").write_text(text, encoding="utf-8")
    log.info("Total artifact size: %.2f MB in %s", total / 1024**2, ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
