"""Phases 3-5: run one model through the rolling-origin backtest and update the leaderboard.

Usage (from the project root):
    python -m scripts.run_backtest --model seasonal_naive --level item
    python -m scripts.run_backtest --model seasonal_naive --level agg
    python -m scripts.run_backtest --model naive --level item --sample 50   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.evaluation.metrics import evaluate_fold
from forecasting.models.baselines import MODEL_REGISTRY
from forecasting.validation.backtest import generate_folds, load_level_data, run_backtest

METRICS_FILE = "metrics_by_fold.csv"
LEADERBOARD_FILE = "leaderboard_cv.csv"
SAMPLE_FILE = "stat_sample_items.csv"
METRIC_COLS = ["wrmsse", "rmsse", "mase", "wape", "bias"]

log = logging.getLogger("run_backtest")


def config_hash(config: dict[str, Any]) -> str:
    """Short, stable hash of the config, stored with every result row."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:8]


def subsets_for(level: str, config: dict[str, Any]) -> dict[str, set[str] | None]:
    """Evaluation subsets: all series, plus the 200-item statistical sample at item level."""
    subsets: dict[str, set[str] | None] = {"all": None}
    if level == "item":
        sample = pd.read_csv(resolve_path(config, "processed") / SAMPLE_FILE)
        subsets["stat_sample"] = set(sample["item_id"].astype(str))
    return subsets


def score(preds: pd.DataFrame, data: pd.DataFrame, folds, subsets, config) -> pd.DataFrame:
    """Metrics per (subset, fold)."""
    window = config["evaluation"]["weight_window_days"]
    rows = []
    for subset, ids in subsets.items():
        for fold in folds:
            p = preds.loc[preds["fold"] == fold.name]
            train = data.loc[data["d"] <= fold.train_end]
            if ids is not None:
                p = p.loc[p["series_id"].astype(str).isin(ids)]
                train = train.loc[train["series_id"].astype(str).isin(ids)]
            rows.append(
                {
                    "subset": subset,
                    "fold": fold.name,
                    **evaluate_fold(p, train, fold.train_end, window),
                }
            )
    return pd.DataFrame(rows)


def update_results(new_rows: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Replace previous rows of the same (model, level), then rebuild the leaderboard."""
    path = results_dir / METRICS_FILE
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(["model", "level"]).index.isin(
            new_rows.set_index(["model", "level"]).index.unique()
        )
        new_rows = pd.concat([old.loc[keep], new_rows], ignore_index=True)
    new_rows = new_rows.sort_values(["level", "subset", "model", "fold"])
    new_rows.to_csv(path, index=False)

    grouped = new_rows.groupby(["model", "level", "subset"])
    board = grouped[METRIC_COLS].agg(["mean", "std"])
    board.columns = [f"{m}_{s}" for m, s in board.columns]
    board["n_folds"] = grouped.size()
    board["runtime_sec_total"] = grouped["runtime_sec"].sum()
    board = board.reset_index()
    board["sort_key"] = board["wrmsse_mean"].fillna(board["rmsse_mean"])
    board = board.sort_values(["level", "subset", "sort_key"]).drop(columns="sort_key")
    board.to_csv(results_dir / LEADERBOARD_FILE, index=False)
    return board


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--level", default="item", choices=["item", "agg"])
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Smoke test: N random series, first fold only, nothing saved",
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
    folds = generate_folds(config)
    data = load_level_data(config, args.level)
    subsets = subsets_for(args.level, config)

    if args.sample:
        rng = np.random.default_rng(config["seed"])
        ids = rng.choice(
            np.sort(data["series_id"].astype(str).unique()), args.sample, replace=False
        )
        data = data.loc[data["series_id"].astype(str).isin(ids)]
        folds = folds[:1]
        subsets = {"all": None}
        log.info(
            "SMOKE TEST: %d series, fold %s only, results are NOT saved", args.sample, folds[0].name
        )

    log.info(
        "Model=%s level=%s | %d series | %d rows | folds=%s",
        args.model,
        args.level,
        data["series_id"].nunique(),
        len(data),
        [f.name for f in folds],
    )
    result = run_backtest(MODEL_REGISTRY[args.model], data, folds)
    metrics = score(result.preds, data, folds, subsets, config)
    metrics.insert(0, "model", args.model)
    metrics.insert(1, "level", args.level)
    metrics["runtime_sec"] = metrics["fold"].map(result.runtime_sec)
    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")
    metrics["config_hash"] = config_hash(config)

    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        log.info(
            "Metrics per fold:\n%s",
            metrics.drop(columns=["timestamp", "config_hash"]).to_string(index=False),
        )

    if args.sample:
        return

    results_dir = resolve_path(config, "results")
    preds_dir = results_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{args.model}_{args.level}.parquet"
    result.preds.to_parquet(preds_path, index=False)
    board = update_results(metrics, results_dir)

    mine = board[(board["model"] == args.model) & (board["level"] == args.level)]
    cols = ["subset", "wrmsse_mean", "wrmsse_std", "rmsse_mean", "wape_mean", "bias_mean"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        log.info(
            "CV summary (mean over %d folds):\n%s", len(folds), mine[cols].to_string(index=False)
        )
    log.info("Saved: %s", preds_path)
    log.info("Updated: %s and %s", results_dir / METRICS_FILE, results_dir / LEADERBOARD_FILE)


if __name__ == "__main__":
    main()
