"""Phase 6: the single evaluation on the held-out test window d_1914..d_1941.

Candidates were locked from the CV results before the test window was opened
(docs D-025). Every model is retrained on all data up to d_1913 with the
hyperparameters / SARIMA orders chosen on F1-F4 and is never tuned again.

The script refuses to run without ``--confirm-final-test`` and refuses to run a
second time when ``reports/results/test_results.csv`` already exists. A re-run
(e.g. after a bug) needs ``--rerun-reason "<why>"`` and must be logged in
docs/DECISIONS.md.

Usage (from the project root, once):
    python -m scripts.final_test --confirm-final-test
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.config import load_config, resolve_path
from forecasting.models.registry import make_model
from forecasting.validation.backtest import (
    get_test_fold,
    load_feature_data,
    load_level_data,
    run_backtest,
)
from scripts.run_backtest import config_hash, score, score_bottom_up, subsets_for

TEST_RESULTS_FILE = "test_results.csv"
# (model, level) pairs locked before the test window was opened (D-025)
CANDIDATES: list[tuple[str, str]] = [
    ("seasonal_naive", "item"),
    ("moving_average", "item"),
    ("sarimax", "item"),  # 200-item sample only
    ("lgbm_tuned", "item"),  # aggregate scores via bottom-up
    ("seasonal_naive", "agg"),
    ("moving_average", "agg"),
    ("sarimax", "agg"),
]
log = logging.getLogger("final_test")


def check_guard(confirmed: bool, out_path: Path, rerun_reason: str | None) -> None:
    """Stop unless the run is explicitly confirmed and is the first one (or a logged re-run)."""
    if not confirmed:
        raise SystemExit(
            "Refusing to touch the test window: pass --confirm-final-test (run this only once)."
        )
    if out_path.exists() and not rerun_reason:
        raise SystemExit(
            f"{out_path} already exists: the test set was already evaluated. A re-run needs "
            '--rerun-reason "<why>" and an entry in docs/DECISIONS.md.'
        )


def _stamp(metrics: pd.DataFrame, model: str, level: str, runtime: float) -> pd.DataFrame:
    metrics.insert(0, "model", model)
    metrics.insert(1, "level", level)
    metrics["runtime_sec"] = runtime
    return metrics


def run_candidate(
    model: str,
    level: str,
    config: dict[str, Any],
    item_data: pd.DataFrame,
    agg_data: pd.DataFrame,
    features: pd.DataFrame,
    preds_dir: Path,
) -> list[pd.DataFrame]:
    """Retrain one candidate on d <= 1913, forecast the test window, return metric rows."""
    test = get_test_fold(config)
    is_lgbm = model.startswith("lgbm")
    subsets = subsets_for(level, config)
    if is_lgbm:
        data = features
    elif level == "agg":
        data = agg_data
    else:
        data = item_data
    if model == "sarimax" and level == "item":
        data = data.loc[data["series_id"].astype(str).isin(subsets["stat_sample"])]
        subsets = {"stat_sample": subsets["stat_sample"]}

    log.info(
        "%s @ %s | %d series | train d <= %d -> forecast d_%d..d_%d",
        model,
        level,
        data["series_id"].nunique(),
        test.train_end,
        test.valid_start,
        test.valid_end,
    )
    result = run_backtest(make_model(model, level, config), data, [test], keep_models=is_lgbm)
    runtime = result.runtime_sec[test.name]
    rows = [_stamp(score(result.preds, data, [test], subsets, config), model, level, runtime)]
    result.preds.to_parquet(preds_dir / f"test_{model}_{level}.parquet", index=False)
    if is_lgbm:
        agg_pred, agg_metrics = score_bottom_up(result, data, [test], config, include_test=True)
        agg_pred.to_parquet(preds_dir / f"test_{model}_agg.parquet", index=False)
        rows.append(_stamp(agg_metrics, model, "agg", runtime))
        imp = result.models[test.name].importance_
        imp.to_frame("gain").to_csv(
            resolve_path(config, "results") / "importance_test_lgbm.csv", index_label="feature"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--rerun-reason", default=None)
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
    out_path = results / TEST_RESULTS_FILE
    check_guard(args.confirm_final_test, out_path, args.rerun_reason)
    if args.rerun_reason:
        log.warning("RE-RUN of the final test. Reason: %s", args.rerun_reason)

    preds_dir = results / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    item_data = load_level_data(config, "item", include_test=True)
    agg_data = load_level_data(config, "agg", include_test=True)
    features = load_feature_data(config, include_test=True)

    rows: list[pd.DataFrame] = []
    for model, level in CANDIDATES:
        rows += run_candidate(model, level, config, item_data, agg_data, features, preds_dir)

    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = datetime.now().isoformat(timespec="seconds")
    out["config_hash"] = config_hash(config)
    out["rerun_reason"] = args.rerun_reason or ""
    out = out.sort_values(["level", "subset", "model"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    cols = ["model", "level", "subset", "wrmsse", "rmsse", "wape", "bias", "runtime_sec"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        log.info("TEST results (d_1914..d_1941):\n%s", out[cols].to_string(index=False))
    log.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
