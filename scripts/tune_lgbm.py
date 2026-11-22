"""Phase 5: tune LightGBM with Optuna on the CV folds F1-F4 (never the test window).

Objective = mean WRMSSE-lite over the folds (all items). The cumulative mean is
reported after every fold so clearly bad trials are pruned early (MedianPruner).
The study is stored in SQLite under data/processed so an interrupted run can
be resumed by running the same command again.

Usage (from the project root):
    python -m scripts.tune_lgbm --objective tweedie            # n_optuna_trials from config
    python -m scripts.tune_lgbm --objective tweedie --trials 2 --folds F1   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from forecasting.config import load_config, resolve_path
from forecasting.models.registry import LGBM_BEST_PARAMS_FILE, make_lgbm
from forecasting.validation.backtest import generate_folds, load_feature_data, run_backtest
from scripts.run_backtest import score

STUDY_DB = "optuna_lgbm.db"
TRIALS_FILE = "optuna_trials_lgbm.csv"
log = logging.getLogger("tune_lgbm")


def suggest_params(trial: optuna.Trial, objective: str) -> dict[str, Any]:
    """Search space (PROJECT_PLAN Phase 5 task 5)."""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.5)
    return params


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--objective", default=None, help="LightGBM objective (default: config)")
    parser.add_argument("--trials", type=int, default=None, help="Default: lgbm.n_optuna_trials")
    parser.add_argument("--folds", nargs="+", default=None, help="Smoke test: subset of folds")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config = load_config(args.config)
    objective = args.objective or config["lgbm"]["objective"]
    n_trials = args.trials or config["lgbm"]["n_optuna_trials"]
    folds = generate_folds(config)
    smoke = args.folds is not None
    if smoke:
        folds = [f for f in folds if f.name in args.folds]
    data = load_feature_data(config)
    log.info(
        "Tuning objective=%s | %d trials | folds=%s | %s rows",
        objective,
        n_trials,
        [f.name for f in folds],
        f"{len(data):,}",
    )

    def run_trial(trial: optuna.Trial) -> float:
        params = suggest_params(trial, objective)
        factory = make_lgbm("lgbm_trial", config, params_override=params, objective=objective)
        scores = []
        t0 = time.perf_counter()
        for step, fold in enumerate(folds):
            result = run_backtest(factory, data, [fold])
            m = score(result.preds, data, [fold], {"all": None}, config)
            scores.append(float(m["wrmsse"].iloc[0]))
            running = sum(scores) / len(scores)
            trial.report(running, step)
            if trial.should_prune():
                log.info(
                    "trial %d pruned after %s (running WRMSSE %.4f)",
                    trial.number,
                    fold.name,
                    running,
                )
                raise optuna.TrialPruned()
        mean = sum(scores) / len(scores)
        trial.set_user_attr("fold_wrmsse", scores)
        log.info(
            "trial %d: WRMSSE %.4f (folds %s) in %.0fs | %s",
            trial.number,
            mean,
            [round(s, 4) for s in scores],
            time.perf_counter() - t0,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()},
        )
        return mean

    storage = None
    if not smoke:
        db = resolve_path(config, "processed") / STUDY_DB
        storage = f"sqlite:///{db.as_posix()}"
    study = optuna.create_study(
        study_name=f"lgbm_{objective}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        load_if_exists=True,
    )
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining <= 0:
        log.info("Study already has %d finished trials", len(study.trials))
    else:
        study.optimize(run_trial, n_trials=remaining)

    trials = study.trials_dataframe()
    best = study.best_trial
    log.info("Best trial %d: WRMSSE %.4f | %s", best.number, best.value, best.params)
    if smoke:
        log.info("Smoke test: nothing saved")
        return

    results = resolve_path(config, "results")
    trials.to_csv(results / TRIALS_FILE, index=False)
    out = {
        "objective": objective,
        "params": best.params,
        "cv_wrmsse": best.value,
        "fold_wrmsse": best.user_attrs.get("fold_wrmsse"),
        "trial": best.number,
        "n_trials": len(study.trials),
        "n_pruned": int((trials["state"] == "PRUNED").sum()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (results / LGBM_BEST_PARAMS_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved: %s and %s", results / LGBM_BEST_PARAMS_FILE, results / TRIALS_FILE)


if __name__ == "__main__":
    main()
