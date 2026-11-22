"""Rolling-origin (expanding window) backtest: folds, level data and the evaluation loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forecasting.config import resolve_path
from forecasting.models.base import ForecastModel

TARGET = "sales"
SALES_LONG_FILE = "sales_long.parquet"
AGGREGATES_FILE = "aggregates.parquet"
CALENDAR_COLS = [
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
]


@dataclass(frozen=True)
class Fold:
    """One backtest split, in ``d`` indices: train on ``d <= train_end``."""

    name: str
    train_end: int
    valid_start: int
    valid_end: int

    def __post_init__(self) -> None:
        if not self.train_end < self.valid_start <= self.valid_end:
            raise ValueError(f"Invalid fold {self}: need train_end < valid_start <= valid_end")

    @property
    def horizon(self) -> int:
        return self.valid_end - self.valid_start + 1


@dataclass
class BacktestResult:
    """Long predictions ``[fold, series_id, d, y_true, y_pred, ...]`` plus fit/predict time."""

    preds: pd.DataFrame
    runtime_sec: dict[str, float] = field(default_factory=dict)


def generate_folds(config: dict[str, Any]) -> list[Fold]:
    """Return the CV folds F1..Fk from the config. The test window is never included."""
    return [Fold(f"F{i}", *map(int, f)) for i, f in enumerate(config["backtest"]["folds"], 1)]


def get_test_fold(config: dict[str, Any]) -> Fold:
    """Return the held-out test fold. Only for the single final evaluation in Phase 6."""
    return Fold("TEST", *map(int, config["backtest"]["test"]))


def load_level_data(config: dict[str, Any], level: str, include_test: bool = False) -> pd.DataFrame:
    """Load the long table for ``level`` ('item' or 'agg') with a common ``series_id`` column.

    By default rows after the last validation day are dropped, so the test
    window cannot leak into model selection by accident.
    """
    last_d = (get_test_fold(config) if include_test else generate_folds(config)[-1]).valid_end
    if level == "item":
        path = resolve_path(config, "interim") / SALES_LONG_FILE
        data = pd.read_parquet(path, filters=[("d", "<=", last_d)])
        data = data.rename(columns={"item_id": "series_id"})
    elif level == "agg":
        path = resolve_path(config, "processed") / AGGREGATES_FILE
        data = pd.read_parquet(path, filters=[("d", "<=", last_d)])
        data = data.rename(columns={"level": "agg_level"})
        # Calendar covariates (events, SNAP) are shared by all series of the store.
        calendar = pd.read_parquet(
            resolve_path(config, "interim") / SALES_LONG_FILE,
            columns=["d", *CALENDAR_COLS],
            filters=[("d", "<=", last_d)],
        ).drop_duplicates("d")
        data = data.merge(calendar, on="d", how="left", validate="many_to_one")
    else:
        raise ValueError(f"Unknown level {level!r}; expected 'item' or 'agg'")
    return data.sort_values(["series_id", "d"]).reset_index(drop=True)


def run_backtest(
    model_factory: Callable[[], ForecastModel],
    data: pd.DataFrame,
    folds: list[Fold],
) -> BacktestResult:
    """Fit a fresh model per fold and predict its validation window.

    The target column is removed from the validation frame before ``predict``,
    and predictions are clipped at 0.
    """
    parts, runtimes = [], {}
    for fold in folds:
        train = data.loc[data["d"] <= fold.train_end]
        window = data.loc[data["d"].between(fold.valid_start, fold.valid_end)]
        future = window.drop(columns=[TARGET])

        t0 = time.perf_counter()
        model = model_factory()
        model.fit(train)
        pred = model.predict(future)
        runtimes[fold.name] = time.perf_counter() - t0

        keys = ["series_id", "d"]
        out = window[[*keys, TARGET, "is_closed"]].merge(
            pred[[*keys, "y_pred"]], on=keys, how="left", validate="one_to_one"
        )
        n_missing = int(out["y_pred"].isna().sum())
        if n_missing:
            raise ValueError(f"{model.name}: {n_missing} predictions missing in fold {fold.name}")

        out = out.rename(columns={TARGET: "y_true"})
        out["y_pred"] = out["y_pred"].clip(lower=0).astype("float32")
        out.insert(0, "fold", fold.name)
        out["model"] = model.name
        parts.append(out)

    return BacktestResult(pd.concat(parts, ignore_index=True), runtimes)
