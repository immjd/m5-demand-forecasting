"""Single entry point that builds a model factory from its name, level and the config."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd

from forecasting.config import resolve_path
from forecasting.models.base import ForecastModel
from forecasting.models.baselines import MovingAverage, Naive, SeasonalNaive

BASELINES = {"naive": Naive, "seasonal_naive": SeasonalNaive, "moving_average": MovingAverage}
STATISTICAL = ("ets", "sarimax", "prophet")
ALL_MODELS = (*BASELINES, *STATISTICAL)

SARIMA_ORDERS_FILE = "sarima_orders.json"
ITEM_CLASSES_FILE = "item_classes.parquet"


def load_sarima_orders(config: dict[str, Any], level: str) -> dict[str, Any]:
    """Map ``series_id -> order`` from ``reports/results/sarima_orders.json``.

    Aggregate series use their own order; items use the order of their demand class.
    """
    from forecasting.models.statistical import str_to_order

    path = resolve_path(config, "results") / SARIMA_ORDERS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m scripts.select_sarima_order`")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if level == "agg":
        return {sid: str_to_order(o) for sid, o in saved["agg"].items()}
    classes = pd.read_parquet(resolve_path(config, "processed") / ITEM_CLASSES_FILE)
    by_class = {cls: str_to_order(o) for cls, o in saved["item_class"].items()}
    return {
        row.item_id: by_class[row.demand_class]
        for row in classes.itertuples()
        if row.demand_class in by_class
    }


def make_model(name: str, level: str, config: dict[str, Any]) -> Callable[[], ForecastModel]:
    """Return a zero-argument factory (a fresh model is built for every fold)."""
    if name in BASELINES:
        return BASELINES[name]

    s = config["statistical"]
    common = {
        "window": s["train_window_days"],
        "season": s["season"],
        "min_history": s["min_history_days"],
        "n_jobs": s["n_jobs"],
    }
    if name == "ets":
        from forecasting.models.statistical import ETSModel

        return partial(ETSModel, select_trend=(level == "agg"), **common)
    if name == "sarimax":
        from forecasting.models.statistical import SARIMAXModel

        orders = load_sarima_orders(config, level)
        return partial(SARIMAXModel, orders=orders, maxiter=s["sarima_maxiter"], **common)
    if name == "prophet":
        from forecasting.models.prophet_model import ProphetModel

        return partial(ProphetModel, **common)
    raise ValueError(f"Unknown model {name!r}; choose from {ALL_MODELS}")
