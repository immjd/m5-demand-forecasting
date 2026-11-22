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
# name -> (objective or None for "from tuned params", feature groups to drop)
LGBM_VARIANTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "lgbm_tweedie": ("tweedie", ()),
    "lgbm_poisson": ("poisson", ()),
    "lgbm_l2": ("regression", ()),
    "lgbm_tuned": (None, ()),
    "lgbm_no_price": (None, ("price",)),
    "lgbm_no_calendar": (None, ("calendar",)),
}
ALL_MODELS = (*BASELINES, *STATISTICAL, *LGBM_VARIANTS)

SARIMA_ORDERS_FILE = "sarima_orders.json"
LGBM_BEST_PARAMS_FILE = "lgbm_best_params.json"
ITEM_CLASSES_FILE = "item_classes.parquet"


def load_lgbm_best(config: dict[str, Any]) -> dict[str, Any]:
    """Read ``reports/results/lgbm_best_params.json`` (written by ``scripts.tune_lgbm``)."""
    path = resolve_path(config, "results") / LGBM_BEST_PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m scripts.tune_lgbm` first")
    return json.loads(path.read_text(encoding="utf-8"))


def make_lgbm(
    name: str,
    config: dict[str, Any],
    params_override: dict[str, Any] | None = None,
    objective: str | None = None,
    drop_groups: tuple[str, ...] = (),
    **overrides: Any,
) -> Callable[[], ForecastModel]:
    """Factory for a LightGBM variant (base params from config, optionally overridden)."""
    from forecasting.features.build import feature_columns
    from forecasting.models.lgbm import LGBMModel

    p = config["lgbm"]
    kwargs = {
        "features": feature_columns(config, drop_groups),
        "params": {**p["params"], **(params_override or {})},
        "objective": objective or p["objective"],
        "num_boost_round": p["num_boost_round"],
        "early_stopping_rounds": p["early_stopping_rounds"],
        "inner_valid_days": p["inner_valid_days"],
        "refit_full": p["refit_full"],
        "train_window_days": p["train_window_days"],
        "seed": config["seed"],
        "name": name,
    }
    kwargs.update(overrides)
    return partial(LGBMModel, **kwargs)


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
    if name in LGBM_VARIANTS:
        objective, drop = LGBM_VARIANTS[name]
        if objective is not None:
            return make_lgbm(name, config, objective=objective)
        best = load_lgbm_best(config)
        return make_lgbm(
            name,
            config,
            params_override=best["params"],
            objective=best["objective"],
            drop_groups=drop,
        )

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
