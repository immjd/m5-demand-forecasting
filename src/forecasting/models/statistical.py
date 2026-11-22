"""Per-series statistical models (ETS, SARIMAX) fitted in parallel with joblib.

Every series is fitted independently on the last ``window`` days before the
cutoff. Closed days are filled with the value of the same weekday one week
earlier so the daily index stays regular. If a fit fails, returns non-finite
values, or the history is too short, the series falls back to the mean of its
last 28 open days (the item-level benchmark) and the fallback is counted.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Callable
from typing import Any, Self

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecasting.models.base import ForecastModel

log = logging.getLogger(__name__)

Order = tuple[tuple[int, int, int], tuple[int, int, int]]
DEFAULT_ORDER: Order = ((1, 0, 1), (0, 1, 1))
FALLBACK_WINDOW = 28


# ---------------------------------------------------------------- helpers
def regular_history(group: pd.DataFrame, cutoff_d: int, season: int = 7) -> pd.Series:
    """Daily series indexed by ``d`` from the first day to the cutoff, closed days filled.

    Closed days take the value of ``d - season`` (0 if unavailable); missing
    days inside the range are treated as zero sales.
    """
    g = group.sort_values("d")
    sales = g.set_index("d")["sales"].astype("float64")
    closed = set(g.loc[g["is_closed"] == 1, "d"])
    y = sales.reindex(range(int(g["d"].min()), cutoff_d + 1), fill_value=0.0)
    for d in sorted(closed):
        y.loc[d] = y.get(d - season, 0.0)
    return y


def fallback_value(y: pd.Series | np.ndarray) -> float:
    """Mean of the last ``FALLBACK_WINDOW`` values (0 for an empty history)."""
    arr = np.asarray(y, dtype="float64")
    return float(arr[-FALLBACK_WINDOW:].mean()) if arr.size else 0.0


def _quiet() -> None:
    warnings.filterwarnings("ignore")
    for name in ("cmdstanpy", "prophet", "statsmodels"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _safe_run(worker: Callable[[dict], np.ndarray], payload: dict) -> tuple[np.ndarray, bool]:
    """Run one series forecast; on any problem return the fallback (flagged True)."""
    _quiet()
    steps = payload["steps"]
    try:
        fc = np.asarray(worker(payload), dtype="float64")
        if fc.shape != (steps,) or not np.all(np.isfinite(fc)):
            raise ValueError("bad forecast")
        return fc, False
    except Exception:  # noqa: BLE001 - any statsmodels/prophet failure -> fallback
        return np.full(steps, payload["fallback"]), True


# ---------------------------------------------------------------- base class
class PerSeriesModel(ForecastModel):
    """Shared fit/predict loop: store regular histories, forecast each series in parallel."""

    name = "per_series"

    def __init__(
        self,
        window: int = 730,
        season: int = 7,
        min_history: int = 56,
        n_jobs: int = -1,
    ) -> None:
        self.window = window
        self.season = season
        self.min_history = min_history
        self.n_jobs = n_jobs

    # subclasses implement these two
    worker: Callable[[dict], np.ndarray]

    def _payload(self, sid: str, y: pd.Series, future_days: pd.DataFrame) -> dict:
        return {"y": y.to_numpy()}

    def fit(self, train: pd.DataFrame) -> Self:
        self.cutoff_ = int(train["d"].max())
        recent = train.loc[train["d"] > self.cutoff_ - self.window]
        recent = recent.assign(series_id=recent["series_id"].astype(str))
        self.history_ = {
            sid: regular_history(g, self.cutoff_, self.season)
            for sid, g in recent.groupby("series_id", sort=True)
        }
        self.train_days_ = recent.drop_duplicates("d").sort_values("d")
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        out = future[["series_id", "d"]].copy()
        out["series_id"] = out["series_id"].astype(str)
        future_days = future.drop_duplicates("d").sort_values("d")
        steps = int(future_days["d"].max()) - self.cutoff_

        series = sorted(out["series_id"].unique())
        forecasts: dict[str, np.ndarray] = {}
        jobs, job_ids = [], []
        n_short = 0
        for sid in series:
            y = self.history_.get(sid)
            if y is None or len(y) < self.min_history:
                n_short += 1
                forecasts[sid] = np.full(steps, fallback_value(y if y is not None else []))
                continue
            payload = {"steps": steps, "fallback": fallback_value(y)}
            payload.update(self._payload(sid, y, future_days))
            jobs.append(delayed(_safe_run)(type(self).worker, payload))
            job_ids.append(sid)

        results = Parallel(n_jobs=self.n_jobs)(jobs) if jobs else []
        n_failed = 0
        for sid, (fc, failed) in zip(job_ids, results, strict=True):
            forecasts[sid] = fc
            n_failed += failed
        self.n_fallback_ = n_short + n_failed
        log.info(
            "%s: %d series forecast, fallback MA28 for %d (short history %d, failed fit %d)",
            self.name,
            len(series),
            self.n_fallback_,
            n_short,
            n_failed,
        )

        step_idx = out["d"].to_numpy() - self.cutoff_ - 1
        out["y_pred"] = [forecasts[s][i] for s, i in zip(out["series_id"], step_idx, strict=True)]
        return out


# ---------------------------------------------------------------- ETS
def _ets_worker(p: dict) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    best, best_aicc = None, np.inf
    for trend in p["trends"]:
        res = ExponentialSmoothing(
            p["y"],
            trend=trend,
            damped_trend=trend is not None,
            seasonal="add",
            seasonal_periods=p["season"],
            initialization_method="estimated",
        ).fit()
        aicc = float(np.nan_to_num(res.aicc, nan=np.inf))
        if best is None or aicc < best_aicc:
            best, best_aicc = res, aicc
    return best.forecast(p["steps"])


class ETSModel(PerSeriesModel):
    """Additive-seasonal exponential smoothing (period 7).

    With ``select_trend=True`` (aggregate level) a damped additive trend is
    tried as well and the lower AICc wins per series; items use no trend.
    """

    name = "ets"
    worker = staticmethod(_ets_worker)

    def __init__(self, select_trend: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.select_trend = select_trend

    def _payload(self, sid: str, y: pd.Series, future_days: pd.DataFrame) -> dict:
        trends = [None, "add"] if self.select_trend else [None]
        return {"y": y.to_numpy(), "trends": trends, "season": self.season}


# ---------------------------------------------------------------- SARIMAX
def fit_sarimax(y: np.ndarray, order: Order, season: int, maxiter: int = 50):
    """Fit one SARIMAX model and return the statsmodels results object."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    (p, d, q), (sp, sd, sq) = order
    model = SARIMAX(
        y,
        order=(p, d, q),
        seasonal_order=(sp, sd, sq, season),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=maxiter)


def _sarimax_worker(p: dict) -> np.ndarray:
    res = fit_sarimax(p["y"], p["order"], p["season"], p["maxiter"])
    return res.forecast(p["steps"])


class SARIMAXModel(PerSeriesModel):
    """SARIMA with a fixed order per series (orders chosen beforehand on fold F1)."""

    name = "sarimax"
    worker = staticmethod(_sarimax_worker)

    def __init__(
        self,
        orders: dict[str, Order] | None = None,
        default_order: Order = DEFAULT_ORDER,
        maxiter: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.orders = orders or {}
        self.default_order = default_order
        self.maxiter = maxiter

    def _payload(self, sid: str, y: pd.Series, future_days: pd.DataFrame) -> dict:
        return {
            "y": y.to_numpy(),
            "order": self.orders.get(sid, self.default_order),
            "season": self.season,
            "maxiter": self.maxiter,
        }


# ---------------------------------------------------------------- order selection
def sarima_grid(grid: dict[str, list[int]]) -> list[Order]:
    """Expand a ``{p, d, q, P, D, Q}`` grid into a list of orders."""
    combos = itertools.product(grid["p"], grid["d"], grid["q"], grid["P"], grid["D"], grid["Q"])
    return [((p, d, q), (sp, sd, sq)) for p, d, q, sp, sd, sq in combos]


def _aic_job(y: np.ndarray, order: Order, season: int, maxiter: int) -> float:
    _quiet()
    try:
        aic = float(fit_sarimax(y, order, season, maxiter).aic)
        return aic if np.isfinite(aic) else np.inf
    except Exception:  # noqa: BLE001
        return np.inf


def aic_table(
    series: dict[str, np.ndarray],
    orders: list[Order],
    season: int = 7,
    maxiter: int = 50,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """AIC for every (order, series) pair; failed fits get +inf.

    Returns:
        DataFrame with one row per order (index = order as string) and one
        column per series.
    """
    pairs = [(o, sid) for o in orders for sid in series]
    aics = Parallel(n_jobs=n_jobs)(
        delayed(_aic_job)(series[sid], o, season, maxiter) for o, sid in pairs
    )
    table = pd.DataFrame(
        np.array(aics).reshape(len(orders), len(series)),
        index=[order_to_str(o) for o in orders],
        columns=list(series),
    )
    return table


def best_order(table: pd.DataFrame, columns: list[str] | None = None) -> Order:
    """Order with the lowest summed AIC over ``columns`` (all by default)."""
    cols = columns or list(table.columns)
    return str_to_order(table[cols].sum(axis=1).idxmin())


def order_to_str(order: Order) -> str:
    (p, d, q), (sp, sd, sq) = order
    return f"({p},{d},{q})({sp},{sd},{sq})"


def str_to_order(text: str) -> Order:
    nums = [int(c) for c in text if c.isdigit()]
    return (nums[0], nums[1], nums[2]), (nums[3], nums[4], nums[5])
