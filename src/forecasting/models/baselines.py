"""Simple baselines: Naive, Seasonal Naive and Moving Average.

All of them ignore closed days (``is_closed == 1``) in the history and predict 0
for series that have no history before the cutoff.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from forecasting.models.base import ForecastModel


def _open_history(train: pd.DataFrame) -> pd.DataFrame:
    """Open-day history sorted by series and day, with ``series_id`` as plain strings."""
    hist = train.loc[train["is_closed"] == 0, ["series_id", "d", "sales"]].copy()
    hist["series_id"] = hist["series_id"].astype(str)
    return hist.sort_values(["series_id", "d"])


def _attach(future: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Broadcast one value per series onto every future row (0 for unseen series)."""
    out = future[["series_id", "d"]].copy()
    mapped = out["series_id"].astype(str).map(values)
    out["y_pred"] = mapped.astype("float64").fillna(0.0).to_numpy()
    return out


class Naive(ForecastModel):
    """Repeat the last observed (open-day) value."""

    name = "naive"

    def fit(self, train: pd.DataFrame) -> Self:
        hist = _open_history(train)
        self.last_ = hist.groupby("series_id", observed=True)["sales"].last()
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        return _attach(future, self.last_)


class MovingAverage(ForecastModel):
    """Mean of the last ``window`` open days."""

    name = "moving_average"

    def __init__(self, window: int = 28) -> None:
        self.window = window

    def fit(self, train: pd.DataFrame) -> Self:
        hist = _open_history(train)
        tail = hist.groupby("series_id", observed=True).tail(self.window)
        self.mean_ = tail.groupby("series_id", observed=True)["sales"].mean()
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        return _attach(future, self.mean_)


class SeasonalNaive(ForecastModel):
    """Repeat the value from the same position in the last season (default: same weekday).

    For a future day ``d`` the reference is the latest ``d' <= cutoff`` with
    ``d' = d - k * season``. If that day is closed or missing, the model steps
    back one more season (up to ``max_lookback`` seasons), then falls back to
    the last open value, then to 0.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 7, max_lookback: int = 4) -> None:
        self.season = season
        self.max_lookback = max_lookback

    def fit(self, train: pd.DataFrame) -> Self:
        self.cutoff_ = int(train["d"].max())
        hist = _open_history(train)
        recent = hist.loc[hist["d"] > self.cutoff_ - self.season * (self.max_lookback + 1)]
        self.lookup_ = recent.set_index(["series_id", "d"])["sales"]
        self.last_ = hist.groupby("series_id", observed=True)["sales"].last()
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        out = future[["series_id", "d"]].copy()
        series = out["series_id"].astype(str)
        d = out["d"].to_numpy().astype(np.int64)

        steps_back = np.ceil((d - self.cutoff_) / self.season).astype(np.int64)
        y = np.full(len(out), np.nan)
        for k in range(self.max_lookback):
            todo = np.isnan(y)
            if not todo.any():
                break
            ref = d - self.season * (steps_back + k)
            idx = pd.MultiIndex.from_arrays([series.to_numpy()[todo], ref[todo]])
            y[todo] = self.lookup_.reindex(idx).to_numpy(dtype="float64")

        fallback = series.map(self.last_).astype("float64").to_numpy()
        y = np.where(np.isnan(y), fallback, y)
        out["y_pred"] = np.nan_to_num(y, nan=0.0)
        return out


MODEL_REGISTRY = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
}
