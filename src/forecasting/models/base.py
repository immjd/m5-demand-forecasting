"""Uniform interface shared by every forecasting model (baselines, statistical, LightGBM)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd


class ForecastModel(ABC):
    """Base class: ``fit`` on history, ``predict`` for a future frame without the target.

    Data contract (long format):
        * ``train``: one row per (series_id, d) up to the fold cutoff, including
          ``sales`` and ``is_closed`` plus any known covariates.
        * ``future``: rows for the forecast window with the same columns
          **except** ``sales`` (the backtest removes it).
        * ``predict`` returns ``series_id, d, y_pred`` for every row of ``future``.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Learn from history (data with ``d <= cutoff`` only)."""

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Return ``series_id, d, y_pred`` for every row in ``future``."""
