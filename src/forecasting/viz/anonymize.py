"""Disguise item-level examples before they appear in public notebooks or figures.

The M5 rules (section 7B) do not allow the competition data to be published, so
single-item charts show a generic label, a relative day and an index instead of the
item name, the calendar date and the units sold. This reduces disclosure, but it is
not strong anonymisation: the relative shape of the series is still visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_label(position: int, demand_class: str | None = None) -> str:
    """Generic label by position: 0 -> ``Item A``, 1 -> ``Item B``, optionally with the class."""
    if not 0 <= position < 26:
        raise ValueError("position must be between 0 and 25")
    label = f"Item {chr(ord('A') + position)}"
    return f"{label} ({demand_class})" if demand_class else label


def index_base(values: pd.Series | np.ndarray) -> float:
    """Mean used as the 100 level; fails loudly when a series has no positive mean."""
    base = float(np.nanmean(np.asarray(values, dtype=float)))
    if not base > 0:
        raise ValueError("cannot index a series whose mean is not positive")
    return base


def to_index(values: pd.Series | np.ndarray, base: float) -> pd.Series | np.ndarray:
    """Rescale so that ``base`` becomes 100."""
    return 100.0 * values / base


def relative_days(dates: pd.Series | np.ndarray, start: object) -> np.ndarray:
    """Whole days since ``start`` (day 0), replacing the calendar date."""
    delta = pd.to_datetime(pd.Series(np.asarray(dates))) - pd.Timestamp(start)
    return delta.dt.days.to_numpy()
