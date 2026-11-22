"""Calendar features. All of them are known in advance, so they may use future dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "year",
    "is_weekend",
    "event_name_1",
    "event_type_1",
    "snap",
    "days_to_next_event",
    "days_since_last_event",
]


def _days_to_event(has_event: np.ndarray, forward: bool) -> np.ndarray:
    """Distance in days to the next (or previous) event day; 0 on event days, NaN if none."""
    n = len(has_event)
    out = np.full(n, np.nan)
    last = None
    order = range(n - 1, -1, -1) if forward else range(n)
    for i in order:
        if has_event[i]:
            last = i
        if last is not None:
            out[i] = abs(last - i)
    return out


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """One row per ``d`` with calendar features.

    Args:
        calendar: Output of :func:`forecasting.data.load.read_calendar` (the full
            M5 calendar, including future dates and scheduled events).
    """
    cal = calendar.sort_values("d").reset_index(drop=True)
    date = pd.to_datetime(cal["date"])
    has_event = cal["event_name_1"].notna().to_numpy()
    out = pd.DataFrame(
        {
            "d": cal["d"].astype("int16"),
            "day_of_week": cal["wday"].astype("int8"),  # 1 = Saturday ... 7 = Friday (M5)
            "day_of_month": date.dt.day.astype("int8"),
            "week_of_year": date.dt.isocalendar().week.astype("int8").to_numpy(),
            "month": date.dt.month.astype("int8"),
            "year": date.dt.year.astype("int16"),
            "is_weekend": cal["wday"].isin([1, 2]).astype("int8"),
            "event_name_1": cal["event_name_1"].astype("category"),
            "event_type_1": cal["event_type_1"].astype("category"),
            "snap": cal["snap"].astype("int8"),
            "days_to_next_event": _days_to_event(has_event, forward=True).astype("float32"),
            "days_since_last_event": _days_to_event(has_event, forward=False).astype("float32"),
        }
    )
    return out
