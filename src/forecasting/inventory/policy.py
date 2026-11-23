"""Periodic-review, order-up-to (R, S) policy."""

from __future__ import annotations

import numpy as np


def extend_forecast(forecast: np.ndarray, extra: int, season: int = 7) -> np.ndarray:
    """Append ``extra`` days by repeating the same weekday of the previous week.

    Used when the last review of a period needs a few days beyond the forecast
    horizon of that period's forecast vintage (the next vintage would use newer data).
    """
    out = np.asarray(forecast, dtype=float)
    for _ in range(extra):
        out = np.c_[out, out[:, -season]]
    return out


def order_up_to_levels(
    forecast: np.ndarray, safety_stock: np.ndarray, review_period: int, lead_time: int
) -> tuple[np.ndarray, np.ndarray]:
    """Order-up-to level S on every review day of one forecast period.

    Reviews happen on day 0, R, 2R, ... of the period. An order placed on review
    day t must cover demand of days t .. t + R + L - 1 (until the next order lands).

    Args:
        forecast: (items, days) daily forecasts of the period's vintage.
        safety_stock: (items,) units added on top of the cumulative forecast.

    Returns:
        ``(S, review)``: S is NaN on non-review days, never below zero, and rounded up to
        whole units (physical items cannot be ordered or sold in fractions).
    """
    n, days = forecast.shape
    review = np.arange(days) % review_period == 0
    protect = review_period + lead_time
    last = np.flatnonzero(review)[-1]
    f = extend_forecast(forecast, max(0, last + protect - days))
    S = np.full((n, days), np.nan)
    for t in np.flatnonzero(review):
        target = np.maximum(f[:, t : t + protect].sum(axis=1) + safety_stock, 0.0)
        S[:, t] = np.ceil(target - 1e-9)  # whole units; tolerance absorbs float noise
    return S, review


def order_quantity(order_up_to: np.ndarray, inventory_position: np.ndarray) -> np.ndarray:
    """Order quantity = max(0, S - inventory position)."""
    return np.maximum(order_up_to - inventory_position, 0.0)
