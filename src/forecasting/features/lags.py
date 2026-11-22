"""Lag and rolling features on the target, always shifted by at least ``min_lag`` days.

Direct multi-step strategy: a feature on day t may only use sales from days
<= t - min_lag (min_lag = horizon = 28), so one model can forecast the whole
28-day window without feeding its own predictions back.
"""

from __future__ import annotations

import pandas as pd

GROUP = "series_id"


def check_contiguous(df: pd.DataFrame) -> None:
    """Raise if any series has missing days (row shifts must equal day shifts)."""
    gaps = df.groupby(GROUP, observed=True)["d"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError("Series are not contiguous daily rows; sort by series_id, d first")


def masked_target(df: pd.DataFrame, target: str = "sales") -> pd.Series:
    """Target as float with closed days set to NaN, so store closures never become lag values."""
    return df[target].astype("float32").where(df["is_closed"] == 0)


def add_lag_features(df: pd.DataFrame, lags: list[int], min_lag: int) -> pd.DataFrame:
    """Add ``lag_<L>`` columns. Every L must be >= ``min_lag``."""
    if min(lags) < min_lag:
        raise ValueError(f"lag {min(lags)} < min_lag {min_lag}: would leak the forecast window")
    out = df.copy()
    y = masked_target(out)
    g = y.groupby(out[GROUP], observed=True)
    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag).astype("float32")
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int], shift: int) -> pd.DataFrame:
    """Add ``roll_mean_<w>``, ``roll_std_<w>`` on the target shifted by ``shift`` days,
    plus ``ratio_mean_7_28`` (short vs. medium level) when both windows exist.
    """
    out = df.copy()
    shifted = masked_target(out).groupby(out[GROUP], observed=True).shift(shift)
    g = shifted.groupby(out[GROUP], observed=True)
    for w in windows:
        roll = g.rolling(w, min_periods=1)
        out[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        out[f"roll_std_{w}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
    if {7, 28} <= set(windows):
        out["ratio_mean_7_28"] = (out["roll_mean_7"] / (out["roll_mean_28"] + 1e-3)).astype(
            "float32"
        )
    return out


def add_shifted(df: pd.DataFrame, column: str, shift: int, name: str) -> pd.DataFrame:
    """Add ``name`` = ``column`` shifted by ``shift`` days within each series."""
    out = df.copy()
    shifted = out[column].groupby(out[GROUP], observed=True).shift(shift)
    out[name] = shifted.astype("float32")
    return out


def lag_feature_names(lags: list[int], windows: list[int]) -> list[str]:
    """Names produced by :func:`add_lag_features` and :func:`add_rolling_features`."""
    names = [f"lag_{lag}" for lag in lags]
    for w in windows:
        names += [f"roll_mean_{w}", f"roll_std_{w}"]
    if {7, 28} <= set(windows):
        names.append("ratio_mean_7_28")
    return names
