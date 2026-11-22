"""Leakage and correctness tests for the LightGBM features.

Core rule (CLAUDE.md #3): a feature on day t may only use sales from days <= t - 28.
The end-to-end test changes every sales value after a cutoff and checks that all
features in the following 28-day forecast window stay identical.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.config import load_config
from forecasting.data.preprocess import add_zero_run_len
from forecasting.features.build import build_features, feature_columns
from forecasting.features.calendar import calendar_features
from forecasting.features.lags import add_lag_features, add_rolling_features
from forecasting.features.price import item_mean_price, relative_price

N_DAYS, CUTOFF, H = 160, 100, 28


def _config() -> dict:
    cfg = load_config()
    cfg["lgbm"]["lags"] = [28, 35, 42]
    cfg["lgbm"]["rolling_windows"] = [7, 28]
    return cfg


def _calendar() -> pd.DataFrame:
    dates = pd.date_range("2015-01-03", periods=N_DAYS)  # a Saturday -> wday 1
    events = [None] * N_DAYS
    events[20], events[115] = "SuperBowl", "Easter"
    return pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": 11500 + np.arange(N_DAYS) // 7,
            "wday": (np.arange(N_DAYS) % 7) + 1,
            "month": dates.month,
            "year": dates.year,
            "d": np.arange(1, N_DAYS + 1),
            "event_name_1": events,
            "event_type_1": [None if e is None else "Cultural" for e in events],
            "event_name_2": None,
            "event_type_2": None,
            "snap": (dates.day <= 10).astype(int),
        }
    )


def _sales(seed: int = 0, closed=(60,)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item, dept in [
        ("FOODS_1_001", "FOODS_1"),
        ("FOODS_1_002", "FOODS_1"),
        ("HOBBIES_1_001", "HOBBIES_1"),
    ]:
        price = np.repeat(rng.choice([1.0, 1.5, 2.0], N_DAYS // 7 + 1), 7)[:N_DAYS]
        for d in range(1, N_DAYS + 1):
            rows.append(
                {
                    "series_id": item,
                    "dept_id": dept,
                    "cat_id": dept.split("_")[0],
                    "d": d,
                    "sales": int(rng.poisson(2)),
                    "sell_price": price[d - 1],
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.Timestamp("2015-01-03") + pd.to_timedelta(df["d"] - 1, unit="D")
    df["is_closed"] = df["d"].isin(closed).astype("int8")
    df.loc[df["is_closed"] == 1, "sales"] = 0
    return df


def _classes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["FOODS_1_001", "FOODS_1_002", "HOBBIES_1_001"],
            "demand_class": ["smooth", "intermittent", "lumpy"],
        }
    )


def _features(sales: pd.DataFrame) -> pd.DataFrame:
    sales = add_zero_run_len(sales.rename(columns={"series_id": "item_id"}))
    sales = sales.rename(columns={"item_id": "series_id"})
    return build_features(sales, _calendar(), _classes(), _config())


# ------------------------------------------------------------------ end-to-end leakage


def test_features_in_forecast_window_ignore_future_sales() -> None:
    """Change all sales after the cutoff -> features for days cutoff+1..cutoff+28 unchanged."""
    base = _sales(seed=0)
    altered = base.copy()
    future = altered["d"] > CUTOFF
    altered.loc[future, "sales"] = np.random.default_rng(99).poisson(50, future.sum())

    cols = [c for c in feature_columns(_config()) if c != "price_rel_item_mean"]
    window = lambda f: f[f["d"].between(CUTOFF + 1, CUTOFF + H)].reset_index(drop=True)  # noqa: E731
    a, b = window(_features(base))[cols], window(_features(altered))[cols]
    pd.testing.assert_frame_equal(a, b)

    # sanity: the alteration is visible right after the window (lag_28 at d = cutoff+29)
    after = lambda f: f.loc[f["d"] == CUTOFF + H + 1, "lag_28"].to_numpy()  # noqa: E731
    assert not np.array_equal(after(_features(base)), after(_features(altered)))


# ------------------------------------------------------------------ unit checks


def test_lag_and_rolling_values() -> None:
    df = _sales(closed=())
    f = add_rolling_features(add_lag_features(df, [28], 28), [7], shift=28)
    one = f[f["series_id"] == "FOODS_1_001"].set_index("d")
    s = one["sales"].astype(float)
    assert one.loc[80, "lag_28"] == s.loc[52]
    assert one.loc[80, "roll_mean_7"] == pytest.approx(s.loc[46:52].mean())
    assert np.isnan(one.loc[28, "lag_28"])  # no history yet


def test_lag_below_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="leak"):
        add_lag_features(_sales(), [7, 28], min_lag=28)


def test_closed_days_never_become_lag_values() -> None:
    f = add_lag_features(_sales(closed=(60,)), [28], 28)
    assert f.loc[f["d"] == 88, "lag_28"].isna().all()  # d - 28 = 60 is closed


def test_lags_do_not_cross_items() -> None:
    f = add_lag_features(_sales(), [28], 28)
    first_days = f.groupby("series_id")["lag_28"].apply(lambda s: s.iloc[:28].isna().all())
    assert first_days.all()


def test_days_to_next_event() -> None:
    cal = calendar_features(_calendar()).set_index("d")
    assert cal.loc[21, "days_to_next_event"] == 0  # event day (index 20 -> d 21)
    assert cal.loc[18, "days_to_next_event"] == 3
    assert cal.loc[25, "days_since_last_event"] == 4
    assert cal.loc[1, "is_weekend"] == 1 and cal.loc[3, "is_weekend"] == 0


def test_item_mean_price_uses_train_only() -> None:
    df = _sales()
    later = df.copy()
    later.loc[later["d"] > CUTOFF, "sell_price"] = 999.0
    pd.testing.assert_series_equal(item_mean_price(df, CUTOFF), item_mean_price(later, CUTOFF))
    rel = relative_price(later[later["d"] == CUTOFF + 1], item_mean_price(later, CUTOFF))
    assert (rel > 100).all()  # future price is used (known in advance), level is from train
