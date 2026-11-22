"""Tests for the per-series statistical models (ETS, SARIMAX, Prophet) and order selection."""

import numpy as np
import pandas as pd
import pytest

from forecasting.models.prophet_model import ProphetModel, holidays_frame
from forecasting.models.statistical import (
    ETSModel,
    SARIMAXModel,
    aic_table,
    best_order,
    order_to_str,
    regular_history,
    sarima_grid,
    str_to_order,
)

N_TRAIN, H = 210, 14
PATTERN = np.array([10, 4, 3, 3, 4, 6, 12], dtype=float)  # weekly shape


def _frame(series: dict[str, np.ndarray], first_d: int = 1) -> pd.DataFrame:
    rows = [
        {"series_id": sid, "d": first_d + i, "sales": float(v)}
        for sid, values in series.items()
        for i, v in enumerate(values)
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.Timestamp("2015-01-01") + pd.to_timedelta(df["d"] - 1, unit="D")
    df["is_closed"] = 0
    df["snap"] = (df["date"].dt.day <= 10).astype(int)
    df["event_name_1"] = np.where(df["d"] % 50 == 0, "TestEvent", None)
    df["event_name_2"] = None
    return df


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    seasonal = np.tile(PATTERN, N_TRAIN // 7) + rng.normal(0, 0.5, N_TRAIN)
    full = _frame(
        {
            "S": np.concatenate([seasonal, np.tile(PATTERN, 2)]),
            "Z": np.zeros(N_TRAIN + H),
        }
    )
    short = _frame({"SHORT": np.ones(20 + H)}, first_d=N_TRAIN - 20 + 1)  # 20 train days
    df = pd.concat([full, short], ignore_index=True)
    train = df[df["d"] <= N_TRAIN]
    future = df[df["d"] > N_TRAIN].drop(columns="sales")
    return train, future


def _check(pred: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
    assert len(pred) == len(future)
    assert np.isfinite(pred["y_pred"]).all()
    return pred.set_index(["series_id", "d"])["y_pred"]


def test_regular_history_fills_closed_day() -> None:
    g = pd.DataFrame({"d": range(1, 15), "sales": range(1, 15), "is_closed": 0})
    g.loc[g["d"] == 12, ["sales", "is_closed"]] = [0, 1]
    y = regular_history(g, cutoff_d=14, season=7)
    assert y.loc[12] == 5  # value of d=5
    assert list(y.index) == list(range(1, 15))


@pytest.mark.parametrize(
    "model",
    [
        ETSModel(n_jobs=1, min_history=56),
        SARIMAXModel(orders={"S": ((1, 0, 0), (0, 1, 1))}, n_jobs=1, min_history=56),
    ],
    ids=["ets", "sarimax"],
)
def test_models_capture_weekly_pattern_and_fall_back(model, data) -> None:
    train, future = data
    p = _check(model.fit(train).predict(future), future)
    corr = np.corrcoef(p.loc["S"].to_numpy(), np.tile(PATTERN, 2))[0, 1]
    assert corr > 0.9
    assert np.allclose(p.loc["SHORT"], 1.0)  # too short -> MA28 fallback
    assert np.allclose(p.loc["Z"], 0.0, atol=1e-6)
    assert model.n_fallback_ >= 1


def test_training_window_is_respected(data) -> None:
    train, _ = data
    model = ETSModel(window=100, n_jobs=1).fit(train)
    assert max(len(y) for y in model.history_.values()) == 100


def test_grid_and_order_roundtrip() -> None:
    grid = sarima_grid({"p": [0, 1], "d": [0], "q": [0, 1], "P": [0], "D": [1], "Q": [0, 1]})
    assert len(grid) == 8
    assert str_to_order(order_to_str(grid[3])) == grid[3]


def test_best_order_picks_lowest_summed_aic() -> None:
    rng = np.random.default_rng(1)
    y = np.tile(PATTERN, 30) + rng.normal(0, 0.5, 210)
    orders = [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 1, 1))]
    table = aic_table({"a": y, "b": y + 1}, orders, n_jobs=1)
    assert table.shape == (2, 2)
    assert best_order(table) == ((1, 0, 0), (0, 1, 1))  # seasonal model fits far better


def test_holidays_frame() -> None:
    days = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=3),
            "event_name_1": [None, "X", None],
            "event_name_2": [None, None, "Y"],
        }
    )
    hol = holidays_frame(days)
    assert sorted(hol["holiday"]) == ["X", "Y"]


def test_prophet_smoke(data) -> None:
    train, future = data
    train = train[train["series_id"] == "S"]
    future = future[future["series_id"] == "S"]
    p = _check(ProphetModel(n_jobs=1).fit(train).predict(future), future)
    corr = np.corrcoef(p.loc["S"].to_numpy(), np.tile(PATTERN, 2))[0, 1]
    assert corr > 0.8
