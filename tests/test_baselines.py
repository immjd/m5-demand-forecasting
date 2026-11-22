"""Tests for the Naive, Seasonal Naive and Moving Average baselines."""

import pandas as pd

from forecasting.models.baselines import MODEL_REGISTRY, MovingAverage, Naive, SeasonalNaive


def _train(values: dict[str, list[float]], closed=()) -> pd.DataFrame:
    rows = [
        {"series_id": s, "d": d, "sales": v, "is_closed": int(d in closed)}
        for s, vals in values.items()
        for d, v in enumerate(vals, start=1)
    ]
    return pd.DataFrame(rows)


def _future(series: list[str], start: int, end: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"series_id": s, "d": d, "is_closed": 0} for s in series for d in range(start, end + 1)]
    )


def _pred(model, train, future) -> pd.Series:
    out = model.fit(train).predict(future)
    return out.set_index(["series_id", "d"])["y_pred"]


def test_naive_uses_last_open_day() -> None:
    p = _pred(Naive(), _train({"A": [1, 2, 5, 0]}, closed=(4,)), _future(["A"], 5, 7))
    assert p.loc["A"].tolist() == [5, 5, 5]


def test_seasonal_naive_repeats_last_week() -> None:
    train = _train({"A": list(range(1, 15))})  # value == d, train_end = 14
    p = _pred(SeasonalNaive(season=7), train, _future(["A"], 15, 28))
    assert p.loc["A"].tolist() == list(range(8, 15)) * 2


def test_seasonal_naive_skips_closed_reference_day() -> None:
    train = _train({"A": list(range(1, 15))}, closed=(10,))
    p = _pred(SeasonalNaive(season=7), train, _future(["A"], 15, 21))
    assert p.loc[("A", 17)] == 3  # reference d=10 closed -> one season earlier (d=3)


def test_moving_average_uses_last_open_days() -> None:
    train = _train({"A": [100, 1, 2, 9, 3]}, closed=(4,))
    p = _pred(MovingAverage(window=3), train, _future(["A"], 6, 7))
    assert p.loc["A"].tolist() == [2.0, 2.0]  # mean of [1, 2, 3]


def test_unseen_series_predict_zero() -> None:
    train = _train({"A": [1] * 14})
    future = _future(["A", "NEW"], 15, 16)
    for model in (Naive(), SeasonalNaive(), MovingAverage()):
        p = _pred(model, train, future)
        assert p.loc["NEW"].tolist() == [0.0, 0.0]


def test_registry_names() -> None:
    assert {"naive", "seasonal_naive", "moving_average"} <= set(MODEL_REGISTRY)
    assert all(MODEL_REGISTRY[k]().name == k for k in MODEL_REGISTRY)
