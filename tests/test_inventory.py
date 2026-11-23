"""Inventory policy, interval calibration and lost-sales simulation (written before the code)."""

import numpy as np
import pandas as pd
import pytest

from forecasting.inventory.intervals import (
    calibrate_quantiles,
    calibration_folds,
    cumulative_windows,
    interval_coverage,
    normalized_errors,
)
from forecasting.inventory.policy import extend_forecast, order_quantity, order_up_to_levels
from forecasting.inventory.simulate import simulate_lost_sales, summarize_items

L = 3


def _sim(demand, S, review, initial=None, lead=L):
    demand = np.atleast_2d(np.asarray(demand, float))
    S = np.atleast_2d(np.asarray(S, float))
    review = np.asarray(review, bool)
    initial = S[:, 0] if initial is None else np.asarray(initial, float)
    return simulate_lost_sales(demand, S, review, lead_time=lead, initial=initial)


# ------------------------------------------------------------------ simulation
def test_stock_never_negative_and_flows_balance() -> None:
    rng = np.random.default_rng(0)
    T = 56
    demand = rng.poisson(3, (20, T))
    review = np.arange(T) % 7 == 0
    S = np.full((20, T), 12.0)
    out = _sim(demand, S, review)
    assert (out["on_hand"] >= 0).all()
    assert (out["served"] <= demand).all()
    np.testing.assert_allclose(out["served"] + out["lost"], demand)
    # stock balance: initial + received - served = final on hand
    np.testing.assert_allclose(
        S[:, 0] + out["received"].sum(1) - out["served"].sum(1), out["on_hand"][:, -1]
    )


def test_zero_demand_keeps_inventory_constant() -> None:
    T = 21
    review = np.arange(T) % 7 == 0
    out = _sim(np.zeros((1, T)), np.full((1, T), 5.0), review)
    assert (out["orders"] == 0).all()
    np.testing.assert_allclose(out["on_hand"], 5.0)


def test_fill_rate_is_one_when_stock_always_sufficient() -> None:
    T = 28
    demand = np.full((3, T), 2.0)
    out = _sim(demand, np.full((3, T), 1000.0), np.arange(T) % 7 == 0)
    items = summarize_items(out, demand, prices=np.ones((3, T)), holding_rate=0.25)
    np.testing.assert_allclose(items["fill_rate"], 1.0)
    assert (items["stockout_days"] == 0).all()


def test_order_arrives_exactly_after_lead_time() -> None:
    T = 10
    demand = np.zeros((1, T))
    demand[0, 0] = 4.0  # consumes stock on day 0 after the review
    S = np.full((1, T), 4.0)
    review = np.zeros(T, bool)
    review[[0, 1]] = True
    out = _sim(demand, S, review)
    # day 0: no order (IP = S); day 1: IP = 0 -> order 4, arrives at start of day 1 + L
    assert out["orders"][0, 1] == 4.0
    assert out["on_hand"][0, 1 + L - 1] == 0.0
    assert out["received"][0, 1 + L] == 4.0
    assert out["on_hand"][0, 1 + L] == 4.0


def test_order_quantity_never_negative() -> None:
    q = order_quantity(np.array([5.0, 10.0, 0.0]), np.array([8.0, 2.0, 1.0]))
    np.testing.assert_allclose(q, [0.0, 8.0, 0.0])


def test_lost_sales_not_backordered() -> None:
    T = 7
    demand = np.full((1, T), 3.0)
    out = _sim(demand, np.full((1, T), 3.0), np.r_[True, np.zeros(T - 1, bool)])
    assert out["served"][0, 0] == 3.0
    assert out["lost"][0, 1:].sum() == 18.0  # no replenishment, nothing carried over


# ------------------------------------------------------------------ policy
def test_extend_forecast_repeats_same_weekday() -> None:
    f = np.arange(14, dtype=float).reshape(1, 14)
    ext = extend_forecast(f, extra=3, season=7)
    assert ext.shape == (1, 17)
    np.testing.assert_allclose(ext[0, 14:], f[0, 7:10])


def test_order_up_to_levels_on_review_days() -> None:
    f = np.ones((2, 28))
    ss = np.array([2.0, -100.0])
    S, review = order_up_to_levels(f, ss, review_period=7, lead_time=3)
    assert review.sum() == 4 and review[[0, 7, 14, 21]].all()
    np.testing.assert_allclose(S[0, review], 10.0 + 2.0)
    assert (S[1, review] == 0.0).all()  # never below zero
    assert np.isnan(S[0, ~review]).all()


# ------------------------------------------------------------------ intervals
def test_calibration_uses_only_previous_folds() -> None:
    names = ["F1", "F2", "F3", "F4", "TEST"]
    assert calibration_folds("F2", names) == ["F1"]
    assert calibration_folds("TEST", names) == ["F1", "F2", "F3", "F4"]
    with pytest.raises(ValueError):
        calibration_folds("F1", names)


def test_cumulative_windows_sum_consecutive_days() -> None:
    preds = pd.DataFrame(
        {
            "fold": "F1",
            "series_id": "A",
            "d": np.arange(1, 6),
            "is_closed": 0,
            "y_true": [1, 2, 3, 4, 5],
            "y_pred": [1.0, 1, 1, 1, 1],
        }
    )
    w = cumulative_windows(preds, window=3)
    assert list(w["start_d"]) == [1, 2, 3]
    assert list(w["y_cum"]) == [6, 9, 12]
    assert list(w["f_cum"]) == [3.0, 3.0, 3.0]


def test_quantiles_per_class_and_coverage() -> None:
    windows = pd.DataFrame(
        {
            "fold": "F1",
            "series_id": ["a"] * 100 + ["b"] * 100,
            "start_d": list(range(100)) * 2,
            "y_cum": np.r_[np.arange(100), np.zeros(100)],
            "f_cum": 0.0,
        }
    )
    scales = pd.Series({"a": 1.0, "b": 2.0})
    classes = pd.Series({"a": "smooth", "b": "lumpy"})
    err = normalized_errors(windows, scales, classes)
    q = calibrate_quantiles(err, [0.9])
    assert q.loc["smooth", 0.9] == pytest.approx(89.1)
    assert q.loc["lumpy", 0.9] == pytest.approx(0.0)
    cov = interval_coverage(windows, q, scales, classes, [0.9])
    assert cov.set_index("demand_class").loc["smooth", "coverage"] == pytest.approx(0.9)
    # scaling: item "b" errors are divided by its scale, so units do not mix across items
    assert err.loc[err["series_id"] == "b", "e_norm"].eq(0).all()


def test_inventory_at_fill_rate_interpolates() -> None:
    from forecasting.inventory.run import inventory_at_fill_rate

    s = pd.DataFrame(
        {
            "model": "m",
            "fill_rate": [0.90, 0.96, 0.99],
            "avg_inventory_value": [100.0, 160.0, 400.0],
        }
    )
    out = inventory_at_fill_rate(s, [0.93, 0.995]).set_index("fill_rate")["avg_inventory_value"]
    assert out[0.93] == pytest.approx(130.0)
    assert np.isnan(out[0.995])


def test_order_up_to_levels_are_whole_units() -> None:
    f = np.full((2, 28), 0.13)  # 10-day sum = 1.3
    S, review = order_up_to_levels(f, np.array([0.0, 0.7]), review_period=7, lead_time=3)
    np.testing.assert_array_equal(S[0, review], 2.0)  # 1.3 -> 2
    np.testing.assert_array_equal(S[1, review], 2.0)  # exactly 2.0 stays 2


def test_integer_demand_and_levels_give_integer_flows() -> None:
    rng = np.random.default_rng(1)
    demand = rng.poisson(0.6, (30, 28))
    S, review = order_up_to_levels(
        rng.uniform(0, 1.5, (30, 28)), rng.uniform(0, 3, 30), review_period=7, lead_time=L
    )
    out = _sim(demand, S, review)
    for key in ["on_hand", "served", "lost", "orders", "received"]:
        np.testing.assert_array_equal(out[key], np.round(out[key]))
