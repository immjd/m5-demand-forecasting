"""Tests for evaluation metrics, using small hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.metrics import (
    bias,
    compute_scales,
    coverage,
    evaluate_fold,
    pinball,
    revenue_weights,
    rmsse_per_series,
    wape,
    wrmsse_lite,
)


def _train(values: dict[str, list[float]], price: float = 1.0, closed=()) -> pd.DataFrame:
    rows = [
        {"series_id": s, "d": d, "sales": v, "sell_price": price, "is_closed": int(d in closed)}
        for s, vals in values.items()
        for d, v in enumerate(vals, start=1)
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ simple metrics


def test_wape_and_bias() -> None:
    y = np.array([10.0, 0.0, 10.0])
    yhat = np.array([8.0, 1.0, 13.0])
    assert wape(y, yhat) == pytest.approx((2 + 1 + 3) / 20)
    assert bias(y, yhat) == pytest.approx((-2 + 1 + 3) / 20)  # positive = over-forecast


def test_pinball_and_coverage() -> None:
    y = np.array([10.0, 10.0])
    assert pinball(y, np.array([8.0, 12.0]), q=0.9) == pytest.approx((0.9 * 2 + 0.1 * 2) / 2)
    assert coverage(y, np.array([9.0, 11.0]), np.array([11.0, 12.0])) == pytest.approx(0.5)


# ------------------------------------------------------------------ scales & weights


def test_scales_hand_computed() -> None:
    # diffs of [1, 3, 2, 4] = [2, -1, 2] -> MSE scale 3, MAE scale 5/3
    scales = compute_scales(_train({"A": [1, 3, 2, 4]}), cutoff_d=4)
    assert scales.loc["A", "mse"] == pytest.approx(3.0)
    assert scales.loc["A", "mae"] == pytest.approx(5 / 3)


def test_scales_skip_closed_days() -> None:
    # day 3 closed -> open series [1, 3, 4], diffs [2, 1]
    scales = compute_scales(_train({"A": [1, 3, 0, 4]}, closed=(3,)), cutoff_d=4)
    assert scales.loc["A", "mse"] == pytest.approx((4 + 1) / 2)


def test_scales_and_weights_have_no_future_leakage() -> None:
    base = _train({"A": [1, 3, 2, 4], "B": [5, 5, 6, 7]})
    future = _train({"A": [100, 0], "B": [0, 900]}, price=50.0).assign(d=lambda x: x["d"] + 4)
    longer = pd.concat([base, future], ignore_index=True)
    pd.testing.assert_frame_equal(compute_scales(base, 4), compute_scales(longer, 4))
    pd.testing.assert_series_equal(revenue_weights(base, 4, 2), revenue_weights(longer, 4, 2))


def test_revenue_weights_use_last_window() -> None:
    train = _train({"A": [9, 9, 1, 1], "B": [0, 0, 3, 3]})
    w = revenue_weights(train, cutoff_d=4, window=2)  # A: 2, B: 6
    assert w["A"] == pytest.approx(0.25)
    assert w["B"] == pytest.approx(0.75)


# ------------------------------------------------------------------ RMSSE family


def test_rmsse_hand_computed() -> None:
    scales = pd.DataFrame({"mse": [3.0], "mae": [1.0]}, index=pd.Index(["A"], name="series_id"))
    preds = pd.DataFrame({"series_id": "A", "y_true": [1, 2, 3], "y_pred": [1, 1, 1]})
    r = rmsse_per_series(preds, scales)
    assert r["A"] == pytest.approx(np.sqrt((5 / 3) / 3))


def test_wrmsse_renormalises_over_scored_series() -> None:
    r = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0})
    w = pd.Series({"A": 0.25, "B": 0.5, "C": 0.25})
    assert wrmsse_lite(r, w) == pytest.approx(2.0)


def test_evaluate_fold_excludes_closed_days() -> None:
    train = _train({"A": [1, 3, 2, 4]})
    preds = pd.DataFrame(
        {
            "series_id": "A",
            "d": [5, 6],
            "y_true": [4, 0],
            "y_pred": [4, 50],  # huge error on a closed day must be ignored
            "is_closed": [0, 1],
        }
    )
    m = evaluate_fold(preds, train, cutoff_d=4, weight_window=2)
    assert m["wape"] == pytest.approx(0.0)
    assert m["rmsse"] == pytest.approx(0.0)
    assert m["n_series"] == 1


def test_breakdowns_by_series_and_horizon() -> None:
    from types import SimpleNamespace

    from forecasting.evaluation.metrics import metrics_by_horizon, per_series_rmsse

    data = pd.DataFrame(
        {
            "series_id": "A",
            "d": range(1, 11),
            "is_closed": 0,
            "sales": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    fold = SimpleNamespace(name="F1", train_end=8)
    preds = pd.DataFrame(
        {
            "fold": "F1",
            "series_id": "A",
            "d": [9, 10],
            "is_closed": 0,
            "y_true": [1, 2],
            "y_pred": [1.0, 4.0],
        }
    )
    r = per_series_rmsse(preds, data, [fold])
    # scale = mean squared 1-step diff on d<=8 = 1; forecast MSE = (0 + 4) / 2 = 2
    assert r.loc[0, "rmsse"] == pytest.approx(np.sqrt(2.0))
    h = metrics_by_horizon(preds, [fold]).set_index("h")
    assert h.loc[1, "wape"] == 0.0
    assert h.loc[2, "bias"] == pytest.approx(1.0)


def test_metrics_by_group_skips_closed_days() -> None:
    from forecasting.evaluation.metrics import metrics_by_group

    preds = pd.DataFrame(
        {
            "grp": ["a", "a", "b", "b"],
            "is_closed": [0, 0, 0, 1],
            "y_true": [2, 2, 4, 0],
            "y_pred": [3.0, 1.0, 2.0, 9.0],
        }
    )
    out = metrics_by_group(preds, "grp").set_index("grp")
    assert out.loc["a", "wape"] == pytest.approx(0.5)
    assert out.loc["a", "bias"] == pytest.approx(0.0)
    assert out.loc["b", "bias"] == pytest.approx(-0.5)  # closed row ignored
    assert out.loc["b", "n_rows"] == 1
