"""Tests for ADI / CV^2 demand classification and stratified sampling."""

import numpy as np
import pandas as pd
import pytest

from forecasting.features.demand_class import (
    DEMAND_CLASSES,
    INSUFFICIENT,
    build_item_classes,
    classify_demand,
    compute_adi_cv2,
    stratified_sample,
)

CONFIG = {
    "backtest": {"folds": [[8, 9, 10]]},  # classification cutoff = d 8
    "demand_class": {"adi_threshold": 1.32, "cv2_threshold": 0.49, "min_nonzero_days": 2},
}


def _long(series: dict[str, list[int]], closed: tuple[int, ...] = ()) -> pd.DataFrame:
    rows = [
        {"item_id": item, "dept_id": "D", "cat_id": "C", "d": d, "sales": s}
        for item, sales in series.items()
        for d, s in enumerate(sales, start=1)
    ]
    df = pd.DataFrame(rows)
    df["is_closed"] = df["d"].isin(closed).astype("int8")
    df["item_id"] = df["item_id"].astype("category")
    return df


def test_adi_cv2_hand_computed() -> None:
    # 8 days up to cutoff, 4 non-zero (2, 2, 4, 4): ADI = 8/4 = 2, mean 3, std 1 -> CV^2 = 1/9
    df = _long({"A": [2, 0, 2, 0, 4, 0, 4, 0, 99, 99]})
    stats = compute_adi_cv2(df, cutoff_d=8).set_index("item_id")
    assert stats.loc["A", "n_periods"] == 8
    assert stats.loc["A", "n_nonzero"] == 4
    assert stats.loc["A", "adi"] == pytest.approx(2.0)
    assert stats.loc["A", "cv2"] == pytest.approx(1 / 9)


def test_closed_days_are_ignored() -> None:
    df = _long({"A": [1, 0, 1, 1, 1, 1, 1, 1]}, closed=(2,))
    stats = compute_adi_cv2(df, cutoff_d=8).set_index("item_id")
    assert stats.loc["A", "n_periods"] == 7
    assert stats.loc["A", "adi"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("adi", "cv2", "expected"),
    [
        (1.0, 0.1, "smooth"),
        (1.0, 0.9, "erratic"),
        (3.0, 0.1, "intermittent"),
        (3.0, 0.9, "lumpy"),
    ],
)
def test_classify_quadrants(adi: float, cv2: float, expected: str) -> None:
    stats = pd.DataFrame({"adi": [adi], "cv2": [cv2], "n_nonzero": [10]})
    assert classify_demand(stats, 1.32, 0.49, 2).iloc[0] == expected


def test_too_few_nonzero_is_insufficient() -> None:
    stats = pd.DataFrame({"adi": [8.0, np.nan], "cv2": [0.0, np.nan], "n_nonzero": [1, 0]})
    assert classify_demand(stats, 1.32, 0.49, 2).tolist() == [INSUFFICIENT, INSUFFICIENT]


def test_item_launched_after_cutoff_is_insufficient() -> None:
    df = _long({"A": [1] * 10, "B": [1] * 10})
    df = df[~((df["item_id"] == "B") & (df["d"] <= 8))]  # B launches after the cutoff
    classes = build_item_classes(df, CONFIG).set_index("item_id")
    assert classes.loc["A", "demand_class"] == "smooth"
    assert classes.loc["B", "demand_class"] == INSUFFICIENT


def test_classes_have_no_future_leakage() -> None:
    """Changing sales after the cutoff must not change any item's class or stats."""
    rng = np.random.default_rng(42)
    base = {f"I{i}": rng.poisson(rng.uniform(0.2, 3), 10).tolist() for i in range(30)}
    altered = {k: v[:8] + [500, 0] for k, v in base.items()}
    a = build_item_classes(_long(base), CONFIG)
    b = build_item_classes(_long(altered), CONFIG)
    pd.testing.assert_frame_equal(a, b)


def _classes(counts: dict[str, int]) -> pd.DataFrame:
    rows = [
        {"item_id": f"{cls}_{i:03d}", "demand_class": cls}
        for cls, n in counts.items()
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_stratified_sample_counts_and_determinism() -> None:
    classes = _classes(
        {"smooth": 80, "erratic": 60, "intermittent": 120, "lumpy": 30, INSUFFICIENT: 10}
    )
    s1 = stratified_sample(classes, n_total=200, seed=42)
    s2 = stratified_sample(classes, n_total=200, seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    counts = s1["demand_class"].value_counts().to_dict()
    assert counts == {"smooth": 50, "erratic": 50, "intermittent": 50, "lumpy": 30}
    assert INSUFFICIENT not in set(s1["demand_class"])
    assert s1["item_id"].is_unique
    assert set(s1["demand_class"]) <= set(DEMAND_CLASSES)
