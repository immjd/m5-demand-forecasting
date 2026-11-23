"""The public app may only ship aggregate tables (M5 rules 7B, docs D-033)."""

import numpy as np
import pandas as pd
import pytest

from forecasting.inventory.intervals import daily_intervals
from scripts.export_app_artifacts import (
    ARTIFACTS_DIR,
    ITEM_NAME,
    assert_aggregate_only,
    with_policy,
)

ALLOWED = {
    "CA_1", "FOODS", "HOBBIES", "HOUSEHOLD", "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2",
}  # fmt: skip
ARTIFACTS = sorted(ARTIFACTS_DIR.glob("*.parquet"))


def test_guard_accepts_aggregates() -> None:
    assert_aggregate_only(pd.DataFrame({"series_id": ["CA_1", "FOODS_1"], "index": 1.0}), ALLOWED)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"item_id": ["x"], "v": [1]}),
        pd.DataFrame({"sales": [3]}),
        pd.DataFrame({"label": ["FOODS_1_001"]}),
        pd.DataFrame({"label": pd.Categorical(["FOODS_1_001_CA_1"])}),
        pd.DataFrame({"series_id": ["TX_1"]}),
    ],
)
def test_guard_rejects_item_level_tables(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        assert_aggregate_only(df, ALLOWED)


@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: p.stem)
def test_exported_artifacts_are_aggregate_only(path) -> None:
    assert_aggregate_only(pd.read_parquet(path), ALLOWED)


def test_exported_artifacts_are_small_and_meta_is_clean() -> None:
    if not ARTIFACTS:
        pytest.skip("artifacts not exported yet")
    assert sum(p.stat().st_size for p in ARTIFACTS) < 50 * 1024**2
    assert not ITEM_NAME.search((ARTIFACTS_DIR / "meta.json").read_text(encoding="utf-8"))


def test_with_policy_does_not_touch_the_original() -> None:
    cfg = {"inventory": {"review_period_days": 7, "lead_time_days": 3}}
    out = with_policy(cfg, 14, 1)
    assert out["inventory"] == {"review_period_days": 14, "lead_time_days": 1}
    assert cfg["inventory"]["review_period_days"] == 7


def test_daily_intervals_use_calibration_residuals_only() -> None:
    cal = pd.DataFrame(
        {"fold": "F1", "series_id": "A", "d": range(101), "is_closed": 0,
         "y_true": np.arange(101.0), "y_pred": 50.0}
    )  # fmt: skip
    target = pd.DataFrame(
        {"series_id": ["A", "A"], "d": [200, 201], "is_closed": 0,
         "y_true": [1e6, -1e6], "y_pred": [100.0, 10.0]}
    )  # fmt: skip
    out = daily_intervals(cal, target, [0.8])
    # residuals are -50..50, so the 10% / 90% quantiles are -40 / +40
    np.testing.assert_allclose(out["hi80"], [140.0, 50.0])
    np.testing.assert_allclose(out["lo80"], [60.0, 0.0])  # never below zero
    changed = daily_intervals(cal, target.assign(y_true=0.0), [0.8])
    np.testing.assert_allclose(out["hi80"], changed["hi80"])  # actuals are not used


def test_daily_intervals_ignore_closed_days_and_need_every_series() -> None:
    cal = pd.DataFrame(
        {"series_id": "A", "is_closed": [0, 0, 1], "y_true": [1.0, 1.0, 999.0], "y_pred": 1.0}
    )
    target = pd.DataFrame({"series_id": ["A"], "y_pred": [5.0], "d": [1], "is_closed": 0})
    out = daily_intervals(cal, target, [0.8])
    assert out["hi80"].iloc[0] == 5.0
    with pytest.raises(ValueError, match="no calibration"):
        daily_intervals(cal, target.assign(series_id="B"), [0.8])
