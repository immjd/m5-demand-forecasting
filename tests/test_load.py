"""Tests for raw CSV loading using tiny synthetic M5-like files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.data.load import load_sales_long, store_state

N_DAYS = 14


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True)
    dates = pd.date_range("2011-01-29", periods=N_DAYS)
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11101] * 7 + [11102] * 7,
            "weekday": dates.day_name(),
            "wday": [(i % 7) + 1 for i in range(N_DAYS)],
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, N_DAYS + 1)],
            "event_name_1": [None, None, "SuperBowl"] + [None] * (N_DAYS - 3),
            "event_type_1": [None, None, "Sporting"] + [None] * (N_DAYS - 3),
            "event_name_2": [None] * N_DAYS,
            "event_type_2": [None] * N_DAYS,
            "snap_CA": [1, 0] * 7,
            "snap_TX": [0, 0, 1] * 4 + [0, 0],
            "snap_WI": [0] * N_DAYS,
        }
    ).to_csv(raw / "calendar.csv", index=False)

    rows = []
    for store, item, base in [
        ("CA_1", "FOODS_1_001", 1),
        ("CA_1", "HOBBIES_1_001", 10),
        ("TX_1", "FOODS_1_001", 100),
    ]:
        cat = item.split("_")[0]
        dept = "_".join(item.split("_")[:2])
        row = {
            "id": f"{item}_{store}_evaluation",
            "item_id": item,
            "dept_id": dept,
            "cat_id": cat,
            "store_id": store,
            "state_id": store_state(store),
        }
        row.update({f"d_{i}": base + i for i in range(1, N_DAYS + 1)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(raw / "sales_train_evaluation.csv", index=False)

    pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": [
                "FOODS_1_001",
                "FOODS_1_001",
                "HOBBIES_1_001",
                "FOODS_1_001",
                "FOODS_1_001",
            ],
            "wm_yr_wk": [11101, 11102, 11102, 11101, 11102],
            "sell_price": [1.0, 1.5, 9.99, 50.0, 50.0],
        }
    ).to_csv(raw / "sell_prices.csv", index=False)


@pytest.fixture
def config(tmp_path: Path) -> dict:
    _write_raw(tmp_path / "raw")
    return {
        "paths": {"raw": str(tmp_path / "raw")},
        "scope": {"store_id": "CA_1", "history_start_d": 1},
    }


def test_store_state() -> None:
    assert store_state("CA_1") == "CA"
    assert store_state("WI_3") == "WI"


def test_filters_store_and_melts(config: dict) -> None:
    df = load_sales_long(config)
    assert set(df["store_id"]) == {"CA_1"}
    assert len(df) == 2 * N_DAYS
    foods = df[df["item_id"] == "FOODS_1_001"]
    np.testing.assert_array_equal(foods["d"], np.arange(1, N_DAYS + 1))
    np.testing.assert_array_equal(foods["sales"], 1 + np.arange(1, N_DAYS + 1))


def test_snap_uses_store_state(config: dict) -> None:
    df = load_sales_long(config)
    foods = df[df["item_id"] == "FOODS_1_001"]
    np.testing.assert_array_equal(foods["snap"], [1, 0] * 7)


def test_weekly_price_join(config: dict) -> None:
    df = load_sales_long(config)
    hobbies = df[df["item_id"] == "HOBBIES_1_001"].set_index("d")["sell_price"]
    assert hobbies.loc[1:7].isna().all()
    assert np.allclose(hobbies.loc[8:14], 9.99)
    foods = df[df["item_id"] == "FOODS_1_001"].set_index("d")["sell_price"]
    assert np.allclose(foods.loc[1:7], 1.0)
    assert np.allclose(foods.loc[8:14], 1.5)


def test_dtypes_are_downcast(config: dict) -> None:
    df = load_sales_long(config)
    assert df["sales"].dtype == "int16"
    assert df["d"].dtype == "int16"
    assert df["sell_price"].dtype == "float32"
    assert isinstance(df["item_id"].dtype, pd.CategoricalDtype)
    assert isinstance(df["event_name_1"].dtype, pd.CategoricalDtype)


def test_history_start_d(config: dict) -> None:
    config["scope"]["history_start_d"] = 5
    df = load_sales_long(config)
    assert df["d"].min() == 5
    assert len(df) == 2 * (N_DAYS - 4)


def test_unknown_store_raises(config: dict) -> None:
    config["scope"]["store_id"] = "XX_9"
    with pytest.raises(ValueError, match="not found"):
        load_sales_long(config)
