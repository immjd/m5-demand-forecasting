"""Tests for the store / category / department aggregate series."""

import pandas as pd

from forecasting.data.aggregate import build_aggregates

ITEMS = [  # item_id, dept_id, cat_id, first d on sale
    ("FOODS_1_001", "FOODS_1", "FOODS", 1),
    ("FOODS_2_001", "FOODS_2", "FOODS", 1),
    ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", 4),  # launches later
]
N_DAYS = 6


def _long() -> pd.DataFrame:
    rows = []
    for item, dept, cat, start in ITEMS:
        for d in range(start, N_DAYS + 1):
            rows.append(
                {
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": "CA_1",
                    "d": d,
                    "sales": d * 10 + start,
                    "is_closed": int(d == 2),
                }
            )
    df = pd.DataFrame(rows)
    for col in ["item_id", "dept_id", "cat_id", "store_id"]:
        df[col] = df[col].astype("category")
    df["date"] = pd.Timestamp("2011-01-28") + pd.to_timedelta(df["d"], unit="D")
    return df


def test_series_count_and_columns() -> None:
    agg = build_aggregates(_long())
    assert list(agg.columns) == ["level", "series_id", "d", "date", "sales", "is_closed"]
    counts = agg.groupby("level", observed=True)["series_id"].nunique()
    assert counts.to_dict() == {"store": 1, "cat": 2, "dept": 3}


def test_levels_sum_to_store_total() -> None:
    agg = build_aggregates(_long())
    totals = agg.groupby(["level", "d"], observed=True)["sales"].sum().unstack("level")
    pd.testing.assert_series_equal(totals["cat"], totals["store"], check_names=False)
    pd.testing.assert_series_equal(totals["dept"], totals["store"], check_names=False)


def test_late_launch_series_is_zero_filled() -> None:
    agg = build_aggregates(_long())
    hobbies = agg[agg["series_id"] == "HOBBIES_1"].set_index("d")["sales"]
    assert len(hobbies) == N_DAYS
    assert hobbies.loc[1:3].eq(0).all()
    assert hobbies.loc[4:].gt(0).all()


def test_is_closed_carried_to_aggregates() -> None:
    agg = build_aggregates(_long())
    store = agg[agg["level"] == "store"].set_index("d")["is_closed"]
    assert store.loc[2] == 1
    assert store.drop(2).eq(0).all()
