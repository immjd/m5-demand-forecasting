"""Tests for closed-day flags, launch trimming and the zero_run_len column."""

import numpy as np
import pandas as pd

from forecasting.data.preprocess import add_zero_run_len, flag_closed_days, trim_leading_zeros


def _frame(
    series: dict[str, list[int]],
    prices: dict[str, list[float]] | None = None,
    closed: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal long table; ``closed`` lists d values with is_closed = 1."""
    rows = []
    for item, sales in series.items():
        item_prices = (prices or {}).get(item, [1.0] * len(sales))
        for d, (s, p) in enumerate(zip(sales, item_prices, strict=True), start=1):
            rows.append({"item_id": item, "d": d, "sales": s, "sell_price": p})
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype("category")
    df["d"] = df["d"].astype("int16")
    df["sales"] = df["sales"].astype("int16")
    df["sell_price"] = df["sell_price"].astype("float32")
    if closed is not None:
        df["is_closed"] = df["d"].isin(closed).astype("int8")
    return df


# ---------------------------------------------------------------- closed days
def test_closed_day_requires_all_items_zero() -> None:
    df = _frame({"A": [3, 0, 0, 2], "B": [1, 0, 4, 2]})
    out = flag_closed_days(df)
    closed = out.groupby("d")["is_closed"].first()
    assert closed.tolist() == [0, 1, 0, 0]


# ---------------------------------------------------------------- launch trim
def test_trim_starts_at_first_price() -> None:
    nan = float("nan")
    df = _frame(
        {"A": [0, 0, 0, 0, 2, 1], "B": [5, 4, 3, 2, 1, 0]},
        prices={"A": [nan, nan, 2.0, 2.0, 2.0, 2.0], "B": [1.0] * 6},
    )
    out = trim_leading_zeros(df)
    # A keeps its true zeros after launch (d=3, d=4) instead of starting at first sale.
    assert out.loc[out["item_id"] == "A", "d"].tolist() == [3, 4, 5, 6]
    assert out.loc[out["item_id"] == "B", "d"].tolist() == [1, 2, 3, 4, 5, 6]


def test_trim_drops_items_without_price() -> None:
    nan = float("nan")
    df = _frame({"A": [1, 2], "B": [0, 0]}, prices={"A": [1.0, 1.0], "B": [nan, nan]})
    out = trim_leading_zeros(df)
    assert set(out["item_id"]) == {"A"}


# ---------------------------------------------------------------- zero runs
def test_zero_run_len_basic() -> None:
    df = _frame({"A": [5, 0, 0, 3, 0]}, closed=[])
    out = add_zero_run_len(df)
    assert out["zero_run_len"].tolist() == [0, 1, 2, 0, 1]


def test_zero_run_len_starts_with_zeros() -> None:
    df = _frame({"A": [0, 0, 1]}, closed=[])
    assert add_zero_run_len(df)["zero_run_len"].tolist() == [1, 2, 0]


def test_zero_run_len_is_per_item() -> None:
    df = _frame({"A": [0, 0, 0], "B": [1, 0, 0]}, closed=[])
    out = add_zero_run_len(df).set_index(["item_id", "d"])["zero_run_len"]
    assert out.loc["A"].tolist() == [1, 2, 3]
    assert out.loc["B"].tolist() == [0, 1, 2]


def test_zero_run_len_skips_closed_days() -> None:
    # d=3 is closed: it carries the previous value and does not extend the run.
    df = _frame({"A": [0, 0, 0, 0, 7]}, closed=[3])
    out = add_zero_run_len(df)
    assert out["zero_run_len"].tolist() == [1, 2, 2, 3, 0]


def test_zero_run_len_has_no_future_leakage() -> None:
    """Changing sales after day t must not change zero_run_len on days <= t."""
    rng = np.random.default_rng(42)
    n_days, cutoff = 60, 40
    base = {item: rng.poisson(0.7, n_days).tolist() for item in ["A", "B", "C"]}
    altered = {
        item: s[:cutoff] + rng.poisson(5, n_days - cutoff).tolist() for item, s in base.items()
    }

    out_base = add_zero_run_len(_frame(base, closed=[10, 45]))
    out_alt = add_zero_run_len(_frame(altered, closed=[10, 45]))

    past_base = out_base.loc[out_base["d"] <= cutoff, "zero_run_len"].to_numpy()
    past_alt = out_alt.loc[out_alt["d"] <= cutoff, "zero_run_len"].to_numpy()
    np.testing.assert_array_equal(past_base, past_alt)
