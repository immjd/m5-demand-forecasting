"""Cleaning steps for the long sales table: closed days, launch trim, zero runs."""

from __future__ import annotations

import pandas as pd


def flag_closed_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_closed`` = 1 on days where total store sales are zero (e.g. Christmas).

    Must be called on the untrimmed table so every item contributes to the total.
    """
    daily_total = df.groupby("d")["sales"].sum()
    closed_d = daily_total.index[daily_total == 0]
    out = df.copy()
    out["is_closed"] = out["d"].isin(closed_d).astype("int8")
    return out


def trim_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before each item's first available sell price.

    The first price date marks the launch of the item in the store. It is known
    in advance (scheduled price), so trimming on it does not leak future sales,
    unlike trimming on the first non-zero sale. Items that never have a price
    are dropped entirely.
    """
    first_price_d = (
        df["d"]
        .where(df["sell_price"].notna())
        .groupby(df["item_id"], observed=True)
        .transform("min")
    )
    keep = df["d"] >= first_price_d
    return df.loc[keep].reset_index(drop=True)


def add_zero_run_len(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``zero_run_len``: consecutive zero-sales days up to and including day t.

    Backward-looking only (no future data). Closed days neither extend nor reset
    the run; they carry the value of the previous open day. Requires
    ``is_closed``. This is a raw column: shift it by ``min_lag`` before using it
    as a model feature.
    """
    out = df.sort_values(["item_id", "d"]).reset_index(drop=True)
    open_rows = out.loc[out["is_closed"] == 0, ["item_id", "sales"]]

    is_zero = open_rows["sales"].eq(0)
    run_id = (~is_zero).groupby(open_rows["item_id"], observed=True).cumsum()
    run_len = (
        is_zero.astype("int32").groupby([open_rows["item_id"], run_id], observed=True).cumsum()
    )

    out["zero_run_len"] = run_len
    out["zero_run_len"] = (
        out.groupby("item_id", observed=True)["zero_run_len"].ffill().fillna(0).astype("int16")
    )
    return out
