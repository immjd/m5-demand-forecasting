"""Build the 11 aggregate series: store total, 3 categories, 7 departments."""

from __future__ import annotations

import pandas as pd

LEVELS: list[tuple[str, str]] = [
    ("store", "store_id"),
    ("cat", "cat_id"),
    ("dept", "dept_id"),
]


def build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Sum item sales into store / category / department daily series.

    Every series covers every day in ``df`` (missing days are filled with 0),
    so series stay contiguous even if all items of a group launched later.

    Returns:
        Columns ``[level, series_id, d, date, sales, is_closed]``.
    """
    days = df[["d", "date", "is_closed"]].drop_duplicates("d").sort_values("d")
    frames = []
    for level, key in LEVELS:
        wide = df.pivot_table(
            index="d",
            columns=key,
            values="sales",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        ).reindex(days["d"], fill_value=0)
        long = wide.stack().rename("sales").reset_index().rename(columns={key: "series_id"})
        long["level"] = level
        frames.append(long)

    agg = pd.concat(frames, ignore_index=True)
    agg["series_id"] = agg["series_id"].astype(str)
    agg = agg.merge(days, on="d", how="left")
    agg["sales"] = agg["sales"].astype("int32")
    agg["level"] = pd.Categorical(agg["level"], categories=[lvl for lvl, _ in LEVELS])
    agg["series_id"] = agg["series_id"].astype("category")
    agg = agg.sort_values(["level", "series_id", "d"]).reset_index(drop=True)
    return agg[["level", "series_id", "d", "date", "sales", "is_closed"]]


def bottom_up(item_preds: pd.DataFrame, item_meta: pd.DataFrame) -> pd.DataFrame:
    """Sum item-level forecasts into the 11 aggregate series (bottom-up reconciliation).

    Args:
        item_preds: ``fold, series_id (item), d, y_pred``.
        item_meta: One row per item with ``series_id, store_id, cat_id, dept_id``.

    Returns:
        ``fold, series_id (aggregate), d, y_pred``.
    """
    meta = item_meta.assign(series_id=item_meta["series_id"].astype(str))
    p = item_preds.assign(series_id=item_preds["series_id"].astype(str)).merge(
        meta, on="series_id", how="left", validate="many_to_one"
    )
    parts = []
    for _, key in LEVELS:
        s = p.groupby(["fold", key, "d"], observed=True)["y_pred"].sum().reset_index()
        parts.append(s.rename(columns={key: "series_id"}))
    out = pd.concat(parts, ignore_index=True)
    out["series_id"] = out["series_id"].astype(str)
    return out
