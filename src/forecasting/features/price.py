"""Price features. Sell prices are scheduled in advance, so the current price is allowed.

Fold-independent features are built once; the per-item price level
(``relative_price``) depends on the training period and is computed per fold.
"""

from __future__ import annotations

import pandas as pd

PRICE_FEATURES = ["sell_price", "price_change_1w", "price_momentum", "price_rank_dept"]
FOLD_PRICE_FEATURES = ["price_rel_item_mean"]


def add_price_features(df: pd.DataFrame, momentum_days: int = 28) -> pd.DataFrame:
    """Add price change vs. one week ago, momentum vs. the trailing mean, and the
    percentile rank of the price within the department on the same day.
    """
    out = df.copy()
    price = out["sell_price"].astype("float32")
    g = price.groupby(out["series_id"], observed=True)
    out["price_change_1w"] = (price / g.shift(7) - 1).astype("float32")
    trailing = g.rolling(momentum_days, min_periods=1).mean().reset_index(level=0, drop=True)
    out["price_momentum"] = (price / trailing - 1).astype("float32")
    out["price_rank_dept"] = (
        out.groupby(["dept_id", "d"], observed=True)["sell_price"].rank(pct=True).astype("float32")
    )
    return out


def item_mean_price(train: pd.DataFrame, cutoff_d: int) -> pd.Series:
    """Mean sell price per series using only days ``<= cutoff_d``."""
    hist = train.loc[train["d"] <= cutoff_d]
    return hist.groupby(hist["series_id"].astype(str))["sell_price"].mean()


def relative_price(df: pd.DataFrame, mean_price: pd.Series) -> pd.Series:
    """Price divided by the item's training-period mean price (NaN for unseen items)."""
    level = df["series_id"].astype(str).map(mean_price).astype("float64")
    return (df["sell_price"] / level).astype("float32")
