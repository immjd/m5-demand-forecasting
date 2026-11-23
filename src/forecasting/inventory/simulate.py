"""Daily lost-sales inventory simulation for many items at once (numpy arrays)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.inventory.policy import order_quantity


def simulate_lost_sales(
    demand: np.ndarray,
    order_up_to: np.ndarray,
    review: np.ndarray,
    lead_time: int,
    initial: np.ndarray,
) -> dict[str, np.ndarray]:
    """Simulate (R, S) replenishment with lost sales (unmet demand is gone).

    Day order: (1) receive orders due today, (2) on review days order
    ``max(0, S - (on hand + on order))`` arriving ``lead_time`` days later,
    (3) serve demand from stock.

    Args:
        demand: (items, days) actual demand.
        order_up_to: (items, days) S, only read on review days.
        review: (days,) bool mask of review days.
        initial: (items,) on-hand stock at the start of day 0.

    Returns:
        Arrays of shape (items, days): ``on_hand`` (end of day), ``served``, ``lost``,
        ``orders`` (placed), ``received``.
    """
    demand = np.asarray(demand, dtype=float)
    n, days = demand.shape
    on_hand = np.asarray(initial, dtype=float).copy()
    pipeline = np.zeros((n, days + lead_time + 1))
    out = {k: np.zeros((n, days)) for k in ["on_hand", "served", "lost", "orders", "received"]}
    for t in range(days):
        arriving = pipeline[:, t]
        on_hand += arriving
        out["received"][:, t] = arriving
        if review[t]:
            on_order = pipeline[:, t + 1 :].sum(axis=1)
            q = order_quantity(np.nan_to_num(order_up_to[:, t]), on_hand + on_order)
            pipeline[:, t + lead_time] += q
            out["orders"][:, t] = q
        served = np.minimum(on_hand, demand[:, t])
        on_hand -= served
        out["served"][:, t] = served
        out["lost"][:, t] = demand[:, t] - served
        out["on_hand"][:, t] = on_hand
    return out


def summarize_items(
    sim: dict[str, np.ndarray], demand: np.ndarray, prices: np.ndarray, holding_rate: float
) -> pd.DataFrame:
    """Per-item KPIs: fill rate, stockout days, average stock (units and value), holding cost."""
    demand = np.asarray(demand, dtype=float)
    days = demand.shape[1]
    total = demand.sum(axis=1)
    served = sim["served"].sum(axis=1)
    value = sim["on_hand"] * prices
    avg_value = value.mean(axis=1)
    return pd.DataFrame(
        {
            "demand": total,
            "served": served,
            "lost": sim["lost"].sum(axis=1),
            "fill_rate": np.divide(served, total, out=np.ones_like(total), where=total > 0),
            "stockout_days": (sim["lost"] > 0).sum(axis=1),
            "avg_on_hand": sim["on_hand"].mean(axis=1),
            "avg_inventory_value": avg_value,
            "holding_cost": avg_value * holding_rate * days / 365.0,
            "n_orders": (sim["orders"] > 0).sum(axis=1),
        }
    )
