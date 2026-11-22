"""Smoke tests: every plotting helper returns a Plotly figure on tiny inputs."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from forecasting.viz import plots


def test_plot_helpers_return_figures() -> None:
    dates = pd.date_range("2015-01-01", periods=60)
    daily = pd.DataFrame({"date": dates, "sales": np.arange(60)})
    wk = pd.DataFrame(
        {
            "weekday": plots.WEEKDAYS * 2,
            "cat_id": ["FOODS"] * 7 + ["HOBBIES"] * 7,
            "index": np.linspace(0.8, 1.2, 14),
        }
    )
    mo = pd.DataFrame(
        {"month": list(range(1, 13)), "cat_id": "FOODS", "index": np.linspace(0.9, 1.1, 12)}
    )
    eff = pd.DataFrame({"event": ["A", "B"], "lift": [10.0, -5.0]})
    item = daily.assign(sell_price=np.where(np.arange(60) < 30, 2.0, 1.5))
    cum = pd.Series(np.linspace(0.1, 1.0, 10))
    classes = pd.DataFrame(
        {
            "item_id": list("abcd"),
            "dept_id": "D",
            "cat_id": ["X", "X", "Y", "Y"],
            "adi": [1.0, 1.1, 2.0, 3.0],
            "cv2": [0.1, 0.9, 0.2, 0.8],
            "demand_class": ["smooth", "erratic", "intermittent", "lumpy"],
        }
    )
    stores = pd.DataFrame({"store_id": ["CA_1", "TX_1"], "mean_daily_units": [3000.0, 2500.0]})

    figs = [
        plots.plot_daily_with_trend(daily, window=7),
        plots.plot_weekday_profile(wk),
        plots.plot_month_profile(mo),
        plots.plot_effect_bars(eff, label="event", value="lift"),
        plots.plot_price_vs_sales(item),
        plots.plot_pareto(cum),
        plots.plot_adi_cv2(classes, 1.32, 0.49),
        plots.plot_class_share(classes),
        plots.plot_store_comparison(stores, "mean_daily_units", highlight="CA_1"),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
