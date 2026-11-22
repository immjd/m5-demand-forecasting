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


def test_model_plot_helpers_return_figures() -> None:
    dates = pd.date_range("2015-01-01", periods=120)
    y = pd.Series(np.tile([5.0, 3, 3, 3, 4, 6, 8], 18)[:120], index=dates)
    folds = pd.DataFrame(
        {
            "fold": ["F1", "TEST"],
            "train_start": [dates[0], dates[0]],
            "train_end": [dates[60], dates[90]],
            "valid_start": [dates[61], dates[91]],
            "valid_end": [dates[74], dates[104]],
        }
    )
    history = pd.DataFrame({"date": dates[:90], "sales": y.iloc[:90].to_numpy()})
    actual = pd.DataFrame({"date": dates[90:], "y_true": y.iloc[90:].to_numpy()})
    preds = pd.concat(
        [actual.rename(columns={"y_true": "y_pred"}).assign(model=m) for m in ["ets", "prophet"]]
    )
    board = pd.DataFrame(
        {"model": ["ets", "moving_average"], "rmsse_mean": [0.8, 0.9], "rmsse_std": [0.1, 0.05]}
    )
    by_class = pd.DataFrame(
        {
            "demand_class": ["smooth", "lumpy"] * 2,
            "model": ["ets"] * 2 + ["prophet"] * 2,
            "rmsse": [0.7, 0.9, 0.8, 1.1],
        }
    )
    figs = [
        plots.plot_backtest_folds(folds),
        plots.plot_stl(y, period=7),
        plots.plot_acf_pacf(y, lags=21),
        plots.plot_forecast_vs_actual(history, actual, preds),
        plots.plot_leaderboard(board, "rmsse", benchmark="moving_average"),
        plots.plot_metric_by_group(by_class, "demand_class", "rmsse"),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_lgbm_plot_helpers_return_figures() -> None:
    rng = np.random.default_rng(0)
    trials = pd.DataFrame(
        {
            "number": range(5),
            "value": [0.9, 0.85, np.nan, 0.84, 0.86],
            "state": ["COMPLETE", "COMPLETE", "PRUNED", "COMPLETE", "COMPLETE"],
        }
    )
    imp = pd.DataFrame(rng.uniform(1, 10, (4, 2)), index=list("abcd"), columns=["F1", "F2"])
    X = pd.DataFrame({"a": rng.normal(size=50), "b": pd.Categorical(rng.choice(["x", "y"], 50))})
    vals = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    hz = pd.DataFrame(
        {
            "h": [1, 2, 1, 2],
            "model": ["ets", "ets", "lgbm_tuned", "lgbm_tuned"],
            "wape": [0.7, 0.72, 0.65, 0.66],
        }
    )
    figs = [
        plots.plot_optuna_history(trials, baseline=0.87),
        plots.plot_importance(imp, top_n=3),
        plots.plot_shap_summary(vals, X),
        plots.plot_metric_by_horizon(hz, "wape"),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_plot_heatmap_returns_figure() -> None:
    table = pd.DataFrame(
        [[0.7, np.nan], [0.9, 1.1]], index=["ets", "sarimax"], columns=["smooth", "lumpy"]
    )
    assert isinstance(plots.plot_heatmap(table, zmid=1.0), go.Figure)
