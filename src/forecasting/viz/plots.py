"""Reusable Plotly figures for the notebooks and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forecasting.config import resolve_path

TEMPLATE = "plotly_white"
CATEGORY_COLORS = {"FOODS": "#2a9d8f", "HOUSEHOLD": "#e9c46a", "HOBBIES": "#e76f51"}
CLASS_COLORS = {
    "smooth": "#2a9d8f",
    "erratic": "#e9c46a",
    "intermittent": "#457b9d",
    "lumpy": "#e76f51",
    "insufficient_history": "#adb5bd",
}
WEEKDAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _layout(fig: go.Figure, title: str, **kwargs: Any) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, title_x=0.01, **kwargs)
    return fig


def plot_daily_with_trend(
    df: pd.DataFrame, x: str = "date", y: str = "sales", window: int = 28, title: str = ""
) -> go.Figure:
    """Daily series (light) plus a centred rolling mean (bold) to show level and trend."""
    s = df.sort_values(x)
    trend = s[y].rolling(window, center=True, min_periods=window // 2).mean()
    fig = go.Figure()
    fig.add_scatter(
        x=s[x], y=s[y], mode="lines", name="harian", line={"width": 0.6, "color": "#adb5bd"}
    )
    fig.add_scatter(
        x=s[x],
        y=trend,
        mode="lines",
        name=f"rata-rata {window} hari",
        line={"width": 2.2, "color": "#1d3557"},
    )
    return _layout(fig, title, xaxis_title=None, yaxis_title="unit terjual", height=380)


def plot_weekday_profile(profile: pd.DataFrame, title: str = "") -> go.Figure:
    """Index of sales by weekday (1.0 = weekly mean), one line per category.

    Args:
        profile: Columns ``weekday, cat_id, index``.
    """
    fig = px.line(
        profile,
        x="weekday",
        y="index",
        color="cat_id",
        markers=True,
        category_orders={"weekday": WEEKDAYS},
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6c757d")
    return _layout(
        fig,
        title,
        xaxis_title=None,
        yaxis_title="indeks (1 = rata-rata minggu)",
        legend_title=None,
        height=360,
    )


def plot_month_profile(profile: pd.DataFrame, title: str = "") -> go.Figure:
    """Index of sales by month (1.0 = yearly mean), one line per category.

    Args:
        profile: Columns ``month`` (1-12), ``cat_id``, ``index``.
    """
    p = profile.assign(month_name=profile["month"].map(lambda m: MONTHS[m - 1]))
    fig = px.line(
        p,
        x="month_name",
        y="index",
        color="cat_id",
        markers=True,
        category_orders={"month_name": MONTHS},
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6c757d")
    return _layout(
        fig,
        title,
        xaxis_title=None,
        yaxis_title="indeks (1 = rata-rata tahun)",
        legend_title=None,
        height=360,
    )


def plot_effect_bars(effects: pd.DataFrame, label: str, value: str, title: str = "") -> go.Figure:
    """Horizontal bars of a relative effect in percent (e.g. event lift vs. baseline)."""
    e = effects.sort_values(value)
    colors = ["#e76f51" if v < 0 else "#2a9d8f" for v in e[value]]
    fig = go.Figure(
        go.Bar(
            x=e[value],
            y=e[label],
            orientation="h",
            marker_color=colors,
            text=e[value].map(lambda v: f"{v:+.0f}%"),
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#6c757d")
    return _layout(
        fig,
        title,
        xaxis_title="perubahan vs. baseline (%)",
        yaxis_title=None,
        height=max(300, 26 * len(e) + 120),
        showlegend=False,
    )


def plot_price_vs_sales(item_df: pd.DataFrame, title: str = "", window: int = 7) -> go.Figure:
    """Weekly-smoothed sales (bars) against sell price (line) for one item."""
    s = item_df.sort_values("date")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(
        x=s["date"],
        y=s["sales"].rolling(window, min_periods=1).mean(),
        name=f"sales (rata-rata {window} hari)",
        mode="lines",
        fill="tozeroy",
        line={"color": "#457b9d", "width": 1.2},
        secondary_y=False,
    )
    fig.add_scatter(
        x=s["date"],
        y=s["sell_price"],
        name="harga",
        mode="lines",
        line={"color": "#e63946", "width": 2, "shape": "hv"},
        secondary_y=True,
    )
    fig.update_yaxes(title_text="unit / hari", secondary_y=False)
    fig.update_yaxes(title_text="harga ($)", secondary_y=True)
    return _layout(fig, title, height=330, legend={"orientation": "h", "y": -0.2})


def plot_pareto(cum_share: pd.Series, mark: float = 0.8, title: str = "") -> go.Figure:
    """Cumulative revenue share vs. share of items (items sorted by revenue)."""
    n = len(cum_share)
    x = [(i + 1) / n * 100 for i in range(n)]
    y = cum_share.to_numpy() * 100
    k = int((cum_share < mark).sum()) + 1
    fig = go.Figure(
        go.Scatter(
            x=x, y=y, mode="lines", line={"color": "#1d3557", "width": 2.5}, name="kumulatif"
        )
    )
    fig.add_hline(y=mark * 100, line_dash="dot", line_color="#e76f51")
    fig.add_vline(x=k / n * 100, line_dash="dot", line_color="#e76f51")
    fig.add_annotation(
        x=k / n * 100,
        y=mark * 100,
        text=f"{k / n:.0%} item = {mark:.0%} revenue",
        showarrow=True,
        ax=60,
        ay=40,
    )
    return _layout(
        fig,
        title,
        xaxis_title="% item (urut dari revenue terbesar)",
        yaxis_title="% revenue kumulatif",
        height=380,
        showlegend=False,
    )


def plot_adi_cv2(
    classes: pd.DataFrame,
    adi_threshold: float,
    cv2_threshold: float,
    title: str = "",
    y_quantile: float = 0.995,
) -> go.Figure:
    """Syntetos-Boylan scatter (log ADI vs CV^2) coloured by demand class.

    The y-axis is capped at the ``y_quantile`` of CV^2 so a few extreme items do
    not flatten the rest; capped items are counted in the axis title.
    """
    c = classes.dropna(subset=["adi", "cv2"])
    y_max = float(c["cv2"].quantile(y_quantile)) * 1.1
    n_hidden = int((c["cv2"] > y_max).sum())
    fig = px.scatter(
        c,
        x="adi",
        y="cv2",
        color="demand_class",
        log_x=True,
        opacity=0.55,
        color_discrete_map=CLASS_COLORS,
        hover_data=["item_id", "dept_id"],
        category_orders={"demand_class": list(CLASS_COLORS)},
    )
    fig.update_traces(marker={"size": 5})
    fig.add_vline(x=adi_threshold, line_dash="dash", line_color="#343a40")
    fig.add_hline(y=cv2_threshold, line_dash="dash", line_color="#343a40")
    fig.update_yaxes(range=[-0.02 * y_max, y_max])
    y_title = "CV² ukuran penjualan"
    if n_hidden:
        y_title += f"<br>({n_hidden} item terpotong)"
    return _layout(
        fig,
        title,
        xaxis_title="ADI (log) — jarak rata-rata antar-penjualan (hari)",
        yaxis_title=y_title,
        legend_title=None,
        height=450,
    )


def plot_class_share(classes: pd.DataFrame, by: str = "cat_id", title: str = "") -> go.Figure:
    """Stacked share of demand classes within each group (e.g. category)."""
    share = (
        classes.groupby([by, "demand_class"], observed=True)
        .size()
        .groupby(level=0)
        .transform(lambda s: s / s.sum() * 100)
        .rename("pct")
        .reset_index()
    )
    fig = px.bar(
        share,
        x=by,
        y="pct",
        color="demand_class",
        color_discrete_map=CLASS_COLORS,
        category_orders={"demand_class": list(CLASS_COLORS)},
        text_auto=".0f",
    )
    return _layout(
        fig, title, xaxis_title=None, yaxis_title="% item", legend_title=None, height=380
    )


def plot_store_comparison(
    overview: pd.DataFrame, metric: str, highlight: str, title: str = "", axis_title: str = ""
) -> go.Figure:
    """Bar chart of one store-level metric with the chosen store highlighted."""
    o = overview.sort_values(metric, ascending=False)
    colors = ["#e76f51" if s == highlight else "#a8dadc" for s in o["store_id"]]
    fig = go.Figure(go.Bar(x=o["store_id"], y=o[metric], marker_color=colors))
    return _layout(
        fig, title, xaxis_title=None, yaxis_title=axis_title or metric, height=340, showlegend=False
    )


MODEL_COLORS = {
    "naive": "#adb5bd",
    "seasonal_naive": "#6c757d",
    "moving_average": "#343a40",
    "ets": "#457b9d",
    "sarimax": "#2a9d8f",
    "prophet": "#e9c46a",
    "lgbm": "#e76f51",
}


def plot_backtest_folds(folds: pd.DataFrame, title: str = "") -> go.Figure:
    """Timeline of the rolling-origin backtest: one row per fold, train vs validation blocks.

    Args:
        folds: Columns ``fold, train_start, train_end, valid_start, valid_end`` (dates).
            A fold named ``TEST`` is drawn in a different colour.
    """
    fig = go.Figure()
    shown = set()
    for _, r in folds.iloc[::-1].iterrows():
        is_test = r["fold"] == "TEST"
        blocks = [
            ("train", r["train_start"], r["train_end"], "#a8dadc"),
            (
                "test" if is_test else "validasi",
                r["valid_start"],
                r["valid_end"],
                "#e63946" if is_test else "#f4a261",
            ),
        ]
        for label, start, end, color in blocks:
            fig.add_bar(
                # on a date axis Plotly reads bar length in milliseconds
                y=[r["fold"]],
                x=[((pd.Timestamp(end) - pd.Timestamp(start)).days + 1) * 86_400_000],
                base=[pd.Timestamp(start)],
                orientation="h",
                marker_color=color,
                name=label,
                showlegend=label not in shown,
                hovertemplate=f"{label}: {pd.Timestamp(start).date()} – {pd.Timestamp(end).date()}"
                "<extra></extra>",
            )
            shown.add(label)
    fig.update_xaxes(type="date")
    return _layout(
        fig,
        title,
        barmode="overlay",
        height=320,
        xaxis_title=None,
        yaxis_title=None,
        legend={"orientation": "h", "y": -0.2},
    )


def plot_stl(series: pd.Series, period: int = 7, title: str = "") -> go.Figure:
    """STL decomposition (observed, trend, seasonal, residual) of a date-indexed series."""
    from statsmodels.tsa.seasonal import STL

    res = STL(series, period=period, robust=True).fit()
    parts = {
        "observed": series,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "residual": res.resid,
    }
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=list(parts)
    )
    for i, (name, values) in enumerate(parts.items(), start=1):
        fig.add_scatter(
            x=series.index,
            y=values,
            mode="lines",
            name=name,
            line={"width": 1, "color": "#1d3557"},
            row=i,
            col=1,
        )
    return _layout(fig, title, height=650, showlegend=False)


def plot_acf_pacf(y: pd.Series, lags: int = 35, title: str = "") -> go.Figure:
    """ACF and PACF bars with the approximate 95% band (±1.96 / sqrt(n))."""
    from statsmodels.tsa.stattools import acf, pacf

    values = pd.Series(y).dropna().to_numpy()
    band = 1.96 / len(values) ** 0.5
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    for col, vals in enumerate([acf(values, nlags=lags), pacf(values, nlags=lags)], start=1):
        lag_idx = list(range(1, lags + 1))
        colors = ["#e76f51" if k % 7 == 0 else "#457b9d" for k in lag_idx]
        fig.add_bar(x=lag_idx, y=vals[1:], marker_color=colors, row=1, col=col)
        for sign in (1, -1):
            fig.add_hline(y=sign * band, line_dash="dot", line_color="#6c757d", row=1, col=col)
    fig.update_xaxes(title_text="lag (hari) — oranye = kelipatan 7")
    return _layout(fig, title, height=340, showlegend=False)


def plot_forecast_vs_actual(
    history: pd.DataFrame, actual: pd.DataFrame, preds: pd.DataFrame, title: str = ""
) -> go.Figure:
    """History + actual validation values + one line per model forecast.

    Args:
        history: ``date, sales`` before the cutoff.
        actual: ``date, y_true`` in the validation window.
        preds: ``date, model, y_pred`` in the validation window.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=history["date"],
        y=history["sales"],
        mode="lines",
        name="history",
        line={"color": "#adb5bd", "width": 1},
    )
    fig.add_scatter(
        x=actual["date"],
        y=actual["y_true"],
        mode="lines+markers",
        name="aktual",
        line={"color": "#1d3557", "width": 2},
        marker={"size": 4},
    )
    for model, g in preds.groupby("model", sort=False):
        fig.add_scatter(
            x=g["date"],
            y=g["y_pred"],
            mode="lines",
            name=model,
            line={"color": MODEL_COLORS.get(model), "width": 2, "dash": "dash"},
        )
    return _layout(
        fig,
        title,
        height=360,
        xaxis_title=None,
        yaxis_title="unit / hari",
        legend={"orientation": "h", "y": -0.2},
    )


def plot_leaderboard(
    board: pd.DataFrame, metric: str, title: str = "", benchmark: str | None = None
) -> go.Figure:
    """Horizontal bars of ``<metric>_mean`` with ±std error bars, best model on top.

    Args:
        board: Rows from ``leaderboard_cv.csv`` for one level/subset.
        benchmark: Model name drawn as a dashed reference line.
    """
    b = board.sort_values(f"{metric}_mean", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=b[f"{metric}_mean"],
            y=b["model"],
            orientation="h",
            error_x={"type": "data", "array": b[f"{metric}_std"].fillna(0)},
            marker_color=[MODEL_COLORS.get(m, "#457b9d") for m in b["model"]],
            text=b[f"{metric}_mean"].map(lambda v: f"{v:.3f}"),
            textposition="inside",
        )
    )
    if benchmark is not None and benchmark in set(b["model"]):
        ref = float(b.loc[b["model"] == benchmark, f"{metric}_mean"].iloc[0])
        fig.add_vline(
            x=ref,
            line_dash="dash",
            line_color="#e63946",
            annotation_text=f"benchmark ({benchmark})",
            annotation_position="top",
        )
    return _layout(
        fig,
        title,
        xaxis_title=f"{metric.upper()} (rata-rata ± std 4 fold)",
        yaxis_title=None,
        height=max(280, 45 * len(b) + 120),
        showlegend=False,
    )


def plot_metric_by_group(
    df: pd.DataFrame, group: str, value: str, title: str = "", axis_title: str = ""
) -> go.Figure:
    """Grouped bars: ``value`` per ``group`` (e.g. demand class), one colour per model."""
    fig = px.bar(
        df,
        x=group,
        y=value,
        color="model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        text_auto=".2f",
    )
    return _layout(
        fig, title, xaxis_title=None, yaxis_title=axis_title or value, legend_title=None, height=400
    )


def save_png(
    fig: go.Figure, name: str, config: dict[str, Any], scale: float = 2.0, width: int = 1000
) -> Path:
    """Export a figure to ``<paths.figures>/<name>.png`` for the README."""
    out_dir = resolve_path(config, "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.write_image(path, scale=scale, width=width, height=fig.layout.height or 450)
    return path
