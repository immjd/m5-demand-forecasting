"""Shared data loading, labels and chart helpers for the dashboard pages.

The app depends only on streamlit, pandas, pyarrow and plotly (see requirements.txt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
REPO_URL = "https://github.com/immjd/m5-demand-forecasting"
RULES_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules"

# model -> (label, colour, dash). Colours follow the model everywhere, in a fixed,
# colour-blind-checked order; the dash is a second cue so identity is never colour alone.
MODELS: dict[str, tuple[str, str, str]] = {
    "lgbm_tuned": ("LightGBM (tuned)", "#2a78d6", "solid"),
    "sarimax": ("SARIMAX", "#eb6834", "dash"),
    "ets": ("ETS", "#1baf7a", "dashdot"),
    "moving_average": ("Moving Average 28", "#eda100", "dot"),
    "seasonal_naive": ("Seasonal Naive", "#e87ba4", "longdash"),
    "prophet": ("Prophet", "#6250d6", "longdashdot"),
}
OTHER_LABELS = {
    "naive": "Naive",
    "lgbm_tweedie": "LightGBM (tweedie, default)",
    "lgbm_poisson": "LightGBM (poisson)",
    "lgbm_l2": "LightGBM (L2)",
    "lgbm_no_price": "LightGBM tanpa harga",
    "lgbm_no_calendar": "LightGBM tanpa kalender",
}
CLASS_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LEVEL_VIEWS = {
    "Item · semua (3.049 item)": ("item", "all", "wrmsse", "WRMSSE-lite"),
    "Item · sampel 200 item": ("item", "stat_sample", "wrmsse", "WRMSSE-lite"),
    "Agregat · 11 deret": ("agg", "all", "rmsse", "RMSSE"),
}


def label(model: str) -> str:
    """Readable model name."""
    return MODELS[model][0] if model in MODELS else OTHER_LABELS.get(model, model)


def color(model: str) -> str:
    """Fixed colour of a model; variants outside the main set are grey."""
    return MODELS[model][1] if model in MODELS else "#8a8984"


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    """One precomputed table from ``app/artifacts``."""
    return pd.read_parquet(ARTIFACTS / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_meta() -> dict[str, Any]:
    """Export settings (horizon, grid, class sizes)."""
    return json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8"))


def num(value: float, digits: int = 3) -> str:
    """Indonesian number format: 1.234,5."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    """Share as a percentage in Indonesian format."""
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{num(100 * value, digits)}%".replace("-", "−")


def usd_k(value: float) -> str:
    """Dollar amount in thousands."""
    return f"${num(value / 1000, 1)}k"


def series_label(series_id: str) -> str:
    """Readable name of an aggregate series."""
    if series_id.startswith("CA_"):
        return f"Total toko {series_id}"
    kind = "Departemen" if series_id[-1].isdigit() else "Kategori"
    return f"{kind} {series_id}"


def style(fig: go.Figure, height: int = 400, **layout: Any) -> go.Figure:
    """Common look: recessive grid, legend on top, Indonesian number separators."""
    fig.update_layout(
        template="plotly_white", height=height, separators=",.",
        margin={"l": 10, "r": 20, "t": 40, "b": 10}, font={"color": INK, "size": 13},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0}, **layout,
    )  # fmt: skip
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    return fig


def show(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure at full width."""
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def model_line(model: str, x: Any, y: Any, hover: str, **kw: Any) -> go.Scatter:
    """Line trace with the model's fixed colour and dash."""
    name, col, dash = MODELS[model]
    return go.Scatter(
        x=x, y=y, name=name, mode=kw.pop("mode", "lines"),
        line={"color": col, "width": 2, "dash": dash},
        hovertemplate=f"{name}: {hover}<extra></extra>", **kw,
    )  # fmt: skip


def inventory_at_fill(grid: pd.DataFrame, target: float) -> pd.Series:
    """Average inventory value per model at one fill rate (linear interpolation, NaN outside)."""
    out = {}
    for model, g in grid.groupby("model"):
        g = g.sort_values("fill_rate")
        x, y = g["fill_rate"].to_numpy(), g["avg_inventory_value"].to_numpy()
        if not x[0] <= target <= x[-1]:
            out[model] = float("nan")
            continue
        i = max(1, int((x < target).sum()))
        out[model] = float(y[i - 1] + (target - x[i - 1]) / (x[i] - x[i - 1]) * (y[i] - y[i - 1]))
    return pd.Series(out)


def delta(value: float) -> str:
    """Metric delta text; needs an ASCII minus so Streamlit picks the right arrow."""
    return pct(value).replace("−", "-")


def kpi_cards(n: int, width: int = 175) -> list[Any]:
    """``n`` fixed-width cards that wrap to the next row instead of truncating numbers."""
    row = st.container(horizontal=True, gap="small")
    return [row.container(width=width) for _ in range(n)]
