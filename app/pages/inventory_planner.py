"""Inventory planner: precomputed (R, S) simulations per demand class."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    CLASS_ORDER,
    INK,
    MODELS,
    inventory_at_fill,
    load,
    load_meta,
    num,
    pct,
    show,
    style,
    usd_k,
)

meta, grid = load_meta(), load("inventory_grid")
models = [m for m in MODELS if m in set(grid["model"])]

st.title("Inventory Planner")
st.caption(
    f"Simulasi {num(meta['n_items'], 0)} item selama {meta['sim_days']} hari dengan kebijakan (R, S): stok "
    "dicek setiap R hari, barang datang L hari kemudian, dan permintaan yang tidak terlayani hilang. "
    "Semua kombinasi sudah dihitung sebelumnya."
)

c1, c2, c3, c4 = st.columns(4)
review = c1.selectbox("Review period R (hari)", meta["review_days"],
                      index=meta["review_days"].index(meta["default_review"]))  # fmt: skip
lead = c2.selectbox(
    "Lead time L (hari)", meta["lead_days"], index=meta["lead_days"].index(meta["default_lead"])
)
level = c3.select_slider("Target service level", meta["service_levels"], value=0.95,
                         format_func=lambda v: pct(v, 1))  # fmt: skip
model = c4.selectbox("Model forecast", models, format_func=lambda m: MODELS[m][0])

cell = grid[(grid["review"] == review) & (grid["lead"] == lead)]
total = cell[cell["demand_class"] == "semua"]
row = total[(total["model"] == model) & (total["level"] == level)].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fill rate tercapai", pct(row["fill_rate"]))
k2.metric("Rata-rata nilai stok", usd_k(row["avg_inventory_value"]))
k3.metric("Tutupan stok", f"{num(row['cover_days'], 1)} hari")
k4.metric(f"Holding cost ({meta['sim_days']} hari)", usd_k(row["holding_cost"]))

st.subheader("Trade-off: fill rate vs nilai stok")
fig = go.Figure()
for m in models:
    g = total[total["model"] == m].sort_values("level")
    name, col, dash = MODELS[m]
    fig.add_scatter(
        x=g["avg_inventory_value"], y=g["fill_rate"], name=name, mode="lines+markers",
        line={"color": col, "width": 2, "dash": dash}, marker={"size": 8},
        customdata=g["level"],
        hovertemplate=f"{name}<br>target %{{customdata:.1%}}<br>fill rate %{{y:.1%}}<br>stok $%{{x:,.0f}}<extra></extra>",
    )  # fmt: skip
fig.add_scatter(
    x=[row["avg_inventory_value"]], y=[row["fill_rate"]], name="Pilihanmu", mode="markers",
    marker={"size": 18, "symbol": "circle-open", "color": INK, "line": {"width": 3}},
    hovertemplate="Pilihanmu: fill rate %{y:.1%}, stok $%{x:,.0f}<extra></extra>",
)  # fmt: skip
style(
    fig, height=430, xaxis_title="rata-rata nilai stok harian ($)", yaxis_title="fill rate tercapai"
)
fig.update_xaxes(tickprefix="$", tickformat="~s")
fig.update_yaxes(tickformat=".0%")
show(fig, "tradeoff")
st.caption(
    "Kurva yang lebih ke kiri atas lebih efisien. Setiap titik adalah satu target service level."
)

targets = [0.95, 0.97, 0.98, 0.99]
need = {pct(t, 0): inventory_at_fill(total, t) for t in targets}
st.markdown("**Nilai stok yang dibutuhkan untuk fill rate yang sama**")
rows = {
    f"Fill rate {t}": {
        **{MODELS[m][0]: "–" if s[m] != s[m] else usd_k(s[m]) for m in models},
        **{
            f"LightGBM vs {MODELS[m][0]}": "–" if s[m] != s[m] or s["lgbm_tuned"] != s["lgbm_tuned"]
            else pct(s["lgbm_tuned"] / s[m] - 1, signed=True)
            for m in models if m != "lgbm_tuned"
        },
    }
    for t, s in need.items()
}  # fmt: skip
st.table(pd.DataFrame(rows).T)
st.caption("Interpolasi linear antar titik. Tanda – berarti di luar rentang simulasi.")

st.subheader(f"Per kelas demand · {MODELS[model][0]}, target {pct(level)}")
per = cell[(cell["model"] == model) & (cell["level"] == level) & (cell["demand_class"] != "semua")]
per = per.set_index("demand_class").reindex(CLASS_ORDER)
table = per.rename(
    columns={
        "n_items": "Jumlah item", "fill_rate": "Fill rate", "avg_inventory_value": "Nilai stok ($)",
        "cover_days": "Tutupan stok (hari)", "safety_stock_share": "Porsi safety stock dari S",
        "stockout_rate": "Item-hari dengan stockout",
    }
)[["Jumlah item", "Fill rate", "Nilai stok ($)", "Tutupan stok (hari)", "Porsi safety stock dari S",
   "Item-hari dengan stockout"]]  # fmt: skip
table.index.name = "Kelas demand"
st.dataframe(
    table.style.format(
        {
            "Jumlah item": lambda v: num(v, 0), "Fill rate": lambda v: pct(v),
            "Nilai stok ($)": lambda v: num(v, 0), "Tutupan stok (hari)": lambda v: num(v, 1),
            "Porsi safety stock dari S": lambda v: pct(v), "Item-hari dengan stockout": lambda v: pct(v, 2),
        }
    ),
    width="stretch",
)  # fmt: skip
st.download_button(
    "Unduh tabel (CSV)",
    table.round(4).to_csv().encode("utf-8"),
    file_name=f"inventory_R{review}_L{lead}_{model}_{level}.csv",
    mime="text/csv",
)
st.caption(
    "S = target stok saat memesan (forecast selama R + L hari + safety stock). Tutupan stok dihitung "
    "gabungan per kelas (total stok / total permintaan harian), jadi berbeda dari median per item di "
    "notebook 05. Tabel per item tidak "
    "ditampilkan karena aturan kompetisi M5 tidak mengizinkan publikasi datanya."
)
