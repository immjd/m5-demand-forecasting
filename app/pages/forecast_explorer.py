"""Forecast explorer: indexed test forecasts of the 11 aggregate series with intervals."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import INK, MODELS, load, load_meta, model_line, num, pct, series_label, show, style

meta = load_meta()
hist, fc, cov = load("series_history"), load("series_forecast"), load("daily_coverage")
models = [m for m in MODELS if m in set(fc["model"])]

st.title("Forecast Explorer")
st.caption(
    f"Forecast {meta['horizon']} hari di jendela test. Semua deret ditampilkan sebagai indeks "
    f"(rata-rata {meta['history_days']} hari sebelum test = 100), bukan jumlah unit."
)

c1, c2, c3 = st.columns([2, 2, 1])
order = {"store": 0, "cat": 1, "dept": 2}
ids = (
    hist[["series_id", "agg_level"]].drop_duplicates()
    .assign(o=lambda d: d["agg_level"].map(order).astype(int)).sort_values(["o", "series_id"])["series_id"]
)  # fmt: skip
series = c1.selectbox("Deret", list(ids), format_func=series_label)
chosen = c2.multiselect("Model", models, default=["lgbm_tuned", "sarimax"],
                        format_func=lambda m: MODELS[m][0])  # fmt: skip
days = c3.slider("Histori (hari)", 28, meta["history_days"], 84, step=7)

names = {"Tanpa interval": None} | {
    f"Interval {lv:.0%}": round(lv * 100) for lv in meta["interval_levels"]
}
pick = st.radio("Interval prediksi", list(names), index=1, horizontal=True)
level = names[pick]

h = hist[(hist["series_id"] == series) & (hist["is_closed"] == 0)].sort_values("date")
past = h[h["part"] == "history"].tail(days)
actual = h[h["part"] == "test"]
f = fc[fc["series_id"] == series].sort_values("date")

fig = go.Figure()
fig.add_scatter(x=past["date"], y=past["index"], name="Histori", mode="lines",
                line={"color": "#b4b3ad", "width": 1.5}, hovertemplate="Histori: %{y:.1f}<extra></extra>")  # fmt: skip
for m in chosen:
    g = f[f["model"] == m]
    if level and len(chosen) <= 2:
        r, gr, b = (int(MODELS[m][1][i : i + 2], 16) for i in (1, 3, 5))
        fig.add_scatter(
            x=pd.concat([g["date"], g["date"][::-1]]),
            y=pd.concat([g[f"hi{level}"], g[f"lo{level}"][::-1]]),
            fill="toself", fillcolor=f"rgba({r},{gr},{b},0.14)", line={"width": 0},
            hoverinfo="skip", showlegend=False,
        )  # fmt: skip
    fig.add_trace(model_line(m, g["date"], g["forecast"], "%{y:.1f}"))
fig.add_scatter(x=actual["date"], y=actual["index"], name="Aktual", mode="lines+markers",
                line={"color": INK, "width": 2}, marker={"size": 5},
                hovertemplate="Aktual: %{y:.1f}<extra></extra>")  # fmt: skip
style(
    fig, height=460, hovermode="x unified", yaxis_title="indeks penjualan (100 = rata-rata histori)"
)
fig.update_xaxes(tickformat="%d %b %Y", hoverformat="%a, %d %b %Y")
show(fig, "forecast")
if level and len(chosen) > 2:
    st.info("Interval hanya digambar untuk maksimal 2 model supaya grafik tetap terbaca.")

st.subheader("Error di jendela test untuk deret ini")
rows = []
truth = actual.set_index("date")["index"]
for m in models:
    g = f[f["model"] == m].set_index("date").reindex(truth.index)
    e = g["forecast"] - truth
    row = {
        "Model": MODELS[m][0],
        "WAPE": e.abs().sum() / truth.sum(),
        "Bias": e.sum() / truth.sum(),
    }
    for lv in meta["interval_levels"]:
        n = round(lv * 100)
        row[f"Aktual di dalam interval {n}%"] = truth.between(g[f"lo{n}"], g[f"hi{n}"]).mean()
    rows.append(row)
tab = pd.DataFrame(rows).set_index("Model").sort_values("WAPE")
st.dataframe(tab.style.format(lambda v: pct(v)), width="stretch")
st.caption(
    f"Satu deret hanya punya {len(truth)} hari test, jadi angka ini mudah bergeser. Bias negatif = forecast terlalu rendah."
)

with st.expander("Seberapa jujur intervalnya? (semua 11 deret)"):
    wide = cov.pivot(index="model", columns="level", values="coverage")
    wide = wide.reindex(models)
    wide.index = [MODELS[m][0] for m in wide.index]
    wide.columns = [f"Target {c:.0%}" for c in wide.columns]
    st.dataframe(wide.style.format(lambda v: pct(v)), width="stretch")
    st.markdown(
        "Interval dihitung dari kuantil error harian tiap deret di fold F1–F4 (112 hari), bukan dari jendela "
        "test. Interval 95% bergantung pada sedikit hari ekstrem, jadi paling kasar. Angka di bawah target "
        f"berarti interval terlalu sempit di jendela test. Deret agregat saling tumpang tindih, sehingga "
        f"{num(11 * len(truth), 0)} titik ini tidak independen."
    )
