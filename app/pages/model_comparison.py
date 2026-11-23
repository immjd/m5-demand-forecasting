"""Model comparison: CV and test leaderboard, errors per horizon, class and department."""

import plotly.graph_objects as go
import streamlit as st
from common import (
    CLASS_ORDER,
    LEVEL_VIEWS,
    MODELS,
    MUTED,
    color,
    label,
    load,
    model_line,
    num,
    show,
    style,
)

st.title("Model Comparison")
st.caption(
    "Semua metrik di sini: makin kecil makin baik, kecuali bias (makin dekat nol makin baik)."
)

left, right = st.columns([2, 1])
view = left.selectbox("Level", list(LEVEL_VIEWS))
variants = right.toggle("Tampilkan varian LightGBM & Naive", value=False)
level, subset, metric, metric_label = LEVEL_VIEWS[view]

cv = load("leaderboard_cv").query("level == @level and subset == @subset")
test = load("test_results").query("level == @level and subset == @subset").set_index("model")
if not variants:
    cv = cv[cv["model"].isin(MODELS)]
cv = cv.sort_values(f"{metric}_mean", ascending=False)
names = [label(m) for m in cv["model"]]

fig = go.Figure(
    go.Bar(
        x=cv[f"{metric}_mean"], y=names, orientation="h", name="CV (rata-rata 4 fold)",
        marker={"color": [color(m) for m in cv["model"]], "cornerradius": 4},
        error_x={"type": "data", "array": cv[f"{metric}_std"], "color": MUTED, "thickness": 1.2},
        hovertemplate="%{y}<br>CV %{x:.3f} ± %{error_x.array:.3f}<extra></extra>",
        showlegend=False,
    )
)  # fmt: skip
tested = [m for m in cv["model"] if m in test.index]
fig.add_scatter(
    x=[test.loc[m, metric] for m in tested], y=[label(m) for m in tested], mode="markers",
    name="TEST (sekali jalan)",
    marker={"symbol": "diamond", "size": 11, "color": "#0b0b0b", "line": {"color": "white", "width": 2}},
    hovertemplate="%{y}<br>TEST %{x:.3f}<extra></extra>",
)  # fmt: skip
style(
    fig,
    height=110 + 42 * len(cv),
    bargap=0.35,
    xaxis_title=f"{metric_label} (batang = CV, garis = ± std)",
)
show(fig, "leaderboard")
if len(tested) < len(cv):
    st.caption("Wajik hanya ada pada model yang dikunci untuk test set sebelum hasilnya dilihat.")

with st.expander("Lihat tabel"):
    table = cv.sort_values(f"{metric}_mean").set_index("model")
    cols = {f"{metric}_mean": f"{metric_label} CV", f"{metric}_std": "std", "wape_mean": "WAPE CV",
            "bias_mean": "Bias CV"}  # fmt: skip
    out = table[list(cols)].rename(columns=cols)
    out[f"{metric_label} TEST"] = test[metric].reindex(out.index)
    out["WAPE TEST"] = test["wape"].reindex(out.index)
    out["Bias TEST"] = test["bias"].reindex(out.index)
    out.index = [label(m) for m in out.index]
    st.dataframe(out.style.format(lambda v: "–" if v != v else num(v)), width="stretch")

st.subheader("Error per hari horizon")
hz = load("error_by_horizon").query("level == @level and subset == @subset")
fig = go.Figure()
for m in [m for m in MODELS if m in set(hz["model"])]:
    g = hz[hz["model"] == m].sort_values("h")
    fig.add_trace(model_line(m, g["h"], g["wape"], "%{y:.1%}"))
style(
    fig,
    height=380,
    hovermode="x unified",
    xaxis_title="hari ke-h setelah cutoff",
    yaxis_title="WAPE",
)
fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(dtick=7, tick0=7)
show(fig, "horizon")
st.caption("WAPE gabungan 4 fold CV. Gelombangnya mengikuti hari dalam minggu.")

if level == "item":
    st.subheader("Error per kelas demand")
    st.caption(
        "smooth = rutin dan stabil · erratic = rutin, jumlahnya bervariasi · intermittent = jarang terjual · "
        "lumpy = jarang dan jumlahnya bervariasi. RMSSE rata-rata per item, tanpa bobot revenue."
    )
    cls = load("error_by_class").query("subset == @subset")
    tab = cls.pivot(index="model", columns="demand_class", values="rmsse")[CLASS_ORDER]
    tab = tab.reindex([m for m in MODELS if m in tab.index])
    tab.index = [label(m) for m in tab.index]
    st.dataframe(
        tab.style.format(lambda v: num(v)).highlight_min(
            axis=0, props="font-weight:700;background:#dbe9fb"
        ),
        width="stretch",
    )
    if subset == "all":
        st.subheader("Error per departemen")
        dep = load("error_by_dept").pivot(index="dept", columns="model", values="rmsse")
        dep = dep[[m for m in MODELS if m in dep.columns]]
        dep.columns = [label(m) for m in dep.columns]
        st.dataframe(
            dep.style.format(lambda v: num(v)).highlight_min(
                axis=1, props="font-weight:700;background:#dbe9fb"
            ),
            width="stretch",
        )
    st.caption("Sel tebal = error terkecil. Selisih kecil belum tentu berarti secara statistik.")
