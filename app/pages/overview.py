"""Overview: the problem, headline results and the validation design."""

import streamlit as st
from common import (
    REPO_URL,
    delta,
    inventory_at_fill,
    kpi_cards,
    load,
    load_meta,
    num,
    pct,
    show,
    style,
    usd_k,
)

from forecasting.viz.plots import plot_backtest_folds

meta = load_meta()
cv, test, grid = load("leaderboard_cv"), load("test_results"), load("inventory_grid")


def score(df, model: str, level: str, metric: str, subset: str = "all") -> float:
    row = df[(df["model"] == model) & (df["level"] == level) & (df["subset"] == subset)]
    return float(row[metric].iloc[0])


st.title("Forecasting permintaan & perencanaan stok")
st.markdown(
    f"""
Seorang planner di satu toko Walmart (**{meta["store"]}**, {num(meta["n_items"], 0)} item) harus
memutuskan setiap minggu berapa banyak barang yang dipesan. Dashboard ini membandingkan model
statistik dan model global berbasis fitur untuk memprediksi penjualan **{meta["horizon"]} hari ke
depan**, lalu mengukur dampaknya ke stok lewat simulasi. Kode: [GitHub]({REPO_URL}).
"""
)

lgb, ma = (
    score(test, "lgbm_tuned", "item", "wrmsse"),
    score(test, "moving_average", "item", "wrmsse"),
)
lgb_cv = score(cv, "lgbm_tuned", "item", "wrmsse_mean")
lgb_sd = score(cv, "lgbm_tuned", "item", "wrmsse_std")
agg_l, agg_s = score(test, "lgbm_tuned", "agg", "rmsse"), score(test, "sarimax", "agg", "rmsse")
base = grid[
    (grid["review"] == meta["default_review"])
    & (grid["lead"] == meta["default_lead"])
    & (grid["demand_class"] == "semua")
]
need = inventory_at_fill(base, 0.97)
vs_sn = need["lgbm_tuned"] / need["seasonal_naive"] - 1
vs_ma = need["lgbm_tuned"] / need["moving_average"] - 1

c1, c2, c3, c4 = kpi_cards(4)
c1.metric("WRMSSE test", num(lgb), delta(lgb / ma - 1), delta_color="inverse")
c1.caption(f"LightGBM vs MA28, semua item. CV: {num(lgb_cv)} ± {num(lgb_sd)}")
c2.metric("RMSSE agregat", num(agg_l), delta(agg_l / agg_s - 1), delta_color="inverse")
c2.caption("LightGBM vs SARIMAX di test, 11 deret")
c3.metric("Nilai stok", usd_k(need["lgbm_tuned"]), delta(vs_sn),
          delta_color="inverse")  # fmt: skip
c3.caption(f"LightGBM vs Seasonal Naive, fill rate 97%, {meta['sim_days']} hari")
c4.metric(
    "Stok vs MA28", pct(vs_ma, signed=True), "praktis sama", delta_color="off", delta_arrow="off"
)
c4.caption("LightGBM vs MA28, fill rate 97%")

st.subheader("Tiga temuan utama")
st.markdown(
    f"""
1. **LightGBM menang di metrik utama**, dan skor test ({num(lgb)}) masih di dalam rentang CV
   ({num(lgb_cv)} ± {num(lgb_sd)}). Di sampel 200 item, LightGBM dan SARIMAX bergantian menang dengan
   selisih lebih kecil dari variasi antar-fold.
2. **Forecast yang lebih akurat tidak otomatis berarti stok lebih hemat.** LightGBM menghemat
   {pct(abs(vs_sn))} modal stok dibanding Seasonal Naive, tetapi hampir sama dengan Moving Average 28 hari.
3. **Target service level lebih menentukan biaya daripada pilihan model.** Coba sendiri di halaman
   *Inventory Planner*.
"""
)

st.subheader("Validasi: rolling-origin backtest")
fig = style(plot_backtest_folds(load("folds")), height=300)
fig.update_layout(title=None)
fig.update_xaxes(range=["2015-06-01", "2016-06-01"])
show(fig, "folds")
st.caption(
    "Model dilatih dengan data sampai sebuah tanggal, lalu diuji pada 28 hari berikutnya. Tanggal itu "
    "digeser 4 kali (F1–F4). Jendela TEST hanya dievaluasi sekali, setelah semua model dikunci. "
    "Sumbu dipotong di 2015; data training mulai 2011."
)
