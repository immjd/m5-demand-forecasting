"""Entry point of the dashboard. It only reads precomputed tables and trains nothing.

Run from the project root:  python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(page_title="Demand Forecasting M5", page_icon="📦", layout="wide")

page = st.navigation(
    [
        st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("pages/model_comparison.py", title="Model Comparison", icon=":material/leaderboard:"),
        st.Page("pages/forecast_explorer.py", title="Forecast Explorer", icon=":material/timeline:"),
        st.Page("pages/inventory_planner.py", title="Inventory Planner", icon=":material/inventory_2:"),
        st.Page("pages/methodology.py", title="Methodology & Limitations", icon=":material/menu_book:"),
    ]
)  # fmt: skip
page.run()

st.divider()
st.caption(
    "Data: kompetisi M5 Forecasting - Accuracy (University of Nicosia, Kaggle; penjualan Walmart). "
    "Dashboard ini non-komersial, hanya menampilkan hasil agregat, dan tidak berafiliasi dengan "
    "pihak-pihak tersebut."
)
