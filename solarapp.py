import streamlit as st
import modelutil as mu
import tab1
import tab2
import tab3
import tab4

st.set_page_config(page_title="Solar Forecast", page_icon="☀️", layout="wide")
mu.inject_theme_css()

model = mu.load_model()
metadata = mu.load_metadata()

mu.render_banner("☀️ Solar Forecast", "Physics-based solar yield estimates, corrected with a trained ML model.")

if model is None:
    st.sidebar.warning("Model file not found, running physics-only.")

tabs = st.tabs(["Forecast", "Plant Performance Analyzer", "Quick Calculator", "How It Works"])

with tabs[0]:
    tab1.render(model, metadata)
with tabs[1]:
    tab2.render(model, metadata)
with tabs[2]:
    tab3.render(model, metadata)
with tabs[3]:
    tab4.render(model, metadata)
