import pandas as pd
import streamlit as st
import plotly.express as px

import physicsutil as pu
import modelutil as mu


def render(model, metadata):
    st.header("How It Works")

    st.subheader("Overview")
    st.markdown(
        "This app estimates solar plant electricity generation by combining a physics-based model "
        "with a machine-learning correction trained on real plant data. Physics alone misses real-world "
        "losses like soiling and wiring. ML alone would not generalize outside its training data. "
        "Combining them keeps predictions grounded."
    )

    st.subheader("Methodology")
    st.markdown(
        "```\n"
        "Weather Inputs\n"
        "     |\n"
        "Physics Model -> Module Temperature -> DC Power -> AC Power (theoretical)\n"
        "     |\n"
        "Machine Learning -> predicts the real-world gap vs theoretical\n"
        "     |\n"
        "Final Prediction = Theoretical AC + ML Correction\n"
        "```"
    )

    st.subheader("Inputs")
    st.markdown(
        "- Plant Capacity (kWp): required, scales every result to kW.\n"
        f"- Irradiation (W/m2): default {pu.STC_IRRADIANCE} (STC).\n"
        f"- Ambient Temperature (C): default {pu.STC_TEMP} (STC).\n"
        "- Wind Speed (m/s): optional, falls back to the NOCT formula if blank.\n"
        f"- Cloud Cover (%): default {pu.DEFAULT_CLOUD_COVER}, used by the ML model only.\n"
        f"- Inverter Efficiency: default {pu.DEFAULT_INVERTER_EFF}."
    )

    st.subheader("Outputs")
    st.markdown(
        "- Theoretical AC/DC Power: physics-only prediction.\n"
        "- Final AC Power: physics + ML correction.\n"
        "- Capacity Factor: average output over 24 hours (including night), divided by capacity.\n"
        "- Daytime Efficiency: average output during sunlight hours only, divided by capacity.\n"
        "- Efficiency Ratio (Analyzer tab): actual output divided by theoretical output."
    )

    st.subheader("Model Performance")
    features = mu.get_features(metadata)
    if model is not None and hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance")
        fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance needs the trained model file.")

    if metadata:
        c1, c2, c3 = st.columns(3)
        c1.metric("Backtest RMSE (kW)", f"{metadata.get('backtest_rmse_kw', float('nan')):.1f}")
        c2.metric("Backtest MAE (kW)", f"{metadata.get('backtest_mae_kw', float('nan')):.1f}")
        c3.metric("Backtest R2", f"{metadata.get('backtest_r2_kw', float('nan')):.3f}")
        n_train = metadata.get("n_train")
        n_test = metadata.get("n_test")
        if n_train and n_test:
            st.caption(f"Trained on {n_train:,} rows, tested on {n_test:,} rows.")
        with st.expander("Full metadata"):
            st.json(metadata)
    else:
        st.info("Metadata file not found.")

    st.subheader("Limitations")
    st.markdown(
        "- Extreme or unusual weather outside the training range.\n"
        "- Equipment faults, soiling, or degradation not captured by the model.\n"
        "- Missing input values reduce accuracy (e.g. no wind data uses the NOCT fallback).\n"
        "- Trained on a limited historical window from one plant/region."
    )