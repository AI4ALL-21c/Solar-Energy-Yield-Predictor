import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import physicsutil as pu
import modelutil as mu
from constants import PLANT_CAPACITY_KWP

DATA_PATH = "SOLAR_PLANT_DATA(GENERATION_AND_WEATHER).csv"


@st.cache_data
def load_plant_data(_model, features):
    df = pd.read_csv(DATA_PATH)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    # This file's WIND_SPEED(m/s) column is mislabeled and is actually km/h.
    df["WIND_SPEED(m/s)"] = df["WIND_SPEED(m/s)"] / 3.6

    module_temp, dc_per_kwp, ac_per_kwp, dc_kw, ac_kw = pu.run_physics(
        PLANT_CAPACITY_KWP,
        df["IRRADIATION(W/m²)"].values,
        df["AMBIENT_TEMPERATURE(°C)"].values,
        df["WIND_SPEED(m/s)"].values,
    )
    df["THEORETICAL_AC_KW"] = ac_kw

    if _model is not None:
        X = mu.build_rows(df, features)
        combined_pk, combined_kw, gap = mu.combine(
            _model, X, df["IRRADIATION(W/m²)"].values, ac_per_kwp, PLANT_CAPACITY_KWP
        )
        df["FINAL_AC_KW"] = combined_kw
    else:
        df["FINAL_AC_KW"] = df["THEORETICAL_AC_KW"]
    return df


def render(model, metadata):
    st.header("Plant Performance Analyzer")
    st.caption("Explore how well the model tracks the real training plant's recorded output over a chosen window.")

    features = mu.get_features(metadata)
    plant_df = load_plant_data(model, features)
    min_date, max_date = plant_df["DATE_TIME"].min().date(), plant_df["DATE_TIME"].max().date()
    picked_range = st.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="t2_daterange"
    )
    if isinstance(picked_range, tuple) and len(picked_range) == 2:
        range_start, range_end = picked_range
    else:
        range_start, range_end = min_date, max_date

    mask = (plant_df["DATE_TIME"].dt.date >= range_start) & (plant_df["DATE_TIME"].dt.date <= range_end)
    window_df = plant_df.loc[mask]

    if window_df.empty:
        st.warning("No data in the selected range.")
        return

    interval_hours = 0.25  # the plant's data is recorded every 15 minutes
    actual_kwh = window_df["ACTUAL_AC_POWER(kW)"].sum() * interval_hours
    final_kwh = window_df["FINAL_AC_KW"].sum() * interval_hours
    theo_kwh = window_df["THEORETICAL_AC_KW"].sum() * interval_hours
    mae = (window_df["ACTUAL_AC_POWER(kW)"] - window_df["FINAL_AC_KW"]).abs().mean()

    mu.render_stat_cards([
        ("Actual Yield", f"{actual_kwh:,.0f} kWh"),
        ("Predicted Yield (physics + AI)", f"{final_kwh:,.0f} kWh"),
        ("Predicted Yield (physics only)", f"{theo_kwh:,.0f} kWh"),
        ("Mean Absolute Error", f"{mae:,.1f} kW"),
    ])
    st.caption(
        "**Actual Yield** is what the plant really produced in this window. **Predicted Yield (physics + AI)** is "
        "the app's final estimate for the same window. **Predicted Yield (physics only)** is the estimate before "
        "the ML correction, so you can see how much the correction is doing. **Mean Absolute Error** is the "
        "average gap, in kW, between the physics + AI prediction and reality -- lower is better."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window_df["DATE_TIME"], y=window_df["ACTUAL_AC_POWER(kW)"], name="Actual", line=dict(color="#f2c14e")
    ))
    fig.add_trace(go.Scatter(
        x=window_df["DATE_TIME"], y=window_df["FINAL_AC_KW"], name="Predicted (physics + AI)", line=dict(color="#2A5298")
    ))
    fig.add_trace(go.Scatter(
        x=window_df["DATE_TIME"], y=window_df["THEORETICAL_AC_KW"], name="Physics baseline",
        line=dict(color="gray", dash="dash")
    ))
    fig.update_layout(title="Actual vs Predicted AC Power", xaxis_title="Time", yaxis_title="kW", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Yellow is what the plant actually produced. Blue is the app's final prediction (physics + AI). The dashed "
        "gray line is the physics-only estimate with no ML correction -- the gap between blue and gray is what the "
        "ML correction is contributing."
    )

    residual = window_df["ACTUAL_AC_POWER(kW)"] - window_df["FINAL_AC_KW"]
    fig_resid = go.Figure(go.Scatter(x=window_df["DATE_TIME"], y=residual, mode="lines", line=dict(color="#b23b3b")))
    fig_resid.update_layout(title="Residual (Actual - Predicted)", xaxis_title="Time", yaxis_title="kW", height=350)
    st.plotly_chart(fig_resid, use_container_width=True)
    st.caption(
        "How far off the physics + AI prediction was at each point in time. Values near zero mean an accurate "
        "prediction; a spike above zero means the plant produced more than predicted, below zero means it "
        "produced less."
    )
