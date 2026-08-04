from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import physicsutil as pu
import modelutil as mu


def render(model, metadata):
    st.header("Quick Calculator")
    st.caption("Test the model for one specific set of conditions.")

    features = mu.get_features(metadata)

    col1, col2 = st.columns(2)
    with col1:
        capacity_kwp = st.number_input("Plant Capacity (kWp)", min_value=0.01, value=100.0, step=1.0, key="t3_capacity")
        date_val = st.date_input("Date", value=datetime.now().date(), key="t3_date")
        time_val = st.time_input("Time", value=datetime.now().time(), key="t3_time")
        irradiation = st.number_input("Irradiation (W/m2)", min_value=0.0, value=float(pu.STC_IRRADIANCE), key="t3_irr")
        ambient_temp = st.number_input("Ambient Temperature (C)", value=float(pu.STC_TEMP), key="t3_temp")
    with col2:
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=0.0, key="t3_wind")
        use_wind = st.checkbox("Use wind speed above (uncheck to use NOCT fallback)", key="t3_usewind")
        cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=0.0, key="t3_cloud")
        inverter_eff = st.number_input("Inverter Efficiency", min_value=0.01, max_value=1.0, value=pu.DEFAULT_INVERTER_EFF, key="t3_inveff")

    run = st.button("Calculate", use_container_width=True, key="t3_calc")
    if not run:
        return

    wind_for_physics = wind_speed if use_wind else None
    wind_for_model = wind_speed if use_wind else np.nan

    module_temp, dc_per_kwp, ac_per_kwp, dc_kw, ac_kw = pu.run_physics(
        capacity_kwp, irradiation, ambient_temp, wind_for_physics, inverter_eff
    )

    forecast_dt = datetime.combine(date_val, time_val)

    if model is None:
        st.warning("Model file not found, showing physics-only result.")
        final_ac_kw = float(ac_kw)
        gap = 0.0
    else:
        X = mu.build_row(irradiation, ambient_temp, wind_for_model, cloud_cover, forecast_dt, features)
        combined_pk, combined_kw, gap_arr = mu.combine(model, X, [irradiation], [float(ac_per_kwp)], capacity_kwp)
        final_ac_kw = float(combined_kw[0])
        gap = float(gap_arr[0])

    mu.render_stat_cards([
        ("Theoretical DC Power", f"{float(dc_kw):.2f} kW"),
        ("Theoretical AC Power", f"{float(ac_kw):.2f} kW"),
        ("Final AC Power", f"{final_ac_kw:.2f} kW"),
        ("Module Temperature", f"{float(module_temp):.1f} C"),
    ])
    mu.render_stat_cards([("ML Correction Applied", f"{gap:+.4f} per kWp")])

    st.subheader("Recommendations")
    pct_vs_theo = ((final_ac_kw - float(ac_kw)) / float(ac_kw) * 100) if float(ac_kw) > 0 else 0.0
    notes = []
    if irradiation <= 0:
        notes.append("No irradiation under these conditions -- expected output is zero.")
    elif abs(pct_vs_theo) < 1:
        notes.append("Expected output is close to the physics-only estimate.")
    elif pct_vs_theo < 0:
        notes.append(f"Expected output is roughly **{abs(pct_vs_theo):.1f}% below** the physics-only estimate.")
    else:
        notes.append(f"Expected output is roughly **{pct_vs_theo:.1f}% above** the physics-only estimate.")
    if float(module_temp) - pu.STC_TEMP > 15:
        notes.append("High module temperature is reducing expected output.")
    if use_wind and wind_speed >= 3:
        notes.append("Wind is helping keep the module cooler.")
    if cloud_cover >= 50:
        notes.append("Cloud cover is a major limiting factor here.")
    if model is None:
        notes.append("This is physics-only -- the ML correction model was not found.")
    for n in notes:
        st.markdown(f"- {n}")

    st.subheader("Charts")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        fig = go.Figure(data=[
            go.Bar(name="DC (physics only)", x=["Power"], y=[float(dc_kw)]),
            go.Bar(name="AC (physics only)", x=["Power"], y=[float(ac_kw)]),
            go.Bar(name="AC (physics + ML)", x=["Power"], y=[final_ac_kw]),
        ])
        fig.update_layout(title="DC vs AC vs Final Prediction", barmode="group", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)

        fig_gauge_temp = go.Figure(go.Indicator(
            mode="gauge+number", value=float(module_temp), title={"text": "Module Temperature (C)"},
            gauge={"axis": {"range": [0, 80]}, "bar": {"color": "#B23B3B"}}
        ))
        st.plotly_chart(fig_gauge_temp, use_container_width=True)

    with vcol2:
        fig_gauge_irr = go.Figure(go.Indicator(
            mode="gauge+number", value=irradiation, title={"text": "Irradiation (W/m2)"},
            gauge={"axis": {"range": [0, 1200]}, "bar": {"color": "#F2A93B"}}
        ))
        st.plotly_chart(fig_gauge_irr, use_container_width=True)

        stc_ratio = irradiation / pu.STC_IRRADIANCE
        temp_loss = float(dc_per_kwp) - stc_ratio
        inverter_loss = float(ac_per_kwp) - float(dc_per_kwp)
        ml_correction = (final_ac_kw / capacity_kwp) - float(ac_per_kwp)
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Irradiance ratio", "Temp. effect", "Inverter loss", "ML correction", "Final (per kWp)"],
            y=[stc_ratio, temp_loss, inverter_loss, ml_correction, 0],
        ))
        fig_waterfall.update_layout(title="Loss / Correction Breakdown (per kWp)", showlegend=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)
