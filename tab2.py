import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import physicsutil as pu


def render():
    st.header("Plant Performance Analyzer")
    st.caption("Upload your plant's data and map your columns to what the app needs.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="t2_upload")
    if uploaded is None:
        st.info("Upload a CSV to see the report.")
        return

    df = pd.read_csv(uploaded)
    columns = ["None"] + list(df.columns)

    st.subheader("Map Your Columns")
    capacity_kwp = st.number_input("Plant Capacity (kWp) (optional, enables efficiency comparison)",
                                     min_value=0.0, value=0.0, key="t2_capacity")

    col1, col2 = st.columns(2)
    with col1:
        ts_col = st.selectbox("Timestamp column", columns, key="t2_ts")
        ac_col = st.selectbox("Actual AC Power (kW) column", columns, key="t2_ac")
        dc_col = st.selectbox("Actual DC Power (kW) column (optional)", columns, key="t2_dc")
        irr_col = st.selectbox("Irradiation column (optional)", columns, key="t2_irr")
    with col2:
        temp_col = st.selectbox("Ambient Temperature column (optional)", columns, key="t2_temp")
        wind_col = st.selectbox("Wind Speed column (optional)", columns, key="t2_wind")
        cloud_col = st.selectbox("Cloud Cover column (optional)", columns, key="t2_cloud")

    run = st.button("Analyze", use_container_width=True, key="t2_analyze")
    if not run:
        return
    if ts_col == "None" or ac_col == "None":
        st.error("Timestamp and Actual AC Power columns are required.")
        return

    data = pd.DataFrame()
    data["DATE_TIME"] = pd.to_datetime(df[ts_col])
    data["ACTUAL_AC_KW"] = df[ac_col]
    if dc_col != "None":
        data["ACTUAL_DC_KW"] = df[dc_col]
    if irr_col != "None":
        data["IRRADIATION"] = df[irr_col]
    if temp_col != "None":
        data["AMBIENT_TEMP"] = df[temp_col]
    if wind_col != "None":
        data["WIND_SPEED"] = df[wind_col]
    if cloud_col != "None":
        data["CLOUD_COVER"] = df[cloud_col]

    data = data.sort_values("DATE_TIME")

    diffs = data["DATE_TIME"].drop_duplicates().sort_values().diff().dropna()
    interval_hours = diffs.median().total_seconds() / 3600.0 if len(diffs) else 1.0

    total_energy = (data["ACTUAL_AC_KW"] * interval_hours).sum()
    avg_ac = data["ACTUAL_AC_KW"].mean()
    peak_row = data.loc[data["ACTUAL_AC_KW"].idxmax()]
    daily = data.groupby(data["DATE_TIME"].dt.date)["ACTUAL_AC_KW"].apply(lambda s: (s * interval_hours).sum())

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Energy", f"{total_energy:.0f} kWh")
    c2.metric("Average AC Power", f"{avg_ac:.2f} kW")
    c3.metric("Peak AC Power", f"{peak_row['ACTUAL_AC_KW']:.2f} kW")

    c4, c5 = st.columns(2)
    c4.metric("Best Day", f"{daily.max():.0f} kWh")
    c5.metric("Worst Day", f"{daily.min():.0f} kWh")

    if capacity_kwp > 0 and "IRRADIATION" in data.columns and "AMBIENT_TEMP" in data.columns:
        module_temp, dc_per_kwp, ac_per_kwp, dc_kw, ac_kw = pu.run_physics(
            capacity_kwp, data["IRRADIATION"].values, data["AMBIENT_TEMP"].values,
            data["WIND_SPEED"].values if "WIND_SPEED" in data.columns else None
        )
        data["THEORETICAL_AC_KW"] = ac_kw
        data["EFFICIENCY_RATIO"] = data["ACTUAL_AC_KW"] / data["THEORETICAL_AC_KW"].replace(0, np.nan)
        eff_ratio = data["EFFICIENCY_RATIO"].mean()
        st.metric("Average Efficiency Ratio", f"{eff_ratio*100:.1f}%",
                   help="Actual output divided by theoretical output.")

    st.subheader("Charts")
    ch1, ch2 = st.columns(2)
    with ch1:
        daily_df = daily.reset_index()
        daily_df.columns = ["date", "kwh"]
        fig1 = px.bar(daily_df, x="date", y="kwh", title="Daily Yield")
        st.plotly_chart(fig1, use_container_width=True)

        hourly = data.groupby(data["DATE_TIME"].dt.hour)["ACTUAL_AC_KW"].mean().reset_index()
        hourly.columns = ["hour", "avg_kw"]
        fig2 = px.bar(hourly, x="hour", y="avg_kw", title="Average Hourly Output")
        st.plotly_chart(fig2, use_container_width=True)

    with ch2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data["DATE_TIME"], y=data["ACTUAL_AC_KW"], name="Actual AC"))
        fig3.update_layout(title="Power Output Over Time", yaxis_title="kW")
        st.plotly_chart(fig3, use_container_width=True)

        if "EFFICIENCY_RATIO" in data.columns:
            fig4 = px.histogram(data, x="EFFICIENCY_RATIO", nbins=30, title="Efficiency Ratio Distribution")
            st.plotly_chart(fig4, use_container_width=True)

    heat = data.copy()
    heat["hour"] = heat["DATE_TIME"].dt.hour
    heat["date"] = heat["DATE_TIME"].dt.date
    pivot = heat.pivot_table(index="hour", columns="date", values="ACTUAL_AC_KW", aggfunc="mean")
    fig5 = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                      labels=dict(x="Date", y="Hour", color="kW"), title="Production Heatmap")
    st.plotly_chart(fig5, use_container_width=True)

    if "AMBIENT_TEMP" in data.columns:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=data["DATE_TIME"], y=data["AMBIENT_TEMP"], name="Ambient Temp"))
        if "THEORETICAL_AC_KW" in data.columns:
            fig6.add_trace(go.Scatter(x=data["DATE_TIME"], y=data["ACTUAL_AC_KW"], name="Actual AC", yaxis="y2"))
            fig6.update_layout(yaxis2=dict(overlaying="y", side="right", title="AC Power (kW)"))
        fig6.update_layout(title="Ambient Temperature Over Time", yaxis_title="C")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Insights")
    insights = []
    if "EFFICIENCY_RATIO" in data.columns:
        insights.append(f"Plant runs at roughly {eff_ratio*100:.0f}% of theoretical expectations on average.")
        data["hour"] = data["DATE_TIME"].dt.hour
        afternoon = data[data["hour"].between(12, 17)]["EFFICIENCY_RATIO"].mean()
        morning = data[data["hour"].between(6, 11)]["EFFICIENCY_RATIO"].mean()
        if pd.notna(afternoon) and pd.notna(morning) and afternoon < morning:
            insights.append("Afternoon efficiency runs lower than morning, consistent with higher module temperatures.")
    insights.append(f"Best day produced {daily.max():.0f} kWh, worst produced {daily.min():.0f} kWh.")
    if not insights:
        insights.append("Map more columns (irradiation, ambient temperature) to unlock efficiency-based insights.")
    for i in insights:
        st.markdown(f"- {i}")