import calendar
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import physicsutil as pu
import modelutil as mu
import weatherapi as wa

SEASON_MONTHS = {
    "Spring (Mar-May)": (3, 5),
    "Summer (Jun-Aug)": (6, 8),
    "Fall (Sep-Nov)": (9, 11),
    "Winter (Dec-Feb)": (12, 2),
}


def month_range(month_idx, year):
    start = date(year, month_idx, 1)
    end = date(year, month_idx, calendar.monthrange(year, month_idx)[1])
    return start, end


def season_range(label, year):
    start_month, end_month = SEASON_MONTHS[label]
    start = date(year, start_month, 1)
    if end_month < start_month:
        end = date(year + 1, end_month, calendar.monthrange(year + 1, end_month)[1])
    else:
        end = date(year, end_month, calendar.monthrange(year, end_month)[1])
    return start, end


def roll_to_future(start, end):
    while end < date.today():
        try:
            start = start.replace(year=start.year + 1)
        except ValueError:
            start = start.replace(year=start.year + 1, day=28)
        try:
            end = end.replace(year=end.year + 1)
        except ValueError:
            end = end.replace(year=end.year + 1, day=28)
    return start, end


def render(model, metadata):
    st.header("Forecast")
    st.caption("Estimate expected generation for an upcoming period, based on historical weather patterns.")

    features = mu.get_features(metadata)
    this_year = date.today().year

    col1, col2 = st.columns(2)
    with col1:
        capacity_kwp = st.number_input("Plant Capacity (kWp)", min_value=0.01, value=100.0, step=1.0, key="t1_capacity")
        location = st.text_input("Location", placeholder="e.g. Chennai, India", key="t1_location")
    with col2:
        mode = st.radio("Pick period by", ["Month", "Season", "Specific Date", "Date Range"], horizontal=True, key="t1_mode")

    if mode == "Month":
        month_name = st.selectbox("Month", list(calendar.month_name)[1:], key="t1_month")
        month_idx = list(calendar.month_name).index(month_name)
        start, end = month_range(month_idx, this_year)
        start, end = roll_to_future(start, end)
    elif mode == "Season":
        season = st.selectbox("Season", list(SEASON_MONTHS.keys()), key="t1_season")
        start, end = season_range(season, this_year)
        start, end = roll_to_future(start, end)
    elif mode == "Specific Date":
        d = st.date_input("Date", value=date.today() + timedelta(days=1), min_value=date.today(), key="t1_date")
        start, end = d, d
    else:
        picked = st.date_input("Date Range",
                                value=(date.today() + timedelta(days=1), date.today() + timedelta(days=7)),
                                min_value=date.today(), key="t1_daterange")
        if isinstance(picked, tuple) and len(picked) == 2:
            start, end = picked
        else:
            st.info("Pick a start and end date.")
            return

    st.caption(f"Forecasting **{start} to {end}** ({(end - start).days + 1} day window), "
               f"projected from up to 10 years of historical weather for this same calendar period.")

    run = st.button("Run Forecast", use_container_width=True, key="t1_run")
    if not run:
        return
    if not location.strip():
        st.error("Location is required.")
        return
    if model is None:
        st.warning("Model file not found, showing physics-only results.")

    lat, lon, name, tz = wa.geocode(location)
    tilt, azimuth = wa.default_tilt_azimuth(lat)
    st.caption(f"Using {name} (lat {lat:.2f}, lon {lon:.2f})")

    with st.spinner("Fetching historical weather..."):
        by_year = wa.fetch_years(lat, lon, tilt, azimuth, start, end, years_back=10, tz=tz)

    if not by_year:
        st.error("No historical weather data found for this location/period.")
        return

    frames = []
    for yr, wdf in by_year.items():
        wdf = wdf.dropna(subset=["IRRADIATION(W/m²)", "AMBIENT_TEMPERATURE(°C)"]).copy()
        if wdf.empty:
            continue
        module_temp, dc_per_kwp, ac_per_kwp, dc_kw, ac_kw = pu.run_physics(
            capacity_kwp, wdf["IRRADIATION(W/m²)"].values, wdf["AMBIENT_TEMPERATURE(°C)"].values,
            wdf["WIND_SPEED(m/s)"].values
        )
        wdf["MODULE_TEMP"] = module_temp
        wdf["THEORETICAL_AC_KW"] = ac_kw
        wdf["THEORETICAL_DC_KW"] = dc_kw

        if model is not None:
            X = mu.build_rows(wdf, features)
            combined_pk, combined_kw, gap = mu.combine(model, X, wdf["IRRADIATION(W/m²)"].values, ac_per_kwp, capacity_kwp)
            wdf["FINAL_AC_KW"] = combined_kw
        else:
            wdf["FINAL_AC_KW"] = wdf["THEORETICAL_AC_KW"]

        wdf["YEAR"] = yr
        wdf["DAY_OFFSET"] = (wdf["DATE_TIME"].dt.date - wdf["DATE_TIME"].dt.date.min()).apply(lambda d: d.days)
        frames.append(wdf)

    data = pd.concat(frames, ignore_index=True)
    num_days = (end - start).days + 1
    yearly_totals = data.groupby("YEAR")["FINAL_AC_KW"].sum()

    daylight = data[data["IRRADIATION(W/m²)"] > 0]
    capacity_factor = data["FINAL_AC_KW"].mean() / capacity_kwp * 100
    daytime_eff = (daylight["FINAL_AC_KW"] / capacity_kwp).mean() * 100 if len(daylight) else 0.0

    out_of_range, reasons = mu.check_out_of_range(lat=lat, requested_months=set(data["DATE_TIME"].dt.month.unique()))
    if out_of_range and model is not None:
        st.info(
            "Heads up: this location/season is outside the window the ML correction was validated against "
            "(a single plant, May-Jun 2020 monsoon season) -- " + "; ".join(reasons) +
            ". The physics baseline still applies anywhere, but treat the AI-corrected numbers here as "
            "directional rather than precise."
        )

    with mu.panel("t1_outputs"):
        st.subheader("Expected Outputs")
        mu.render_stat_cards([
            ("Avg AC Power (physics + AI)", f"{data['FINAL_AC_KW'].mean():.2f} kW"),
            ("Avg AC Power (physics only)", f"{data['THEORETICAL_AC_KW'].mean():.2f} kW"),
            ("Avg DC Power (physics only)", f"{data['THEORETICAL_DC_KW'].mean():.2f} kW"),
            ("Avg Module Temp", f"{data['MODULE_TEMP'].mean():.1f} C"),
        ])
        mu.render_stat_cards([
            ("Avg Daily Yield", f"{yearly_totals.mean() / num_days:.0f} kWh"),
            (
                "Capacity Factor (24h basis)", f"{capacity_factor:.1f}%",
                "Average output over all hours including night, divided by rated capacity.",
            ),
            (
                "Daytime Efficiency", f"{daytime_eff:.1f}%",
                "Average output only during sunlight hours, divided by rated capacity.",
            ),
        ])

    with mu.panel("t1_compare"):
        fig_compare = go.Figure(data=[
            go.Bar(name="DC (physics only)", x=["Power"], y=[data["THEORETICAL_DC_KW"].mean()]),
            go.Bar(name="AC (physics only)", x=["Power"], y=[data["THEORETICAL_AC_KW"].mean()]),
            go.Bar(name="AC (physics + AI)", x=["Power"], y=[data["FINAL_AC_KW"].mean()]),
        ])
        fig_compare.update_layout(title="Physics vs. AI-Corrected Prediction (avg over window)", yaxis_title="kW", barmode="group")
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption(
            "Three ways of estimating the same average power. **DC (physics only)** is straight out of the panel "
            "before conversion losses. **AC (physics only)** applies the inverter efficiency on top. "
            "**AC (physics + AI)** is the final number used everywhere else on this page -- it starts from the "
            "physics estimate and applies the ML correction learned from real plant data."
        )

    with mu.panel("t1_range"):
        st.subheader("Range of Likely Outcomes")
        st.caption("Built from up to 10 past years of weather for this same calendar window -- "
                   "use this to judge how much variation to plan around, not as a record of past output.")
        quantiles = yearly_totals.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        mu.render_stat_cards([
            ("P10", f"{quantiles[0.1]:.0f} kWh"),
            ("P25", f"{quantiles[0.25]:.0f} kWh"),
            ("P50 (Median)", f"{quantiles[0.5]:.0f} kWh"),
            ("P75", f"{quantiles[0.75]:.0f} kWh"),
            ("P90", f"{quantiles[0.9]:.0f} kWh"),
        ])

        fig_box = go.Figure(go.Box(y=yearly_totals.values, boxpoints="all", name="Yearly Totals"))
        fig_box.update_layout(title="Spread of Total Energy for This Window (across past years)", yaxis_title="kWh")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(
            "Each dot is one historical year's total energy for this exact calendar window. The box shows the "
            "middle 50% of years and the line inside it is the median -- a tall box or scattered dots mean this "
            "period's output varies a lot year to year, so treat a single-number estimate with more caution."
        )

    with mu.panel("t1_shape"):
        st.subheader("Forecast Shape")
        ch1, ch2 = st.columns(2)
        with ch1:
            hourly = data.groupby(data["DATE_TIME"].dt.hour)["FINAL_AC_KW"].mean().reset_index()
            hourly.columns = ["hour", "avg_kw"]
            fig_hour = px.line(hourly, x="hour", y="avg_kw", title="Typical Day Shape (avg AC power by hour)")
            st.plotly_chart(fig_hour, use_container_width=True)
            st.caption(
                "The average shape of a single day within this window, averaged across every hour of every "
                "historical year fetched. Shows when generation typically starts, peaks, and ends -- useful for "
                "figuring out when to run high-draw appliances."
            )

            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(name="Irradiation (W/m2)", x=["Avg"], y=[data["IRRADIATION(W/m²)"].mean()]))
            fig_w.add_trace(go.Bar(name="Ambient Temp (C)", x=["Avg"], y=[data["AMBIENT_TEMPERATURE(°C)"].mean()]))
            fig_w.update_layout(title="Average Weather Conditions", barmode="group")
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption(
                "The average irradiance and ambient temperature the forecast is based on, averaged over every hour "
                "(day and night) in the window. Low irradiation relative to a sunny location, or very high "
                "temperatures, help explain lower-than-expected output."
            )

        with ch2:
            daily_per_year = data.groupby(["YEAR", "DAY_OFFSET"])["FINAL_AC_KW"].sum().reset_index()
            avg_by_day = daily_per_year.groupby("DAY_OFFSET")["FINAL_AC_KW"].mean().reset_index()
            avg_by_day["DAY_OFFSET"] = avg_by_day["DAY_OFFSET"] + 1
            fig_day = px.bar(avg_by_day, x="DAY_OFFSET", y="FINAL_AC_KW",
                              title="Expected Daily Yield Within Window (avg across years)")
            st.plotly_chart(fig_day, use_container_width=True)
            st.caption(
                "Total expected energy for each individual day within the window (day 1, day 2, ...), averaged "
                "across all historical years. A flat line means fairly stable conditions throughout the window; a "
                "slope or dip means some days in this window are typically better than others."
            )

    with mu.panel("t1_interpretation"):
        st.subheader("Interpretation")
        theo_mean = data["THEORETICAL_AC_KW"].mean()
        final_mean = data["FINAL_AC_KW"].mean()
        correction_pct = ((final_mean - theo_mean) / theo_mean * 100) if theo_mean > 0 else 0.0
        st.markdown(f"- For **{start} to {end}**, expect roughly **{quantiles[0.1]:.0f} to {quantiles[0.9]:.0f} kWh** total (P10-P90 range), median **{quantiles[0.5]:.0f} kWh**.")
        if abs(correction_pct) < 1:
            st.markdown("- The AI correction is minimal here -- the physics-only estimate is already close to real-world behavior for these conditions.")
        elif correction_pct < 0:
            st.markdown(f"- Real-world output tends to run **{abs(correction_pct):.1f}% below** the physics-only estimate for these conditions.")
        else:
            st.markdown(f"- Real-world output tends to run **{correction_pct:.1f}% above** the physics-only estimate for these conditions.")
        st.markdown(f"- Daytime efficiency runs around **{daytime_eff:.1f}%** of rated capacity.")
        if model is None:
            st.markdown("- These numbers are physics-only since the ML correction model was not found.")

        st.download_button(
            "Download forecast data (CSV)",
            data.to_csv(index=False).encode("utf-8"),
            file_name="solar_forecast.csv",
            mime="text/csv",
        )
