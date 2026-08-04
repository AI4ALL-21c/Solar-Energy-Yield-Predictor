import json
import pathlib
from datetime import date, datetime, time, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from constants import PLANT_CAPACITY_KWP
from pipeline_functions import (
    MONTH_NAMES,
    SEASON_MONTHS,
    STC_IRRADIANCE,
    build_feature_row,
    calculate_ac_power_per_kwp,
    calculate_dc_power_per_kwp,
    estimate_module_temperature,
    features as FEATURE_COLUMNS,
    fetch_archive,
    geocode_location,
    historical_windows,
    next_month_window,
    next_season_window,
    predict_combined_ac_kw,
    run_forecast_pipeline,
)

APP_DIR = pathlib.Path(__file__).parent
DATA_PATH = APP_DIR / 'SOLAR_PLANT_DATA(GENERATION_AND_WEATHER).csv'
MODEL_PATH = APP_DIR / 'solar_gap_model.joblib'
METADATA_PATH = APP_DIR / 'solar_gap_model_metadata.json'

st.set_page_config(
    page_title='Solar Forecast',
    page_icon='☀️',
    layout='wide',
)

SOLAR_PANEL_SVG = """
<svg viewBox="0 0 220 150" width="180" height="123" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Solar panel illustration">
  <circle cx="196" cy="22" r="13" fill="#FFC94A"/>
  <g stroke="#FFC94A" stroke-width="2.5" stroke-linecap="round">
    <line x1="196" y1="2" x2="196" y2="8"/>
    <line x1="196" y1="36" x2="196" y2="42"/>
    <line x1="172" y1="22" x2="178" y2="22"/>
    <line x1="214" y1="22" x2="220" y2="22"/>
    <line x1="180" y1="6" x2="184" y2="10"/>
    <line x1="208" y1="34" x2="212" y2="38"/>
    <line x1="212" y1="6" x2="208" y2="10"/>
    <line x1="184" y1="34" x2="180" y2="38"/>
  </g>
  <g transform="rotate(-6 110 75)">
    <rect x="20" y="35" width="180" height="80" rx="5" fill="#173F2E"/>
    <rect x="26.0" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="54.5" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="83.0" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="111.5" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="140.0" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="168.5" y="41.0" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="26.0" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="54.5" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="83.0" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="111.5" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="140.0" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="168.5" y="64.7" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="26.0" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="54.5" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="83.0" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="111.5" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="140.0" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/><rect x="168.5" y="88.3" width="25.5" height="20.7" rx="1.5" fill="#0F2A1D" stroke="#8CC63F" stroke-width="0.8"/>
    <rect x="105" y="115" width="10" height="24" fill="#173F2E"/>
    <polygon points="80,139 140,139 150,146 70,146" fill="#0B2416"/>
  </g>
</svg>
"""

st.markdown(
    """
    <style>
    .app-banner {
        background: linear-gradient(135deg, #173F2E 0%, #1F5C3F 100%);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        flex-wrap: wrap;
    }
    .app-banner h1 { color: white; margin: 0; font-size: 2.4rem; }
    .app-banner p { color: #C9E6B8; margin: 6px 0 0 0; font-size: 1rem; }
    .app-banner-art { flex-shrink: 0; }
    @media (max-width: 900px) { .app-banner-art { display: none; } }
    .stat-card { border-radius: 10px; padding: 18px 14px; text-align: center; color: white; margin-bottom: 10px; }
    .stat-card .stat-value { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
    .stat-card .stat-label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
        opacity: 0.9; margin-top: 4px;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

STAT_CARD_COLORS = ['#8CC63F', '#173F2E', '#2C7873', '#4A4A4A']


def render_stat_cards(items):
    """items: list of (label, value) or (label, value, tooltip) tuples, rendered as colored cards."""
    cols = st.columns(len(items))
    for i, (col, item) in enumerate(zip(cols, items)):
        label, value = item[0], item[1]
        tooltip = item[2] if len(item) > 2 else ''
        color = STAT_CARD_COLORS[i % len(STAT_CARD_COLORS)]
        with col:
            st.markdown(
                f'<div class="stat-card" style="background:{color};" title="{tooltip}">'
                f'<div class="stat-value">{value}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(
            'Could not load the trained XGBoost model. This is almost always a '
            'missing native dependency, not a bug in the app:\n\n'
            '- **macOS**: run `brew install libomp` (XGBoost needs the OpenMP '
            'runtime library), then restart the app.\n'
            '- **Linux**: install `libgomp1` (e.g. `sudo apt install libgomp1`).\n'
            '- **Windows**: install the Visual C++ Redistributable.\n\n'
            f'Original error: {exc}'
        )
        st.stop()


@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text())


@st.cache_data
def load_plant_data(_model) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    # This file's WIND_SPEED(m/s) column is mislabeled and is actually km/h.
    df['WIND_SPEED(m/s)'] = df['WIND_SPEED(m/s)'] / 3.6
    return run_forecast_pipeline(df, capacity_kwp=PLANT_CAPACITY_KWP, model=_model)


model = load_model()
metadata = load_metadata()

st.markdown(
    '<div class="app-banner">'
    '<div><h1>☀️ Solar Forecast</h1>'
    '<p>Physics-based solar yield estimates, corrected with a trained ML model.</p></div>'
    f'<div class="app-banner-art">{SOLAR_PANEL_SVG}</div>'
    '</div>',
    unsafe_allow_html=True,
)

tab_forecast, tab_analyzer, tab_calc, tab_how = st.tabs(
    ['Forecast', 'Plant Performance Analyzer', 'Quick Calculator', 'How It Works']
)

# ---------------------------------------------------------------- Forecast --
with tab_forecast:
    st.header('Forecast')
    st.caption('Estimate expected generation for an upcoming period, based on historical weather patterns.')

    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        capacity_kwp = st.number_input('Plant Capacity (kWp)', min_value=0.1, value=100.0, step=10.0)
    with col_right:
        period_mode = st.radio(
            'Pick period by', ['Month', 'Season', 'Specific Date', 'Date Range'], horizontal=True
        )

    location_input = st.text_input('Location', placeholder='e.g. Chennai, India')

    if period_mode == 'Month':
        month_choice = st.selectbox('Month', MONTH_NAMES, index=0)
        start_date, end_date = next_month_window(MONTH_NAMES.index(month_choice) + 1)
    elif period_mode == 'Season':
        season_choice = st.selectbox('Season', list(SEASON_MONTHS.keys()), index=0)
        start_date, end_date = next_season_window(season_choice)
    elif period_mode == 'Specific Date':
        start_date = st.date_input('Date', value=date.today() + timedelta(days=30))
        end_date = start_date
    else:
        default_start = date.today() + timedelta(days=30)
        picked = st.date_input('Date Range', value=(default_start, default_start + timedelta(days=6)))
        if isinstance(picked, tuple) and len(picked) == 2:
            start_date, end_date = picked
        else:
            start_date = end_date = picked

    n_days = (end_date - start_date).days + 1
    st.caption(
        f'Forecasting **{start_date.isoformat()} to {end_date.isoformat()}** ({n_days} day window), '
        'projected from up to 10 years of historical weather for this same calendar period.'
    )

    run_forecast = st.button('Run Forecast', width='stretch')

    if run_forecast:
        if not location_input.strip():
            st.error('Please enter a location.')
        else:
            try:
                with st.spinner(f'Looking up "{location_input}"...'):
                    lat, lon, tz = geocode_location(location_input)
                st.caption(f'Using {location_input} (lat {lat:.2f}, lon {lon:.2f})')

                windows = historical_windows(start_date, end_date, historical_years=10)
                by_year_weather = {}
                with st.spinner('Fetching historical weather...'):
                    for yr, (w_start, w_end) in sorted(windows.items()):
                        try:
                            by_year_weather[yr] = fetch_archive(
                                lat, lon, w_start.isoformat(), w_end.isoformat(), tz
                            )
                        except requests.exceptions.RequestException:
                            continue

                if not by_year_weather:
                    st.error('Could not fetch historical weather for this location and period.')
                else:
                    by_year_results = {
                        yr: run_forecast_pipeline(wdf, capacity_kwp, model)
                        for yr, wdf in by_year_weather.items()
                    }
                    combined_all_years = pd.concat(by_year_results.values(), ignore_index=True)

                    avg_ac_ai = combined_all_years['FINAL_AC_KW'].mean()
                    avg_ac_physics = combined_all_years['THEORETICAL_AC_KW'].mean()
                    avg_dc_physics = combined_all_years['DC_POWER_KW'].mean()
                    avg_module_temp = combined_all_years['MODULE_TEMPERATURE(°C)'].mean()
                    avg_daily_yield = avg_ac_ai * 24
                    capacity_factor = avg_ac_ai / capacity_kwp * 100
                    daytime_mask = combined_all_years['IRRADIATION(W/m²)'] > 0
                    daytime_efficiency = (
                        combined_all_years.loc[daytime_mask, 'FINAL_AC_KW'].mean() / capacity_kwp * 100
                        if daytime_mask.any() else 0.0
                    )

                    st.subheader('Expected Outputs')
                    render_stat_cards([
                        ('Avg AC Power (physics + AI)', f'{avg_ac_ai:.2f} kW'),
                        ('Avg AC Power (physics only)', f'{avg_ac_physics:.2f} kW'),
                        ('Avg DC Power (physics only)', f'{avg_dc_physics:.2f} kW'),
                        ('Avg Module Temp', f'{avg_module_temp:.1f} C'),
                    ])
                    render_stat_cards([
                        ('Avg Daily Yield', f'{avg_daily_yield:,.0f} kWh'),
                        (
                            'Capacity Factor (24h basis)', f'{capacity_factor:.1f}%',
                            'Average AC power (physics + AI), divided by plant capacity, averaged over all 24 hours of the day.',
                        ),
                        (
                            'Daytime Efficiency', f'{daytime_efficiency:.1f}%',
                            'Average AC power (physics + AI), divided by plant capacity, averaged only over hours with nonzero irradiation.',
                        ),
                    ])

                    st.subheader('Physics vs. AI-Corrected Prediction (avg over window)')
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(x=['Power'], y=[avg_dc_physics], name='DC (physics only)', marker_color='#8fd3fe'))
                    fig_compare.add_trace(go.Bar(x=['Power'], y=[avg_ac_physics], name='AC (physics only)', marker_color='#1f6fb4'))
                    fig_compare.add_trace(go.Bar(x=['Power'], y=[avg_ac_ai], name='AC (physics + AI)', marker_color='#f7a8a8'))
                    fig_compare.update_layout(yaxis_title='kW', height=450)
                    st.plotly_chart(fig_compare, width='stretch')

                    if len(by_year_results) > 1:
                        st.subheader('Year-to-Year Variability')
                        yearly_totals = [df['FINAL_AC_KW'].sum() * 1.0 for df in by_year_results.values()]
                        fig_box = go.Figure(go.Box(y=yearly_totals, name='Yearly Totals', boxpoints='all', jitter=0.5, pointpos=-1.8))
                        fig_box.update_layout(yaxis_title='kWh', height=400)
                        st.plotly_chart(fig_box, width='stretch')

                    st.subheader('Forecast Shape')
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        hourly_shape = (
                            combined_all_years.assign(hour=combined_all_years['DATE_TIME'].dt.hour)
                            .groupby('hour')['FINAL_AC_KW'].mean().reset_index().rename(columns={'FINAL_AC_KW': 'avg_kw'})
                        )
                        fig_day = go.Figure(go.Scatter(
                            x=hourly_shape['hour'], y=hourly_shape['avg_kw'], mode='lines', line=dict(color='#173F2E')
                        ))
                        fig_day.update_layout(
                            title='Typical Day Shape (avg AC power by hour)', xaxis_title='hour', yaxis_title='avg_kw', height=400
                        )
                        st.plotly_chart(fig_day, width='stretch')
                    with fc2:
                        daily_frames = []
                        for yr_df in by_year_results.values():
                            daily = yr_df.groupby(yr_df['DATE_TIME'].dt.date)['FINAL_AC_KW'].sum().reset_index()
                            daily.columns = ['date', 'FINAL_AC_KW']
                            daily['DAY_OFFSET'] = range(1, len(daily) + 1)
                            daily_frames.append(daily)
                        daily_avg = pd.concat(daily_frames, ignore_index=True).groupby('DAY_OFFSET')['FINAL_AC_KW'].mean().reset_index()
                        fig_daily = go.Figure(go.Bar(x=daily_avg['DAY_OFFSET'], y=daily_avg['FINAL_AC_KW'], marker_color='#8CC63F'))
                        fig_daily.update_layout(
                            title='Expected Daily Yield Within Window (avg across years)',
                            xaxis_title='DAY_OFFSET', yaxis_title='FINAL_AC_KW', height=400,
                        )
                        st.plotly_chart(fig_daily, width='stretch')

                    st.subheader('Average Weather Conditions')
                    avg_irr = combined_all_years['IRRADIATION(W/m²)'].mean()
                    avg_temp = combined_all_years['AMBIENT_TEMPERATURE(°C)'].mean()
                    fig_weather = go.Figure()
                    fig_weather.add_trace(go.Bar(x=['Irradiation (W/m2)'], y=[avg_irr], name='Irradiation (W/m2)', marker_color='#8CC63F'))
                    fig_weather.add_trace(go.Bar(x=['Ambient Temp (C)'], y=[avg_temp], name='Ambient Temp (C)', marker_color='#173F2E'))
                    fig_weather.update_layout(height=400)
                    st.plotly_chart(fig_weather, width='stretch')
            except requests.exceptions.RequestException as exc:
                st.error(f'Could not reach the weather API: {exc}')
            except ValueError as exc:
                st.error(str(exc))

# -------------------------------------------------- Plant Performance Analyzer --
with tab_analyzer:
    st.header('Plant Performance Analyzer')
    st.caption("Explore how well the model tracks the real training plant's recorded output over a chosen window.")

    plant_df = load_plant_data(model)
    min_date, max_date = plant_df['DATE_TIME'].min().date(), plant_df['DATE_TIME'].max().date()
    picked_range = st.date_input('Date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(picked_range, tuple) and len(picked_range) == 2:
        range_start, range_end = picked_range
    else:
        range_start, range_end = min_date, max_date

    mask = (plant_df['DATE_TIME'].dt.date >= range_start) & (plant_df['DATE_TIME'].dt.date <= range_end)
    window_df = plant_df.loc[mask]

    if window_df.empty:
        st.warning('No data in the selected range.')
    else:
        interval_hours = 0.25
        actual_kwh = window_df['ACTUAL_AC_POWER(kW)'].sum() * interval_hours
        final_kwh = window_df['FINAL_AC_KW'].sum() * interval_hours
        theo_kwh = window_df['THEORETICAL_AC_KW'].sum() * interval_hours
        mae = (window_df['ACTUAL_AC_POWER(kW)'] - window_df['FINAL_AC_KW']).abs().mean()

        render_stat_cards([
            ('Actual Yield', f'{actual_kwh:,.0f} kWh'),
            ('Predicted Yield (physics + AI)', f'{final_kwh:,.0f} kWh'),
            ('Predicted Yield (physics only)', f'{theo_kwh:,.0f} kWh'),
            ('Mean Absolute Error', f'{mae:,.1f} kW'),
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_df['DATE_TIME'], y=window_df['ACTUAL_AC_POWER(kW)'], name='Actual', line=dict(color='#f2c14e')))
        fig.add_trace(go.Scatter(x=window_df['DATE_TIME'], y=window_df['FINAL_AC_KW'], name='Predicted (physics + AI)', line=dict(color='#2A5298')))
        fig.add_trace(go.Scatter(x=window_df['DATE_TIME'], y=window_df['THEORETICAL_AC_KW'], name='Physics baseline', line=dict(color='gray', dash='dash')))
        fig.update_layout(title='Actual vs Predicted AC Power', xaxis_title='Time', yaxis_title='kW', height=450)
        st.plotly_chart(fig, width='stretch')

        residual = window_df['ACTUAL_AC_POWER(kW)'] - window_df['FINAL_AC_KW']
        fig_resid = go.Figure(go.Scatter(x=window_df['DATE_TIME'], y=residual, mode='lines', line=dict(color='#b23b3b')))
        fig_resid.update_layout(title='Residual (Actual - Predicted)', xaxis_title='Time', yaxis_title='kW', height=350)
        st.plotly_chart(fig_resid, width='stretch')

# ------------------------------------------------------- Quick Calculator --
with tab_calc:
    st.header('Quick Calculator')
    st.caption('Test the model for one specific set of conditions.')

    qc_left, qc_right = st.columns(2)
    with qc_left:
        qc_capacity = st.number_input('Plant Capacity (kWp)', min_value=0.1, value=100.0, step=10.0, key='qc_capacity')
        qc_date = st.date_input('Date', value=date.today(), key='qc_date')
        qc_time = st.time_input('Time', value=time(6, 58), key='qc_time')
        qc_irradiation = st.number_input('Irradiation (W/m2)', min_value=0.0, value=1000.0, step=10.0, key='qc_irr')
        qc_ambient = st.number_input('Ambient Temperature (C)', value=25.0, step=0.5, key='qc_ambient')
    with qc_right:
        qc_wind = st.number_input('Wind Speed (m/s)', min_value=0.0, value=0.0, step=0.5, key='qc_wind')
        qc_use_wind = st.checkbox('Use wind speed above (uncheck to use NOCT fallback)', value=False, key='qc_use_wind')
        qc_cloud = st.number_input('Cloud Cover (%)', min_value=0.0, max_value=100.0, value=0.0, step=5.0, key='qc_cloud')
        qc_inverter_eff = st.number_input('Inverter Efficiency', min_value=0.5, max_value=1.0, value=0.97, step=0.01, key='qc_inv_eff')

    qc_submitted = st.button('Calculate', width='stretch')

    if qc_submitted:
        timestamp = datetime.combine(qc_date, qc_time)
        hour = timestamp.hour + timestamp.minute / 60.0
        day_of_year = timestamp.timetuple().tm_yday
        wind_for_physics = qc_wind if qc_use_wind else None
        wind_for_model = qc_wind if qc_use_wind else 0.0

        module_temp = estimate_module_temperature(qc_ambient, qc_irradiation, wind_for_physics)
        irradiance_ratio = qc_irradiation / STC_IRRADIANCE
        dc_per_kwp = calculate_dc_power_per_kwp(qc_irradiation, module_temp)
        ac_per_kwp_theoretical = calculate_ac_power_per_kwp(dc_per_kwp, qc_inverter_eff)

        feature_row = build_feature_row(qc_irradiation, qc_ambient, wind_for_model, qc_cloud, hour, day_of_year)
        combined_kw, predicted_gap = predict_combined_ac_kw(
            model, feature_row, [qc_irradiation], [ac_per_kwp_theoretical], qc_capacity
        )
        final_ac_kw = float(combined_kw[0])
        gap = float(predicted_gap[0])
        combined_per_kwp = final_ac_kw / qc_capacity

        g1, g2 = st.columns(2)
        with g1:
            fig_power = go.Figure(go.Indicator(
                mode='gauge+number',
                value=final_ac_kw,
                title={'text': 'Power'},
                gauge={
                    'axis': {'range': [0, qc_capacity]},
                    'bar': {'color': '#2A5298'},
                    'steps': [
                        {'range': [0, qc_capacity / 3], 'color': '#8fd3fe'},
                        {'range': [qc_capacity / 3, 2 * qc_capacity / 3], 'color': '#4f8fc0'},
                        {'range': [2 * qc_capacity / 3, qc_capacity], 'color': '#f7a8a8'},
                    ],
                },
            ))
            fig_power.update_layout(height=350)
            st.plotly_chart(fig_power, width='stretch')
        with g2:
            fig_temp = go.Figure(go.Indicator(
                mode='gauge+number',
                value=module_temp,
                title={'text': 'Module Temperature (C)'},
                gauge={
                    'axis': {'range': [0, 80]},
                    'bar': {'color': '#b23b3b'},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(90,180,90,0.35)'},
                        {'range': [30, 45], 'color': 'rgba(230,190,80,0.35)'},
                        {'range': [45, 80], 'color': 'rgba(230,90,90,0.35)'},
                    ],
                },
            ))
            fig_temp.update_layout(height=350)
            st.plotly_chart(fig_temp, width='stretch')

        fig_waterfall = go.Figure(go.Waterfall(
            x=['Irradiance ratio', 'Temp. effect', 'Inverter loss', 'ML correction', 'Final (per kWp)'],
            measure=['relative', 'relative', 'relative', 'relative', 'total'],
            y=[
                irradiance_ratio,
                dc_per_kwp - irradiance_ratio,
                ac_per_kwp_theoretical - dc_per_kwp,
                gap,
                combined_per_kwp,
            ],
            increasing={'marker': {'color': '#8fd3fe'}},
            decreasing={'marker': {'color': '#f7a8a8'}},
            totals={'marker': {'color': '#5aa9e6'}},
        ))
        fig_waterfall.update_layout(title='Loss / Correction Breakdown (per kWp)', height=450)
        st.plotly_chart(fig_waterfall, width='stretch')

# ------------------------------------------------------------- How It Works --
with tab_how:
    st.header('How It Works')

    st.subheader('Overview')
    st.write(
        'This app estimates solar plant electricity generation by combining a physics-based model '
        'with a machine-learning correction trained on real plant data. Physics alone misses '
        'real-world losses like soiling and wiring. ML alone would not generalize outside its '
        'training data. Combining them keeps predictions grounded.'
    )

    st.subheader('Methodology')
    st.code(
        'Weather Inputs\n'
        '      |\n'
        'Physics Model -> Module Temperature -> DC Power -> AC Power (theoretical)\n'
        '      |\n'
        'Machine Learning -> predicts the real-world gap vs theoretical\n'
        '      |\n'
        'Final Prediction = Theoretical AC + ML Correction',
        language=None,
    )

    st.subheader('Inputs')
    st.markdown(
        '- **Plant Capacity (kWp)**: required, scales every result to kW.\n'
        '- **Irradiation (W/m2)**: default 1000 (STC).\n'
        '- **Ambient Temperature (C)**: default 25 (STC).\n'
        '- **Wind Speed (m/s)**: optional, falls back to the NOCT formula if blank.\n'
        '- **Cloud Cover (%)**: default 0.0, used by the ML model only.\n'
        '- **Inverter Efficiency**: default 0.97, used for the physics baseline only.\n'
    )

    st.subheader('Model Performance')
    st.markdown('**Feature Importance**')
    importance_df = pd.DataFrame(
        {'feature': FEATURE_COLUMNS, 'importance': model.feature_importances_}
    ).sort_values('importance')
    fig_importance = go.Figure(go.Bar(
        x=importance_df['importance'], y=importance_df['feature'], orientation='h', marker_color='#2C7873'
    ))
    fig_importance.update_layout(xaxis_title='importance', yaxis_title='feature', height=450)
    st.plotly_chart(fig_importance, width='stretch')

    render_stat_cards([
        ('Backtest RMSE (kW)', f"{metadata['backtest_rmse_kw']:.1f}"),
        ('Backtest MAE (kW)', f"{metadata['backtest_mae_kw']:.1f}"),
        ('Backtest R2', f"{metadata['backtest_r2_kw']:.3f}"),
    ])

    with st.expander('Full metadata'):
        st.json(metadata)

    st.subheader('Limitations')
    st.markdown(
        '- Extreme or unusual weather outside the training range.\n'
        '- Equipment faults, soiling, or degradation not captured by the model.\n'
        '- Missing input values reduce accuracy (e.g. no wind data uses the NOCT fallback).\n'
        '- Trained on a limited historical window from one plant/region.\n'
    )
