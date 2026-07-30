import json
import pathlib

import joblib
import pandas as pd
import streamlit as st

from constants import PLANT_CAPACITY_KWP
from pipeline_functions import (
    build_feature_row,
    build_time_features,
    calculate_ac_power_per_kwp,
    calculate_dc_power_per_kwp,
    check_out_of_range,
    estimate_module_temperature,
    features as FEATURE_COLUMNS,
    predict_combined_ac_kw,
)

APP_DIR = pathlib.Path(__file__).parent
DATA_PATH = APP_DIR / 'solar_plant_data_cleaned.csv'
MODEL_PATH = APP_DIR / 'solar_gap_model.joblib'
METADATA_PATH = APP_DIR / 'solar_gap_model_metadata.json'

st.set_page_config(
    page_title='Solar Yield Predictor',
    page_icon='☀️',
    layout='wide',
)

st.title('☀️ Solar Energy Yield Predictor')
st.write(
    'Enter weather and time inputs to get a DC power estimate from the trained '
    'XGBoost gap model. The model predicts the residual gap between a physics-based '
    'theoretical baseline and actual plant output, then combines the two.'
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
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df = build_time_features(df)
    df['ACTUAL_AC_PER_KWP'] = df['ACTUAL_AC_POWER(kW)'] / PLANT_CAPACITY_KWP
    return df


model = load_model()
metadata = load_metadata()
df = load_data()

split_idx = max(int(len(df) * 0.8), 1)
holdout = df.iloc[split_idx:]
holdout_combined_kw, _ = predict_combined_ac_kw(
    model,
    holdout[FEATURE_COLUMNS],
    holdout['IRRADIATION(W/m²)'],
    holdout['THEORETICAL_AC_PER_KWP'],
    PLANT_CAPACITY_KWP,
)
test_preview = pd.DataFrame(
    {
        'Actual': holdout['ACTUAL_AC_POWER(kW)'].reset_index(drop=True),
        'Predicted': pd.Series(holdout_combined_kw),
    }
)

left, right = st.columns([1.05, 1])

with left:
    st.subheader('Model Summary')
    st.metric('Active model', 'XGBoost (gap model)')
    st.metric('Backtest R² (AC kW)', f"{metadata['backtest_r2_kw']:.3f}")
    st.metric('Backtest RMSE', f"{metadata['backtest_rmse_kw']:.2f} kW")
    st.metric('Backtest MAE', f"{metadata['backtest_mae_kw']:.2f} kW")
    st.caption(
        'Official metrics from the retrained gap model: '
        f"Linear Regression ≈ 0.696 R², Random Forest ≈ 0.795 R², "
        f"XGBoost ≈ 0.963 R² (see README/deployment_notes)."
    )

    st.subheader('Make a Prediction')
    with st.form('prediction_form'):
        date_value = st.date_input('Date', value=pd.Timestamp(df['DATE_TIME'].iloc[-1]).date())
        time_value = st.time_input('Time', value=pd.Timestamp(df['DATE_TIME'].iloc[-1]).time())
        irradiation = st.number_input('Irradiation (W/m²)', min_value=0.0, value=800.0, step=10.0)
        ambient_temp = st.number_input('Ambient temperature (°C)', value=30.0, step=0.5)
        wind_speed = st.number_input('Wind speed (m/s)', min_value=0.0, value=2.0, step=0.5)
        cloud_cover = st.number_input('Cloud cover (%)', min_value=0.0, max_value=100.0, value=40.0, step=5.0)
        submitted = st.form_submit_button('Predict AC Power')

    if submitted:
        timestamp = pd.Timestamp.combine(date_value, time_value)
        hour = timestamp.hour + timestamp.minute / 60.0
        day_of_year = timestamp.dayofyear

        module_temp = estimate_module_temperature(ambient_temp, irradiation, wind_speed)
        theoretical_dc_per_kwp = calculate_dc_power_per_kwp(irradiation, module_temp)
        theoretical_ac_per_kwp = calculate_ac_power_per_kwp(theoretical_dc_per_kwp)

        feature_row = build_feature_row(irradiation, ambient_temp, wind_speed, cloud_cover, hour, day_of_year)
        combined_kw, predicted_gap = predict_combined_ac_kw(
            model, feature_row, [irradiation], [theoretical_ac_per_kwp], PLANT_CAPACITY_KWP
        )
        st.success(f'Estimated AC power: {combined_kw[0]:.2f} kW')
        st.caption(
            f'Theoretical baseline: {theoretical_ac_per_kwp * PLANT_CAPACITY_KWP:.2f} kW, '
            f'model-predicted gap: {predicted_gap[0]:+.4f} per kWp, '
            f'estimated module temperature: {module_temp:.1f} °C'
        )

        out_of_range, reasons = check_out_of_range(lat=8.85, requested_months={timestamp.month})
        if out_of_range:
            st.warning(
                'This input falls outside the training conditions, so treat the estimate with caution: '
                + '; '.join(reasons)
            )

        st.dataframe(feature_row, width='stretch')

with right:
    st.subheader('Quick Check on Holdout Data')
    st.line_chart(test_preview.head(200), x='Actual', y='Predicted')
    st.dataframe(test_preview.head(10), width='stretch')

st.divider()

st.subheader('Feature Importance')
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame(
        {
            'Feature': FEATURE_COLUMNS,
            'Importance': model.feature_importances_,
        }
    ).sort_values('Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))
else:
    st.info('Feature importance is not available for this estimator.')

