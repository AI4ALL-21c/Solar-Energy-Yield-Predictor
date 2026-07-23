import pathlib

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = pathlib.Path(__file__).with_name('SOLAR_PLANT_DATA(GENERATION_AND_WEATHER).csv')
PREFERRED_MODEL_NAME = 'XGBoost'

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    XGBRegressor = None

st.set_page_config(
    page_title='Solar Yield Predictor',
    page_icon='☀️',
    layout='wide',
)

st.title('☀️ Solar Energy Yield Predictor')
st.write(
    'Interact with the trained model using weather and time inputs. '
    'The notebook evaluation favored XGBoost, and this app will use it when available; '
    'otherwise it falls back to a Random Forest so the demo still runs.'
)

FEATURE_COLUMNS = [
    'IRRADIATION(W/m²)',
    'AMBIENT_TEMPERATURE(°C)',
    'MODULE_TEMPERATURE(°C)',
    'hour',
    'month',
    'dayofyear',
]
TARGET_COLUMN = 'DC_POWER(kW)'


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    df = df.dropna(subset=['DATE_TIME', TARGET_COLUMN]).copy()
    df = df.sort_values('DATE_TIME').reset_index(drop=True)
    df['hour'] = df['DATE_TIME'].dt.hour + (df['DATE_TIME'].dt.minute / 60.0)
    df['month'] = df['DATE_TIME'].dt.month
    df['dayofyear'] = df['DATE_TIME'].dt.dayofyear
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    split_idx = max(int(len(df) * 0.8), 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        active_model_name = 'XGBoost'
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        active_model_name = 'Random Forest'

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        'r2': float(r2_score(y_test, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'mae': float(mean_absolute_error(y_test, predictions)),
    }

    test_preview = pd.DataFrame(
        {
            'Actual': y_test.reset_index(drop=True),
            'Predicted': pd.Series(predictions),
        }
    )

    return model, metrics, test_preview, active_model_name, df


@st.cache_data
def make_prediction_frame(date_value, time_value, irradiation, ambient_temp, module_temp):
    timestamp = pd.Timestamp.combine(date_value, time_value)
    return pd.DataFrame(
        {
            'IRRADIATION(W/m²)': [irradiation],
            'AMBIENT_TEMPERATURE(°C)': [ambient_temp],
            'MODULE_TEMPERATURE(°C)': [module_temp],
            'hour': [timestamp.hour + (timestamp.minute / 60.0)],
            'month': [timestamp.month],
            'dayofyear': [timestamp.dayofyear],
        }
    )


model, metrics, test_preview, active_model_name, df = train_model()

left, right = st.columns([1.05, 1])

with left:
    st.subheader('Model Summary')
    st.metric('Active model', active_model_name)
    st.metric('Test R²', f"{metrics['r2']:.3f}")
    st.metric('Test RMSE', f"{metrics['rmse']:.2f} kW")
    st.metric('Test MAE', f"{metrics['mae']:.2f} kW")
    st.caption(
        'Notebook evaluation used to choose the model: '
        'Linear Regression ≈ 0.696 R², Random Forest ≈ 0.795 R², '
        'XGBoost ≈ 0.963 R².'
    )

    st.subheader('Make a Prediction')
    with st.form('prediction_form'):
        date_value = st.date_input('Date', value=pd.Timestamp(df['DATE_TIME'].iloc[-1]).date())
        time_value = st.time_input('Time', value=pd.Timestamp(df['DATE_TIME'].iloc[-1]).time())
        irradiation = st.number_input('Irradiation (W/m²)', min_value=0.0, value=800.0, step=10.0)
        ambient_temp = st.number_input('Ambient temperature (°C)', value=30.0, step=0.5)
        module_temp = st.number_input('Module temperature (°C)', value=35.0, step=0.5)
        submitted = st.form_submit_button('Predict DC Power')

    if submitted:
        input_frame = make_prediction_frame(
            date_value,
            time_value,
            irradiation,
            ambient_temp,
            module_temp,
        )
        prediction = float(model.predict(input_frame)[0])
        st.success(f'Estimated DC power: {prediction:.2f} kW')
        st.dataframe(input_frame, width='stretch')

with right:
    st.subheader('Quick Check on Holdout Data')
    st.line_chart(test_preview.head(100), x='Actual', y='Predicted')
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
