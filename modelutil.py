import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "solar_gap_model.joblib"
METADATA_PATH = "solar_gap_model_metadata.json"

DEFAULT_FEATURES = [
    "IRRADIATION(W/m²)", "AMBIENT_TEMPERATURE(°C)", "WIND_SPEED(m/s)", "CLOUD_COVER(%)",
    "TIME_OF_DAY_SIN", "TIME_OF_DAY_COS", "DAY_OF_YEAR_SIN", "DAY_OF_YEAR_COS",
]

STAT_CARD_COLORS = ['#8CC63F', '#173F2E', '#2C7873', '#4A4A4A']


def inject_theme_css():
    st.markdown(
        """
        <style>
        .app-banner {
            background: linear-gradient(135deg, #173F2E 0%, #1F5C3F 100%);
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 18px;
        }
        .app-banner h1 { color: white; margin: 0; font-size: 2.4rem; }
        .app-banner p { color: #C9E6B8; margin: 6px 0 0 0; font-size: 1rem; }
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
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_features(metadata):
    if metadata and "feature_order" in metadata:
        return metadata["feature_order"]
    return DEFAULT_FEATURES


def build_row(irradiation, temp, wind, cloud, dt, features):
    hour = dt.hour + dt.minute / 60.0
    day = dt.timetuple().tm_yday
    row = {
        "IRRADIATION(W/m²)": irradiation,
        "AMBIENT_TEMPERATURE(°C)": temp,
        "WIND_SPEED(m/s)": wind,
        "CLOUD_COVER(%)": cloud,
        "TIME_OF_DAY_SIN": np.sin(2 * np.pi * hour / 24),
        "TIME_OF_DAY_COS": np.cos(2 * np.pi * hour / 24),
        "DAY_OF_YEAR_SIN": np.sin(2 * np.pi * day / 365.25),
        "DAY_OF_YEAR_COS": np.cos(2 * np.pi * day / 365.25),
    }
    return pd.DataFrame([row])[features]


def build_rows(df, features, dt_col="DATE_TIME"):
    df = df.copy()
    hour = df[dt_col].dt.hour + df[dt_col].dt.minute / 60.0
    df["TIME_OF_DAY_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["TIME_OF_DAY_COS"] = np.cos(2 * np.pi * hour / 24)
    day = df[dt_col].dt.dayofyear
    df["DAY_OF_YEAR_SIN"] = np.sin(2 * np.pi * day / 365.25)
    df["DAY_OF_YEAR_COS"] = np.cos(2 * np.pi * day / 365.25)
    return df[features]


def combine(model, X, irradiation, theoretical_ac_per_kwp, capacity_kwp):
    gap = model.predict(X)
    combined = np.asarray(theoretical_ac_per_kwp) + gap
    combined = np.where(np.asarray(irradiation) <= 0, 0.0, combined)
    combined = np.maximum(combined, 0.0)
    combined_kw = combined * capacity_kwp
    return combined, combined_kw, gap
