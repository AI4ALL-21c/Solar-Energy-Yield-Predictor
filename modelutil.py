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

# Where/when the ML correction model was actually trained and validated -- see check_out_of_range().
PLANT_LATITUDE_DEG = 8.85
TRAINING_MONTHS = {5, 6}

STAT_CARD_COLORS = ['#8CC63F', '#173F2E', '#2C7873', '#4A4A4A']


def check_out_of_range(lat, requested_months):
    """Flags requests far from the plant/season the ML correction was trained on.

    The physics baseline generalizes to any location or season; this only concerns
    whether the *learned correction* on top of it has been validated for these conditions.
    """
    lat_gap = abs(lat - PLANT_LATITUDE_DEG)
    season_mismatch = not set(requested_months).issubset(TRAINING_MONTHS)
    reasons = []
    if lat_gap > 15:
        reasons.append(f"requested latitude is {lat_gap:.1f} deg from the training plant's latitude")
    if season_mismatch:
        reasons.append("requested period falls outside the May-Jun training window")
    return (len(reasons) > 0), reasons

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


def inject_theme_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #F7F9F5; }
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
            box-shadow: 0 4px 16px rgba(23,63,46,0.2);
        }
        .app-banner h1 { color: white; margin: 0; font-size: 2.4rem; }
        .app-banner p { color: #C9E6B8; margin: 6px 0 0 0; font-size: 1rem; }
        .app-banner-art { flex-shrink: 0; }
        @media (max-width: 900px) { .app-banner-art { display: none; } }
        .stat-card {
            border-radius: 10px; padding: 18px 14px; text-align: center; color: white; margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }
        .stat-card .stat-value { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
        .stat-card .stat-label {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
            opacity: 0.9; margin-top: 4px;
        }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        /* Panel containers: any st.container(border=True, key="panel_...") */
        div[class*="st-key-panel_"] {
            background: white;
            border: 1px solid #E3EBDD !important;
            border-radius: 14px !important;
            padding: 6px 14px 18px 14px !important;
            box-shadow: 0 2px 10px rgba(23,63,46,0.06);
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_banner(title, subtitle):
    st.markdown(
        '<div class="app-banner">'
        f'<div><h1>{title}</h1><p>{subtitle}</p></div>'
        f'<div class="app-banner-art">{SOLAR_PANEL_SVG}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def panel(key):
    """A styled bordered container -- use as `with mu.panel("unique_key"):`."""
    return st.container(border=True, key=f'panel_{key}')


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
