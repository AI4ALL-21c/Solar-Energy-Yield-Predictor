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