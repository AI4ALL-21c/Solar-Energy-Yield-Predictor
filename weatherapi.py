from datetime import date
import requests
import pandas as pd
import streamlit as st

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,wind_speed_10m,cloud_cover,global_tilted_irradiance"


@st.cache_data(ttl=3600)
def geocode(location):
    resp = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Location not found: {location}")
    top = results[0]
    return top["latitude"], top["longitude"], top.get("name", location), top.get("timezone", "auto")


def default_tilt_azimuth(lat):
    tilt = round(abs(lat), 1)
    azimuth = 0.0 if lat >= 0 else 180.0
    return tilt, azimuth


def to_df(payload):
    hourly = payload["hourly"]
    return pd.DataFrame({
        "DATE_TIME": pd.to_datetime(hourly["time"]),
        "IRRADIATION(W/m²)": hourly["global_tilted_irradiance"],
        "AMBIENT_TEMPERATURE(°C)": hourly["temperature_2m"],
        "WIND_SPEED(m/s)": hourly["wind_speed_10m"],
        "CLOUD_COVER(%)": hourly["cloud_cover"],
    })


@st.cache_data(ttl=1800)
def fetch_forecast(lat, lon, days, tilt, azimuth, tz="auto"):
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY_VARS, "wind_speed_unit": "ms",
              "tilt": tilt, "azimuth": azimuth, "forecast_days": min(days, 16), "timezone": tz}
    resp = requests.get(FORECAST_URL, params=params, timeout=20)
    resp.raise_for_status()
    return to_df(resp.json())


@st.cache_data(ttl=3600)
def fetch_archive(lat, lon, start, end, tilt, azimuth, tz="auto"):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": HOURLY_VARS, "wind_speed_unit": "ms", "tilt": tilt, "azimuth": azimuth, "timezone": tz}
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return to_df(resp.json())


def fetch_years(lat, lon, tilt, azimuth, start, end, years_back=10, tz="auto"):
    by_year = {}
    current_year = date.today().year
    for yb in range(1, years_back + 1):
        try:
            hist_start = start.replace(year=start.year - yb)
            hist_end = end.replace(year=end.year - yb)
        except ValueError:
            hist_start = start.replace(year=start.year - yb, day=28)
            hist_end = end.replace(year=end.year - yb, day=28)
        try:
            df = fetch_archive(lat, lon, hist_start.isoformat(), hist_end.isoformat(), tilt, azimuth, tz)
            if not df.empty and df["IRRADIATION(W/m²)"].notna().any():
                by_year[current_year - yb] = df
        except Exception:
            continue
    return by_year