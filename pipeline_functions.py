"""
pipeline_functions.py

Pure functions extracted from Model.ipynb, with no top-level execution code (no API calls, no data loading, no example runs). This exists so the test suite (test_pipeline.py) can import and test the actual pipeline logic directly, since Model.ipynb itself is a notebook, not an importable module.

If we change a function in the notebook, mirror the change here so the tests stay honest about what our pipeline does.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

# Constants (physicsCalc.py)
STC_IRRADIANCE = 1000
STC_TEMP = 25
DEFAULT_TEMP_COEFF = -0.004
DEFAULT_NOCT = 45
DEFAULT_INVERTER_EFF = 0.97

TRAINING_MONTHS = {5, 6}
PLANT_LATITUDE_DEG = 8.85

features = [
    'IRRADIATION(W/m²)', 'AMBIENT_TEMPERATURE(°C)', 'WIND_SPEED(m/s)', 'CLOUD_COVER(%)',
    'TIME_OF_DAY_SIN', 'TIME_OF_DAY_COS', 'DAY_OF_YEAR_SIN', 'DAY_OF_YEAR_COS',
]


# Physics functions
def estimate_module_temperature(ambient_temp_c, irradiance_wm2, wind_speed_ms=None, noct=DEFAULT_NOCT):
    if wind_speed_ms is not None and not pd.isna(wind_speed_ms):
        u0, u1 = 25.0, 6.84
        delta_t = irradiance_wm2 / (u0 + u1 * wind_speed_ms)
    else:
        delta_t = (noct - 20) * (irradiance_wm2 / 800.0)
    return ambient_temp_c + delta_t


def calculate_dc_power_per_kwp(irradiance_wm2, module_temp_c, temp_coefficient=DEFAULT_TEMP_COEFF):
    if irradiance_wm2 <= 0:
        return 0.0
    irradiance_ratio = irradiance_wm2 / STC_IRRADIANCE
    temp_adjustment = 1 + temp_coefficient * (module_temp_c - STC_TEMP)
    raw = irradiance_ratio * temp_adjustment
    return float(np.clip(raw, 0.0, 1.0))


def calculate_ac_power_per_kwp(dc_power_per_kwp, inverter_efficiency=DEFAULT_INVERTER_EFF):
    return dc_power_per_kwp * inverter_efficiency


def build_feature_row(irradiation, ambient_temp, wind_speed, cloud_cover, hour, day_of_year_val):
    time_f = hour
    return pd.DataFrame([{
        'IRRADIATION(W/m²)': irradiation,
        'AMBIENT_TEMPERATURE(°C)': ambient_temp,
        'WIND_SPEED(m/s)': wind_speed,
        'CLOUD_COVER(%)': cloud_cover,
        'TIME_OF_DAY_SIN': np.sin(2 * np.pi * time_f / 24.0),
        'TIME_OF_DAY_COS': np.cos(2 * np.pi * time_f / 24.0),
        'DAY_OF_YEAR_SIN': np.sin(2 * np.pi * day_of_year_val / 365.25),
        'DAY_OF_YEAR_COS': np.cos(2 * np.pi * day_of_year_val / 365.25),
    }])[features]


# Combine physics + model correction
def predict_combined_ac_per_kwp(model, X_rows, irradiation_values, theoretical_ac_per_kwp_values):
    predicted_gap = model.predict(X_rows)
    combined = np.asarray(theoretical_ac_per_kwp_values) + predicted_gap
    combined = np.where(np.asarray(irradiation_values) <= 0, 0.0, combined)
    combined = np.maximum(combined, 0.0)
    return combined, predicted_gap


def predict_combined_ac_kw(model, X_rows, irradiation_values, theoretical_ac_per_kwp_values, capacity_kwp):
    combined_per_kwp, predicted_gap = predict_combined_ac_per_kwp(
        model, X_rows, irradiation_values, theoretical_ac_per_kwp_values
    )
    return combined_per_kwp * capacity_kwp, predicted_gap


# Cyclical time features
def build_time_features(df):
    df = df.copy()
    hour_dec = df['DATE_TIME'].dt.hour + df['DATE_TIME'].dt.minute / 60.0
    df['TIME_OF_DAY_SIN'] = np.sin(2 * np.pi * hour_dec / 24.0)
    df['TIME_OF_DAY_COS'] = np.cos(2 * np.pi * hour_dec / 24.0)
    doy = df['DATE_TIME'].dt.dayofyear
    df['DAY_OF_YEAR_SIN'] = np.sin(2 * np.pi * doy / 365.25)
    df['DAY_OF_YEAR_COS'] = np.cos(2 * np.pi * doy / 365.25)
    return df


# User input collection and defaulting
def build_user_inputs(location, capacity_kwp, time_range_days,
                       tilt_deg=None, azimuth_deg=None, inverter_efficiency=None,
                       temp_coefficient=None, degradation_pct_per_year=None):
    if location is None or str(location).strip() == '':
        raise ValueError("location is required. There is no reasonable default for where a system is.")
    if capacity_kwp is None:
        raise ValueError("capacity_kwp is required. There is no reasonable default for system size.")
    if time_range_days is None:
        raise ValueError("time_range_days is required. The user must choose an estimate horizon.")

    return {
        'location': {'value': location, 'was_defaulted': False, 'note': 'User provided location.'},
        'capacity_kwp': {'value': capacity_kwp, 'was_defaulted': False, 'note': 'User provided system size.'},
        'time_range_days': {'value': time_range_days, 'was_defaulted': False, 'note': 'User-chosen estimate horizon.'},
        'tilt_deg': ({'value': tilt_deg, 'was_defaulted': False, 'note': 'User provided panel tilt.'}
                     if tilt_deg is not None else
                     {'value': None, 'was_defaulted': True,
                      'note': "Defaults to the location's latitude once geocoded."}),
        'azimuth_deg': ({'value': azimuth_deg, 'was_defaulted': False, 'note': 'User provided panel azimuth.'}
                        if azimuth_deg is not None else
                        {'value': None, 'was_defaulted': True,
                         'note': 'Defaults to true south (0 deg) in the northern hemisphere or true north '
                                 '(180 deg) in the southern hemisphere.'}),
        'inverter_efficiency': ({'value': inverter_efficiency, 'was_defaulted': False,
                                  'note': 'User provided inverter efficiency.'}
                                 if inverter_efficiency is not None else
                                 {'value': DEFAULT_INVERTER_EFF, 'was_defaulted': True,
                                  'note': f'Defaults to a typical inverter efficiency of {DEFAULT_INVERTER_EFF*100:.0f}%.'}),
        'temp_coefficient': ({'value': temp_coefficient, 'was_defaulted': False,
                               'note': 'User provided temperature coefficient.'}
                              if temp_coefficient is not None else
                              {'value': DEFAULT_TEMP_COEFF, 'was_defaulted': True,
                               'note': 'Defaults to a typical crystalline silicon value (~-0.4%/deg C).'}),
        'degradation_pct_per_year': ({'value': degradation_pct_per_year, 'was_defaulted': False,
                                       'note': 'User provided degradation rate.'}
                                      if degradation_pct_per_year is not None else
                                      {'value': 0.0, 'was_defaulted': True,
                                       'note': 'Defaults to treating the system as new, with no degradation applied.'}),
    }


# Confidence label and out of range check
def compute_confidence_label(user_inputs):
    optional_fields = ['tilt_deg', 'azimuth_deg', 'inverter_efficiency', 'temp_coefficient', 'degradation_pct_per_year']
    n_defaulted = sum(user_inputs[f]['was_defaulted'] for f in optional_fields)
    n_total = len(optional_fields)
    n_provided = n_total - n_defaulted
    if n_defaulted == 0:
        label = 'High confidence -- all optional fields were provided directly.'
    elif n_defaulted <= n_total // 2:
        label = 'Moderate confidence -- most optional fields were provided, a few defaulted.'
    else:
        label = 'Lower confidence -- most optional fields are using typical defaults, not your actual system specs.'
    return label, n_provided, n_total


def check_out_of_range(lat, requested_months):
    lat_gap = abs(lat - PLANT_LATITUDE_DEG)
    season_mismatch = not set(requested_months).issubset(TRAINING_MONTHS)
    reasons = []
    if lat_gap > 15:
        reasons.append(f"requested latitude is {lat_gap:.1f} deg from the training plant's latitude")
    if season_mismatch:
        reasons.append("requested period falls outside the May-Jun training window")
    return (len(reasons) > 0), reasons


# Capacity derivation formula, isolated as a function for testing
def derive_capacity_kwp(actual_ac_power_kw, theoretical_ac_per_kwp, threshold=0.5):
    actual_ac_power_kw = pd.Series(actual_ac_power_kw)
    theoretical_ac_per_kwp = pd.Series(theoretical_ac_per_kwp)
    mask = theoretical_ac_per_kwp > threshold
    if mask.sum() == 0:
        raise ValueError("No rows exceed the daytime threshold; cannot derive a stable capacity.")
    return (actual_ac_power_kw[mask] / theoretical_ac_per_kwp[mask]).median()


# ---- Weather API integration (Open-Meteo) ----
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,wind_speed_10m,cloud_cover,shortwave_radiation"

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

SEASON_MONTHS = {
    'Spring (Mar-May)': (3, 5),
    'Summer (Jun-Aug)': (6, 8),
    'Fall (Sep-Nov)': (9, 11),
    'Winter (Dec-Feb)': (12, 2),
}


def geocode_location(location_str):
    resp = requests.get(GEOCODING_URL, params={'name': location_str, 'count': 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get('results')
    if not results:
        raise ValueError(f"Could not geocode location: {location_str!r}")
    top = results[0]
    return top['latitude'], top['longitude'], top.get('timezone', 'auto')


def _hourly_json_to_weather_df(payload):
    hourly = payload['hourly']
    return pd.DataFrame({
        'DATE_TIME': pd.to_datetime(hourly['time']),
        'IRRADIATION(W/m²)': hourly['shortwave_radiation'],
        'AMBIENT_TEMPERATURE(°C)': hourly['temperature_2m'],
        'WIND_SPEED(m/s)': hourly['wind_speed_10m'],
        'CLOUD_COVER(%)': hourly['cloud_cover'],
    })


def fetch_archive(lat, lon, start_date, end_date, timezone='auto'):
    params = {
        'latitude': lat, 'longitude': lon, 'start_date': start_date, 'end_date': end_date,
        'hourly': HOURLY_VARS, 'wind_speed_unit': 'ms', 'timezone': timezone,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _hourly_json_to_weather_df(resp.json())


def _end_of_month(year, month):
    next_month = month % 12 + 1
    next_year = year + 1 if next_month == 1 else year
    return date(next_year, next_month, 1) - timedelta(days=1)


def next_month_window(month_number, today=None):
    """(start, end) for the next upcoming occurrence of the given month (1-12)."""
    today = today or date.today()
    year = today.year if month_number > today.month else today.year + 1
    if month_number == today.month:
        year = today.year
    start = date(year, month_number, 1)
    return start, _end_of_month(year, month_number)


def next_season_window(season_label, today=None):
    """(start, end) for the next upcoming occurrence of a 3-month season."""
    start_month, end_month = SEASON_MONTHS[season_label]
    today = today or date.today()
    year = today.year if start_month >= today.month else today.year + 1
    start = date(year, start_month, 1)
    end_year = year + 1 if end_month < start_month else year
    return start, _end_of_month(end_year, end_month)


def historical_windows(start_date, end_date, historical_years=10):
    """Maps each of the past `historical_years` years to the same calendar (start, end) window."""
    windows = {}
    for years_back in range(1, historical_years + 1):
        try:
            hist_start = start_date.replace(year=start_date.year - years_back)
        except ValueError:  # Feb 29 in a non-leap year
            hist_start = start_date.replace(year=start_date.year - years_back, day=28)
        try:
            hist_end = end_date.replace(year=end_date.year - years_back)
        except ValueError:
            hist_end = end_date.replace(year=end_date.year - years_back, day=28)
        windows[hist_start.year] = (hist_start, hist_end)
    return windows


def run_forecast_pipeline(weather_df, capacity_kwp, model, inverter_efficiency=DEFAULT_INVERTER_EFF,
                           temp_coefficient=DEFAULT_TEMP_COEFF):
    """Runs the physics baseline + ML gap correction over a weather dataframe, scaled to capacity_kwp.

    Adds MODULE_TEMPERATURE(°C), DC_POWER_KW (physics only), THEORETICAL_AC_KW (physics only),
    and FINAL_AC_KW (physics + ML correction) columns.
    """
    df = build_time_features(weather_df)

    module_temp = np.array([
        estimate_module_temperature(t, irr, w)
        for t, irr, w in zip(df['AMBIENT_TEMPERATURE(°C)'], df['IRRADIATION(W/m²)'], df['WIND_SPEED(m/s)'])
    ])
    dc_per_kwp = np.array([
        calculate_dc_power_per_kwp(irr, mt, temp_coefficient)
        for irr, mt in zip(df['IRRADIATION(W/m²)'], module_temp)
    ])
    theoretical_ac_per_kwp = calculate_ac_power_per_kwp(dc_per_kwp, inverter_efficiency)

    combined_ac_per_kwp, predicted_gap = predict_combined_ac_per_kwp(
        model, df[features], df['IRRADIATION(W/m²)'].values, theoretical_ac_per_kwp
    )

    df['MODULE_TEMPERATURE(°C)'] = module_temp
    df['DC_PER_KWP'] = dc_per_kwp
    df['THEORETICAL_AC_PER_KWP'] = theoretical_ac_per_kwp
    df['MODEL_GAP_PER_KWP'] = predicted_gap
    df['COMBINED_AC_PER_KWP'] = combined_ac_per_kwp

    df['DC_POWER_KW'] = dc_per_kwp * capacity_kwp
    df['THEORETICAL_AC_KW'] = theoretical_ac_per_kwp * capacity_kwp
    df['FINAL_AC_KW'] = combined_ac_per_kwp * capacity_kwp
    return df
