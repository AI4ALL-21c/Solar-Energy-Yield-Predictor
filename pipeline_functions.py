"""
pipeline_functions.py

Pure functions extracted from Model.ipynb, with no top-level execution code (no API calls, no data loading, no example runs). This exists so the test suite (test_pipeline.py) can import and test the actual pipeline logic directly, since Model.ipynb itself is a notebook, not an importable module.

If we change a function in the notebook, mirror the change here so the tests stay honest about what our pipeline does.
"""
import numpy as np
import pandas as pd

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
    return max(irradiance_ratio * temp_adjustment, 0.0)


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
