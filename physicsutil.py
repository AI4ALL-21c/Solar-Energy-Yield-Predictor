import numpy as np

STC_IRRADIANCE = 1000
STC_TEMP = 25
DEFAULT_TEMP_COEFF = -0.004
DEFAULT_NOCT = 45
DEFAULT_INVERTER_EFF = 0.97
DEFAULT_MODULE_EFFICIENCY = 0.18
DEFAULT_CLOUD_COVER = 0.0


def estimate_module_temp(ambient_temp, irradiance, wind_speed=None, noct=DEFAULT_NOCT):
    irradiance = np.asarray(irradiance, dtype=float)
    noct_delta = (noct - 20) * (irradiance / 800.0)
    if wind_speed is None:
        return ambient_temp + noct_delta
    wind_speed = np.asarray(wind_speed, dtype=float)
    wind_delta = irradiance / (25.0 + 6.84 * wind_speed)
    delta = np.where(np.isnan(wind_speed), noct_delta, wind_delta)
    return ambient_temp + delta


def calc_dc_per_kwp(irradiance, module_temp, temp_coeff=DEFAULT_TEMP_COEFF):
    irradiance = np.asarray(irradiance, dtype=float)
    ratio = irradiance / STC_IRRADIANCE
    adjustment = 1 + temp_coeff * (module_temp - STC_TEMP)
    power = ratio * adjustment
    return np.where(irradiance <= 0, 0.0, np.maximum(power, 0.0))


def calc_ac_per_kwp(dc_per_kwp, inverter_eff=DEFAULT_INVERTER_EFF):
    return np.asarray(dc_per_kwp, dtype=float) * inverter_eff


def run_physics(capacity_kwp, irradiance, ambient_temp, wind_speed=None,
                 inverter_eff=DEFAULT_INVERTER_EFF, temp_coeff=DEFAULT_TEMP_COEFF, noct=DEFAULT_NOCT):
    module_temp = estimate_module_temp(ambient_temp, irradiance, wind_speed, noct)
    dc_per_kwp = calc_dc_per_kwp(irradiance, module_temp, temp_coeff)
    ac_per_kwp = calc_ac_per_kwp(dc_per_kwp, inverter_eff)
    dc_kw = dc_per_kwp * capacity_kwp
    ac_kw = ac_per_kwp * capacity_kwp
    return module_temp, dc_per_kwp, ac_per_kwp, dc_kw, ac_kw