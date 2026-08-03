import pandas as pd
import numpy as np



IRRADIATION_COL = 'IRRADIATION(W/m²)'  # confirm against solar_data.columns if unsure

# ---- Typical defaults ----
STC_IRRADIANCE = 1000          # W/m^2, standard test condition irradiance
STC_TEMP = 25                  # deg C, standard test condition module temp
DEFAULT_TEMP_COEFF = -0.004    # per deg C, typical crystalline-silicon panel (~-0.4%/C)
DEFAULT_NOCT = 45              # deg C, typical nominal operating cell temp
DEFAULT_INVERTER_EFF = 0.97    # typical inverter efficiency
DEFAULT_MODULE_EFFICIENCY = 0.18  # used to back out an assumed capacity from area


def estimate_module_temperature(ambient_temp_c, irradiance_wm2, wind_speed_ms=None, noct=DEFAULT_NOCT):
    noct_delta = (noct - 20) * (irradiance_wm2 / 800.0)
    if wind_speed_ms is None:
        return ambient_temp_c + noct_delta
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)
    wind_delta = irradiance_wm2 / (25.0 + 6.84 * wind_speed_ms)
    delta_t = np.where(np.isnan(wind_speed_ms), noct_delta, wind_delta)
    return ambient_temp_c + delta_t


def calculate_dc_power_per_kwp(irradiance_wm2, module_temp_c, temp_coefficient=DEFAULT_TEMP_COEFF):
    irradiance_ratio = irradiance_wm2 / STC_IRRADIANCE
    temp_adjustment = 1 + temp_coefficient * (module_temp_c - STC_TEMP)
    power = irradiance_ratio * temp_adjustment

    return np.where(
        irradiance_wm2 <= 0,
        0.0,
        np.maximum(power, 0.0)
    )


def calculate_ac_power_per_kwp(dc_power_per_kwp, inverter_efficiency=DEFAULT_INVERTER_EFF):
    return dc_power_per_kwp * inverter_efficiency


# ---- NEW: Master Wrapper to calculate both per-kWp ratios and absolute kW ----

def calculate_theoretical_system_outputs(
    ambient_temp_c,
    irradiance_wm2,
    wind_speed_ms=None,
    effective_area_m2=None,
    module_efficiency=None,
    inverter_efficiency=None,
    temp_coefficient=DEFAULT_TEMP_COEFF,
    noct=DEFAULT_NOCT,
):
    """
    Computes all pipeline metrics. Handles custom overrides and falls back 
    to standard constants if inputs are not supplied.
    """
    # 1. Fallback matching logic for rates
    mod_eff = module_efficiency if module_efficiency is not None else DEFAULT_MODULE_EFFICIENCY
    inv_eff = inverter_efficiency if inverter_efficiency is not None else DEFAULT_INVERTER_EFF
    
    # 2. Calculate Module Temperature
    mod_temp = estimate_module_temperature(ambient_temp_c, irradiance_wm2, wind_speed_ms, noct)
    
    # 3. Calculate original fractional performance baselines (per kWp)
    dc_per_kwp = calculate_dc_power_per_kwp(irradiance_wm2, mod_temp, temp_coefficient)
    ac_per_kwp = calculate_ac_power_per_kwp(dc_per_kwp, inv_eff)
    
    # 4. Calculate Target Capacity scale using Area (m²) and efficiency
    if effective_area_m2 is not None:
        target_capacity_kwp = np.where(
        pd.isna(effective_area_m2),
        1.0,
        effective_area_m2 * mod_eff
    )
    else:
         target_capacity_kwp = 1.0


    absolute_dc_power_kw = dc_per_kwp * target_capacity_kwp
    absolute_ac_power_kw = absolute_dc_power_kw * inv_eff

    return (
        mod_temp,
        dc_per_kwp,
        ac_per_kwp,
        target_capacity_kwp,
        absolute_dc_power_kw,
        absolute_ac_power_kw
    )
         

# ---- Simple Tester Block ----
if __name__ == "__main__":
    print("=" * 70)
    print("   SYSTEM TEST: VERIFYING PER_KWP, ABSOLUTES, & ACTUAL BASELINES   ")
    print("=" * 70)
    
    # Inputs
    test_ambient_temp = 30.0   # °C
    test_irradiance = 900.0    # W/m²
    test_wind = 2.5            # m/s
    
    # -------------------------------------------------------------------------
    # SCENARIO 1: TESTING SYSTEM DEFAULTS
    # -------------------------------------------------------------------------
    print("▶ SCENARIO 1: Using Strict System Defaults")
    print(f"  [Inputs] Ambient: {test_ambient_temp}°C | Irradiance: {test_irradiance} W/m²")
    print(f"  [Baselines] Area: None | Mod Eff: {DEFAULT_MODULE_EFFICIENCY*100}% | Inv Eff: {DEFAULT_INVERTER_EFF*100}%")
    
    outputs_def = calculate_theoretical_system_outputs(
        ambient_temp_c=test_ambient_temp,
        irradiance_wm2=test_irradiance,
        wind_speed_ms=test_wind,
        effective_area_m2=None,
        module_efficiency=None,
        inverter_efficiency=None
    )
    
    print(f"  [Returned Outputs]")
    print(f"    - Module Temp:     {outputs_def[0]:.2f} °C")
    print(f"    - DC per kWp:      {outputs_def[1]:.4f}")
    print(f"    - AC per kWp:      {outputs_def[2]:.4f}")
    print(f"    - Target Capacity: {outputs_def[3]:.1f} kWp (Fallback Baseline)")
    print(f"    - Absolute DC:     {outputs_def[4]:.4f} kW")
    print(f"    - Absolute AC:     {outputs_def[5]:.4f} kW")
    print("-" * 70)
    
    # -------------------------------------------------------------------------
    # SCENARIO 2: TESTING CUSTOM ENTRIES (Using Area to scale to thousands)
    # -------------------------------------------------------------------------
    custom_area = 15000.0        # m²
    custom_mod_eff = 0.20        # 20%
    custom_inv_eff = 0.94        # 94%
    
    print("▶ SCENARIO 2: Using Custom Overrides")
    print(f"  [Inputs] Ambient: {test_ambient_temp}°C | Irradiance: {test_irradiance} W/m²")
    print(f"  [Baselines] Area: {custom_area} m² | Mod Eff: {custom_mod_eff*100}% | Inv Eff: {custom_inv_eff*100}%")
    
    outputs_cust = calculate_theoretical_system_outputs(
        ambient_temp_c=test_ambient_temp,
        irradiance_wm2=test_irradiance,
        wind_speed_ms=test_wind,
        effective_area_m2=custom_area,
        module_efficiency=custom_mod_eff,
        inverter_efficiency=custom_inv_eff
    )
    
    print(f"  [Returned Outputs]")
    print(f"    - Module Temp:     {outputs_cust[0]:.2f} °C")
    print(f"    - DC per kWp:      {outputs_cust[1]:.4f}")
    print(f"    - AC per kWp:      {outputs_cust[2]:.4f}")
    print(f"    - Target Capacity: {outputs_cust[3]:.1f} kWp (Calculated Baseline Size)")
    print(f"    - Absolute DC:     {outputs_cust[4]:.2f} kW (Scaled smoothly into thousands!)")
    print(f"    - Absolute AC:     {outputs_cust[5]:.2f} kW")
    print("=" * 70)