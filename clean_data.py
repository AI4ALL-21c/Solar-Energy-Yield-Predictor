"""Data cleaning pipeline for SOLAR_PLANT_DATA(GENERATION_AND_WEATHER).csv.

Fixes, in dependency order:
  1. Wind speed unit bug: column is labeled m/s but values are km/h
     (mean 15.78 "m/s" = 35 mph sustained monsoon wind -> implausible;
     divide by 3.6 to get real m/s).
  2. Insert the 5 missing 15-minute timestamps and time-interpolate the
     3 trailing nulls (last 3 rows, WIND_SPEED/CLOUD_COVER only).
  3. Recompute THEORETICAL_MODULE_TEMPERATURE using corrected wind
     (Faiman model — same formula as physicsCalc.py, vectorized).
  4. Recompute THEORETICAL_DC_PER_KWP / THEORETICAL_AC_PER_KWP from the
     corrected module temperature (formulas already matched
     physicsCalc.py and the existing INVERTOR_EFFICIENCY relationship
     before this fix — only their *inputs* were wrong).
  5. Scale theoretical power to real kW using PLANT_CAPACITY_KWP
     (previously these columns were left as bare per-kWp ratios,
     never multiplied by capacity).
  6. Recompute DC_RESIDUAL(kW) / AC_RESIDUAL(kW) = actual - theoretical
     now that theoretical is correctly scaled. Before this fix the
     residual was silently ~equal to actual power itself.
  7. Drop CALCULATED_PLANT_CAPACITY(kWp) (was a hardcoded constant 1000,
     contradicted PLANT_CAPACITY_KWP in constants.py) — capacity now
     lives in exactly one place.

DC_EFFICIENCY / AC_EFFICIENCY are intentionally left as-is (actual power
divided by the dataset's own observed peak, not true capacity) — that
definition is outside this fix's scope; only recomputed here for the
newly-interpolated rows so the column stays internally consistent.
"""
import numpy as np
import pandas as pd

from constants import PLANT_CAPACITY_KWP

SRC = 'SOLAR_PLANT_DATA(GENERATION_AND_WEATHER).csv'
OUT = 'solar_plant_data_cleaned.csv'

# Physics constants — must match physicsCalc.py exactly.
FAIMAN_U0, FAIMAN_U1 = 25.0, 6.84
STC_IRRADIANCE, STC_TEMP = 1000.0, 25.0
TEMP_COEFFICIENT = -0.004

WIND_KMH_TO_MS = 3.6


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    return df.sort_values('DATE_TIME').reset_index(drop=True)


def fix_wind_units(df: pd.DataFrame) -> pd.DataFrame:
    df['WIND_SPEED(m/s)'] = df['WIND_SPEED(m/s)'] / WIND_KMH_TO_MS
    return df


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df['DATE_TIME'].min(), df['DATE_TIME'].max(), freq='15min')
    df = df.set_index('DATE_TIME').reindex(full_index)
    df.index.name = 'DATE_TIME'

    numeric_cols = [c for c in df.columns if c != 'PLANT_ID']
    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both')
    df['PLANT_ID'] = df['PLANT_ID'].ffill().bfill().astype(int)

    return df.reset_index()


def recompute_theoretical(df: pd.DataFrame) -> pd.DataFrame:
    irr = df['IRRADIATION(W/m²)']
    wind = df['WIND_SPEED(m/s)']
    ambient = df['AMBIENT_TEMPERATURE(°C)']

    mod_temp = ambient + irr / (FAIMAN_U0 + FAIMAN_U1 * wind)
    df['THEORETICAL_MODULE_TEMPERATURE(°C)'] = mod_temp

    irr_ratio = irr / STC_IRRADIANCE
    temp_adj = 1 + TEMP_COEFFICIENT * (mod_temp - STC_TEMP)
    dc_per_kwp = np.where(irr <= 0, 0.0, np.maximum(irr_ratio * temp_adj, 0.0))
    df['THEORETICAL_DC_PER_KWP'] = dc_per_kwp

    actual_dc = df['ACTUAL_DC_POWER(kW)']
    actual_ac = df['ACTUAL_AC_POWER(kW)']
    inverter_eff = np.where(actual_dc > 0, actual_ac / actual_dc, 0.0)
    df['INVERTOR_EFFICIENCY'] = inverter_eff
    df['THEORETICAL_AC_PER_KWP'] = dc_per_kwp * inverter_eff

    df['THEORETICAL_DC_POWER(kW)'] = df['THEORETICAL_DC_PER_KWP'] * PLANT_CAPACITY_KWP
    df['THEORETICAL_AC_POWER(kW)'] = df['THEORETICAL_AC_PER_KWP'] * PLANT_CAPACITY_KWP

    df['DC_RESIDUAL(kW)'] = actual_dc - df['THEORETICAL_DC_POWER(kW)']
    df['AC_RESIDUAL(kW)'] = actual_ac - df['THEORETICAL_AC_POWER(kW)']

    return df


def recompute_efficiency_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve the existing (non-capacity) definition: actual power over the
    # dataset's own observed peak. Only refreshed so the newly-filled rows
    # are consistent with the rest of the column.
    dc_peak = df['ACTUAL_DC_POWER(kW)'].max()
    ac_peak = df['ACTUAL_AC_POWER(kW)'].max()
    df['DC_EFFICIENCY'] = df['ACTUAL_DC_POWER(kW)'] / dc_peak
    df['AC_EFFICIENCY'] = df['ACTUAL_AC_POWER(kW)'] / ac_peak
    return df


def drop_conflicting_capacity_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['CALCULATED_PLANT_CAPACITY(kWp)'])


def main():
    df = load_raw()
    df = fix_wind_units(df)
    df = fill_gaps(df)
    df = recompute_theoretical(df)
    df = recompute_efficiency_columns(df)
    df = drop_conflicting_capacity_column(df)
    df.to_csv(OUT, index=False)
    print(f'Wrote {OUT}: {len(df)} rows, {len(df.columns)} columns.')


if __name__ == '__main__':
    main()
