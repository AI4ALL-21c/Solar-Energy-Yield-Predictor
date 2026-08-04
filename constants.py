"""Single source of truth for plant constants.

Import PLANT_CAPACITY_KWP wherever a capacity value is needed (data cleaning,
model training/metadata, and the Streamlit app) instead of hardcoding or
re-deriving it. Do not duplicate this number elsewhere.

Derivation: theoretical per-kWp output peaks at 1.0425 (see
THEORETICAL_DC_PER_KWP in the cleaned dataset). Actual DC power peaks at
26,630.5 kW. A real system cannot exceed its nameplate capacity once
temperature/irradiance derating is already accounted for in the per-kWp
figure, so capacity = peak_actual_kW / peak_theoretical_per_kWp:

    26630.5 / 1.0425 = 25,545 kWp  (ratio 1.042x, consistent with the
    theoretical ceiling)

The two other candidates considered and ruled out:
  - 1,000 kWp   -> peak/capacity = 26.6x   (physically impossible)
  - 20,230 kWp  -> peak/capacity = 1.32x   (implausible overshoot)
"""

PLANT_CAPACITY_KWP = 25545
