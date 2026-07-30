"""
This automated test suite verifies the core pipeline logic for Model.ipynb.

By turning manual checks into permanent unit tests, it ensures the model's key physical and logical rules are reverified every time the code changes.

HOW TO RUN:
    pip install pytest
    pytest test_pipeline.py -v
"""
import numpy as np
import pandas as pd
import pytest

from pipeline_functions import (
    estimate_module_temperature,
    calculate_dc_power_per_kwp,
    calculate_ac_power_per_kwp,
    build_feature_row,
    predict_combined_ac_per_kwp,
    predict_combined_ac_kw,
    build_time_features,
    build_user_inputs,
    compute_confidence_label,
    check_out_of_range,
    derive_capacity_kwp,
    DEFAULT_INVERTER_EFF,
    DEFAULT_TEMP_COEFF,
    STC_TEMP,
    PLANT_LATITUDE_DEG,
)

# A fake model to test the combination logic in isolation without needing the real XGBoost model. Controlling the predicted gap lets us verify that predict_combined_ac_per_kwp handles every scenario correctly, including edge cases.

class FakeModel:
    def __init__(self, fixed_gap):
        self.fixed_gap = fixed_gap

    def predict(self, X):
        return np.full(len(X), self.fixed_gap)


# Cyclical Time Features

class TestCyclicalFeatures:

    # verifies that 11:59 PM (23:59) and 12:00 AM (00:00) lie right next to each other
    def test_midnight_and_almost_midnight_are_close(self):
        df = pd.DataFrame({'DATE_TIME': pd.to_datetime(['2020-06-01 23:59:00', '2020-06-02 00:00:00'])})
        out = build_time_features(df)
        sin_diff = abs(out['TIME_OF_DAY_SIN'].iloc[0] - out['TIME_OF_DAY_SIN'].iloc[1])
        cos_diff = abs(out['TIME_OF_DAY_COS'].iloc[0] - out['TIME_OF_DAY_COS'].iloc[1])
        assert sin_diff < 0.01
        assert cos_diff < 0.01

    # asserts the identity of sin^2(theta)+cos^2(theta)=1
    def test_sin_cos_always_on_unit_circle(self):
        """sin^2 + cos^2 must equal 1 for every valid timestamp,  a basic correctness check on the encoding itself."""
        df = pd.DataFrame({'DATE_TIME': pd.date_range('2020-01-01', periods=50, freq='7h')})
        out = build_time_features(df)
        magnitude = out['TIME_OF_DAY_SIN']**2 + out['TIME_OF_DAY_COS']**2
        assert np.allclose(magnitude, 1.0, atol=1e-9)
        doy_magnitude = out['DAY_OF_YEAR_SIN']**2 + out['DAY_OF_YEAR_COS']**2
        assert np.allclose(doy_magnitude, 1.0, atol=1e-9)

    # confirms that dec 31st and jan 1st map smoothly across the year boundary
    def test_dec31_and_jan1_are_close(self):
        df = pd.DataFrame({'DATE_TIME': pd.to_datetime(['2020-12-31 12:00:00', '2020-01-01 12:00:00'])})
        out = build_time_features(df)
        sin_diff = abs(out['DAY_OF_YEAR_SIN'].iloc[0] - out['DAY_OF_YEAR_SIN'].iloc[1])
        assert sin_diff < 0.02


# Capacity Derivation

class TestCapacityDerivation:

    # confirms exact mathematical recovery of capacity
    def test_recovers_known_capacity_from_clean_data(self):
        """If we construct fake data where the true capacity is exactly
        1500 kWp, the derivation formula should recover ~1500."""
        true_capacity = 1500.0
        theoretical = pd.Series([0.6, 0.7, 0.8, 0.9, 0.95] * 20)
        actual_kw = theoretical * true_capacity
        result = derive_capacity_kwp(actual_kw, theoretical)
        assert result == pytest.approx(true_capacity, rel=1e-6)

    # corrupts data points with massive numbers to confirm that the median based calculation ignored sensor outliers
    def test_robust_to_outlier_rows(self):
        true_capacity = 1500.0
        theoretical = pd.Series([0.6, 0.7, 0.8, 0.9, 0.95] * 20)
        actual_kw = theoretical * true_capacity
        actual_kw.iloc[:3] = [50000, 1, 99999]
        result = derive_capacity_kwp(actual_kw, theoretical)
        assert result == pytest.approx(true_capacity, rel=0.05)

    # confirms that low light hours are filtered out to prevent division by near zero denominators
    def test_excludes_low_theoretical_rows(self):
        theoretical = pd.Series([0.01, 0.02, 0.03] + [0.8] * 10)
        actual_kw = pd.Series([500, 500, 500] + [1200] * 10)  # first 3 rows would imply a wildly different capacity
        result = derive_capacity_kwp(actual_kw, theoretical)
        assert result == pytest.approx(1500.0, rel=1e-6)  # 1200/0.8 = 1500, unaffected by the noisy rows

    # ensures and error is thrown if no valid daytime data is available
    def test_raises_if_no_rows_pass_threshold(self):
        theoretical = pd.Series([0.1, 0.2, 0.3])
        actual_kw = pd.Series([10, 20, 30])
        with pytest.raises(ValueError):
            derive_capacity_kwp(actual_kw, theoretical)


# Physics Functions
class TestPhysicsFunctions:

    # confirms no light means 0 kw output
    def test_zero_irradiance_gives_zero_dc_power(self):
        assert calculate_dc_power_per_kwp(0.0, module_temp_c=30) == 0.0

    def test_negative_irradiance_gives_zero_dc_power(self):
        assert calculate_dc_power_per_kwp(-5.0, module_temp_c=30) == 0.0

    # check that a at standard test conditions (1000 W/m^2, 25 degrees Celsius module temp), output per kwp equals exactly 1
    def test_dc_power_at_exactly_stc_conditions(self):
        result = calculate_dc_power_per_kwp(1000.0, module_temp_c=25.0)
        assert result == pytest.approx(1.0)

    # validates thermal loss, hotter panels are less efficient
    def test_hotter_module_produces_less_dc_power(self):
        cooler = calculate_dc_power_per_kwp(800.0, module_temp_c=25.0)
        hotter = calculate_dc_power_per_kwp(800.0, module_temp_c=45.0)
        assert cooler > hotter

    # validates thermal gain, cold panels operate above rated efficiency
    def test_cooler_than_stc_gives_a_efficiency_bonus(self):
        below_stc = calculate_dc_power_per_kwp(1000.0, module_temp_c=15.0)
        assert below_stc > 1.0

    # checks conversion efficiency from dc to ac power
    def test_ac_power_applies_inverter_efficiency(self):
        dc = 0.8
        ac = calculate_ac_power_per_kwp(dc, inverter_efficiency=0.97)
        assert ac == pytest.approx(dc * 0.97)

    def test_ac_power_uses_default_inverter_efficiency(self):
        dc = 0.8
        ac_default = calculate_ac_power_per_kwp(dc)
        assert ac_default == pytest.approx(dc * DEFAULT_INVERTER_EFF)

    def test_module_temp_increases_with_irradiance(self):
        """More sunlight hitting the panel should heat it up more, all else equal."""
        cool_sun = estimate_module_temperature(ambient_temp_c=25, irradiance_wm2=200, wind_speed_ms=2.0)
        hot_sun = estimate_module_temperature(ambient_temp_c=25, irradiance_wm2=900, wind_speed_ms=2.0)
        assert hot_sun > cool_sun

    def test_more_wind_cools_the_module(self):
        """More wind should carry away more heat, lowering module temp relative to the same irradiance with less wind."""
        low_wind = estimate_module_temperature(ambient_temp_c=25, irradiance_wm2=800, wind_speed_ms=0.5)
        high_wind = estimate_module_temperature(ambient_temp_c=25, irradiance_wm2=800, wind_speed_ms=8.0)
        assert high_wind < low_wind

    def test_module_temp_falls_back_to_noct_when_wind_missing(self):
        """When wind speed is unavailable, the NOCT-based fallback formula should be used instead of the wind-based one"""
        result = estimate_module_temperature(ambient_temp_c=25, irradiance_wm2=800, wind_speed_ms=None)
        assert result > 25


class TestExtremeClimates:
    """Tests how the linear physics functions perform under extreme climates (polar cold, desert heat) far outside the tropical training conditions."""

    def test_polar_night_still_forces_zero_regardless_of_extreme_cold(self):
        """Antarctic winter: zero sunlight for weeks at a time, at extreme cold. The zero-irradiation rule must hold no matter how cold it gets."""
        assert calculate_dc_power_per_kwp(0.0, module_temp_c=-60.0) == 0.0

    def test_extreme_wind_pulls_module_temp_toward_ambient(self):
        """High Antarctic winds (up to 40 m/s) increase cooling, causing module temperature to converge toward ambient temperature rather than diverge."""
        ambient = -20.0
        result = estimate_module_temperature(ambient_temp_c=ambient, irradiance_wm2=500, wind_speed_ms=40.0)
        assert abs(result - ambient) < 5.0  # within 5C of ambient, not wildly hotter

    def test_desert_heat_produces_a_large_but_bounded_efficiency_loss(self):
        """Extreme heat (50°C, full sun) significantly reduces output below rated capacity due to high module temperature, but output remains positive and within a reasonable range."""
        mod_temp = estimate_module_temperature(ambient_temp_c=50.0, irradiance_wm2=1000.0, wind_speed_ms=1.0)
        dc = calculate_dc_power_per_kwp(1000.0, mod_temp)
        assert 0.0 <= dc < 1.0  # reduced output due to heat, but not negative or invalid

    def test_KNOWN_LIMITATION_extreme_cold_can_exceed_rated_capacity(self):
        """Known issue: extreme cold and bright sun can cause predictions over 100% capacity because the formula has no cap. This test logs the output until a cap is added."""
        mod_temp = estimate_module_temperature(ambient_temp_c=-30.0, irradiance_wm2=900.0, wind_speed_ms=2.0)
        dc = calculate_dc_power_per_kwp(900.0, mod_temp)
        print(f"\n[KNOWN LIMITATION] -30C/900W/m^2 -> DC_per_kWp={dc:.4f} "
              f"({'EXCEEDS' if dc > 1.0 else 'within'} rated capacity, no cap enforced)")
        assert dc > 0  # sanity check: output remains positive without crashing, but not capped at 1.0.

    def test_antarctic_research_station_latitude_flags_out_of_range(self):
        mcmurdo_lat = -77.85
        is_out, reasons = check_out_of_range(lat=mcmurdo_lat, requested_months={6})
        assert is_out is True
        assert any('latitude' in r for r in reasons)

    def test_northern_canada_latitude_flags_out_of_range(self):
        yellowknife_lat = 62.45
        is_out, reasons = check_out_of_range(lat=yellowknife_lat, requested_months={6})
        assert is_out is True
        assert any('latitude' in r for r in reasons)

    def test_even_temperate_canada_still_flags_on_latitude_alone(self):
        """Verifies the distance check flags non-polar locations (like Toronto) that exceed the 15-degree latitude threshold from the training plant."""
        toronto_lat = 43.65
        is_out, reasons = check_out_of_range(lat=toronto_lat, requested_months={6})
        assert is_out is True


# Combine Physics and Model Correction
class TestCombinationLogic:

    def test_zero_irradiation_forces_zero_output_regardless_of_model(self):
        """Verifies zero sunlight always forces zero output, overriding any non-zero model predictions."""
        model = FakeModel(fixed_gap=0.5)  # model wrongly thinks there's a huge gap
        row = build_feature_row(0.0, 22.0, 3.0, 50.0, hour=2, day_of_year_val=150)
        combined, raw_gap = predict_combined_ac_per_kwp(model, row, irradiation_values=[0.0], theoretical_ac_per_kwp_values=[0.0])
        assert combined[0] == 0.0
        assert raw_gap[0] == 0.5  # confirms the safety override forced 0.0 power despite the model predicting 0.5

    def test_negative_combined_result_gets_floored_to_zero(self):
        """Ensures negative model predictions are floored at 0, preventing impossible negative power output."""
        model = FakeModel(fixed_gap=-5.0)  # absurd negative gap
        row = build_feature_row(500.0, 30.0, 2.0, 20.0, hour=12, day_of_year_val=150)
        combined, _ = predict_combined_ac_per_kwp(model, row, irradiation_values=[500.0], theoretical_ac_per_kwp_values=[0.5])
        assert combined[0] == 0.0

    def test_combined_equals_theoretical_plus_gap_in_the_normal_case(self):
        """Verifies standard behavior: output equals theoretical power plus predicted gap when no edge conditions apply."""
        model = FakeModel(fixed_gap=0.05)
        row = build_feature_row(600.0, 28.0, 2.5, 30.0, hour=12, day_of_year_val=150)
        combined, gap = predict_combined_ac_per_kwp(model, row, irradiation_values=[600.0], theoretical_ac_per_kwp_values=[0.55])
        assert combined[0] == pytest.approx(0.55 + 0.05)

    def test_kw_scaling_is_a_simple_linear_multiplication(self):
        """Verifies output scales linearly: doubling capacity exactly doubles predicted kW."""
        model = FakeModel(fixed_gap=0.05)
        row = build_feature_row(600.0, 28.0, 2.5, 30.0, hour=12, day_of_year_val=150)
        kw_at_1x, _ = predict_combined_ac_kw(model, row, [600.0], [0.55], capacity_kwp=1000.0)
        kw_at_2x, _ = predict_combined_ac_kw(model, row, [600.0], [0.55], capacity_kwp=2000.0)
        assert kw_at_2x[0] == pytest.approx(2 * kw_at_1x[0])

    def test_handles_a_batch_of_rows_not_just_one(self):
        """Verifies the function works on multiple rows at once, like in the live pipeline."""
        model = FakeModel(fixed_gap=0.02)
        irr = [0.0, 300.0, 600.0, 900.0]
        theo = [0.0, 0.25, 0.55, 0.85]
        rows = pd.concat([
            build_feature_row(i, 28.0, 2.5, 30.0, hour=h, day_of_year_val=150)
            for i, h in zip(irr, [0, 8, 12, 15])
        ], ignore_index=True)
        combined, _ = predict_combined_ac_per_kwp(model, rows, irr, theo)
        assert combined[0] == 0.0  # night row still forced to zero
        assert combined[1] == pytest.approx(0.25 + 0.02)
        assert combined[2] == pytest.approx(0.55 + 0.02)
        assert combined[3] == pytest.approx(0.85 + 0.02)


# User Input Collection
class TestUserInputs:

    # confirms required inputs throw errors if missing
    def test_missing_location_raises(self):
        with pytest.raises(ValueError):
            build_user_inputs(location=None, capacity_kwp=5000, time_range_days=5)

    def test_blank_string_location_raises(self):
        with pytest.raises(ValueError):
            build_user_inputs(location="   ", capacity_kwp=5000, time_range_days=5)

    def test_missing_capacity_raises(self):
        with pytest.raises(ValueError):
            build_user_inputs(location="Chennai", capacity_kwp=None, time_range_days=5)

    def test_missing_time_range_raises(self):
        with pytest.raises(ValueError):
            build_user_inputs(location="Chennai", capacity_kwp=5000, time_range_days=None)

    # ensures 0 kwp isn't mistakenly caught as a missing input
    def test_zero_capacity_does_not_incorrectly_raise(self):
        result = build_user_inputs(location="Chennai", capacity_kwp=0, time_range_days=5)
        assert result['capacity_kwp']['value'] == 0
        assert result['capacity_kwp']['was_defaulted'] is False

    # checks that missing optional parameters default correctly and track flags
    def test_optional_fields_default_when_omitted(self):
        result = build_user_inputs(location="Chennai", capacity_kwp=5000, time_range_days=5)
        for field in ['tilt_deg', 'azimuth_deg', 'inverter_efficiency', 'temp_coefficient', 'degradation_pct_per_year']:
            assert result[field]['was_defaulted'] is True

    def test_optional_fields_respected_when_provided(self):
        result = build_user_inputs(
            location="Chennai", capacity_kwp=5000, time_range_days=5,
            tilt_deg=20.0, azimuth_deg=170.0, inverter_efficiency=0.95,
            temp_coefficient=-0.0035, degradation_pct_per_year=0.4,
        )
        assert result['tilt_deg']['value'] == 20.0
        assert result['tilt_deg']['was_defaulted'] is False
        assert result['inverter_efficiency']['value'] == 0.95
        assert result['inverter_efficiency']['was_defaulted'] is False


# Confidence Label and Out of Range
class TestConfidenceAndRange:

    # ensures confidence scores correctly label predictions as high, moderate, or lower based on many parameters were given and defaulted
    def test_all_fields_provided_gives_high_confidence(self):
        inputs = build_user_inputs(
            location="Chennai", capacity_kwp=5000, time_range_days=5,
            tilt_deg=20.0, azimuth_deg=170.0, inverter_efficiency=0.95,
            temp_coefficient=-0.0035, degradation_pct_per_year=0.4,
        )
        label, n_provided, n_total = compute_confidence_label(inputs)
        assert 'High confidence' in label
        assert n_provided == n_total

    def test_all_fields_defaulted_gives_lower_confidence(self):
        inputs = build_user_inputs(location="Chennai", capacity_kwp=5000, time_range_days=5)
        label, n_provided, _ = compute_confidence_label(inputs)
        assert 'Lower confidence' in label
        assert n_provided == 0

    def test_exactly_half_defaulted_is_moderate_not_lower(self):
        inputs = build_user_inputs(
            location="Chennai", capacity_kwp=5000, time_range_days=5,
            tilt_deg=20.0, azimuth_deg=170.0, inverter_efficiency=0.95,
            # temp_coefficient and degradation_pct_per_year left blank -> 2 defaulted
        )
        label, n_provided, n_total = compute_confidence_label(inputs)
        assert 'Moderate confidence' in label
        assert n_provided == 3

    # validates seasonal applicability flags
    def test_within_training_window_is_not_out_of_range(self):
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG, requested_months={5, 6})
        assert is_out is False
        assert reasons == []

    def test_season_mismatch_alone_flags_out_of_range(self):
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG, requested_months={12})
        assert is_out is True
        assert any('May-Jun' in r for r in reasons)

    def test_far_latitude_alone_flags_out_of_range(self):
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG + 20, requested_months={5, 6})
        assert is_out is True
        assert any('latitude' in r for r in reasons)

    # confirms that +- 15 degrees latitude threshold check triggers out of range flags accurately
    def test_latitude_just_inside_15_degrees_is_fine(self):
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG + 14.9, requested_months={5, 6})
        assert is_out is False

    def test_latitude_just_outside_15_degrees_flags(self):
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG + 15.1, requested_months={5, 6})
        assert is_out is True

    def test_both_season_and_latitude_mismatch_gives_two_reasons(self):
        """Verifies both failure reasons are reported when multiple conditions fail at once."""
        is_out, reasons = check_out_of_range(lat=PLANT_LATITUDE_DEG + 30, requested_months={12})
        assert is_out is True
        assert len(reasons) == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
