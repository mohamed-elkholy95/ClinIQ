"""Tests for the clinical vital signs extraction module.

Covers all 9 vital sign types, unit conversions, clinical interpretation,
section awareness, trend detection, qualitative descriptors, deduplication,
batch extraction, BMI calculation, edge cases, and realistic clinical notes.
"""

from __future__ import annotations

import pytest

from app.ml.vitals.extractor import (
    ClinicalInterpretation,
    ClinicalVitalSignsExtractor,
    VitalSignReading,
    VitalSignsResult,
    VitalSignType,
    VitalTrend,
    _calculate_bmi,
    _celsius_to_fahrenheit,
    _detect_trend,
    _feet_inches_to_cm,
    _in_vitals_section,
    _inches_to_cm,
    _interpret,
    _interpret_diastolic,
    _is_valid,
    _lbs_to_kg,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def extractor() -> ClinicalVitalSignsExtractor:
    """Return a default extractor."""
    return ClinicalVitalSignsExtractor()


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify enum members match expected values."""

    def test_vital_sign_types_count(self) -> None:
        assert len(VitalSignType) == 9

    def test_vital_sign_type_values(self) -> None:
        expected = {
            "blood_pressure", "heart_rate", "temperature",
            "respiratory_rate", "oxygen_saturation", "weight",
            "height", "bmi", "pain_scale",
        }
        assert {v.value for v in VitalSignType} == expected

    def test_clinical_interpretation_count(self) -> None:
        assert len(ClinicalInterpretation) == 5

    def test_vital_trend_count(self) -> None:
        assert len(VitalTrend) == 4

    def test_vital_trend_values(self) -> None:
        assert {v.value for v in VitalTrend} == {
            "improving", "worsening", "stable", "unknown",
        }


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass serialisation."""

    def test_reading_to_dict(self) -> None:
        r = VitalSignReading(
            vital_type=VitalSignType.HEART_RATE,
            value=72.0,
            unit="bpm",
            raw_text="HR 72",
            start=0,
            end=5,
            confidence=0.88,
            interpretation=ClinicalInterpretation.NORMAL,
        )
        d = r.to_dict()
        assert d["vital_type"] == "heart_rate"
        assert d["interpretation"] == "normal"
        assert d["trend"] == "unknown"
        assert d["value"] == 72.0

    def test_result_to_dict(self) -> None:
        result = VitalSignsResult(
            readings=[],
            text_hash="abc",
            extraction_time_ms=1.234,
            summary={"total": 0, "by_type": {}, "critical_findings": []},
        )
        d = result.to_dict()
        assert d["readings"] == []
        assert d["extraction_time_ms"] == 1.23
        assert d["text_hash"] == "abc"


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


class TestConversions:
    """Test unit conversion functions."""

    def test_celsius_to_fahrenheit_37(self) -> None:
        assert _celsius_to_fahrenheit(37.0) == 98.6

    def test_celsius_to_fahrenheit_0(self) -> None:
        assert _celsius_to_fahrenheit(0.0) == 32.0

    def test_celsius_to_fahrenheit_100(self) -> None:
        assert _celsius_to_fahrenheit(100.0) == 212.0

    def test_lbs_to_kg(self) -> None:
        result = _lbs_to_kg(154.0)
        assert 69.5 <= result <= 70.0

    def test_inches_to_cm(self) -> None:
        assert _inches_to_cm(70.0) == 177.8

    def test_feet_inches_to_cm(self) -> None:
        result = _feet_inches_to_cm(5, 10)
        assert 177.0 <= result <= 178.0

    def test_calculate_bmi_normal(self) -> None:
        bmi = _calculate_bmi(70.0, 175.0)
        assert bmi is not None
        assert 22.0 <= bmi <= 23.0

    def test_calculate_bmi_zero_height(self) -> None:
        assert _calculate_bmi(70.0, 0.0) is None


# ---------------------------------------------------------------------------
# Interpretation and validation
# ---------------------------------------------------------------------------


class TestInterpretation:
    """Test clinical interpretation logic."""

    def test_normal_hr(self) -> None:
        assert _interpret(VitalSignType.HEART_RATE, 75.0) == ClinicalInterpretation.NORMAL

    def test_high_hr(self) -> None:
        assert _interpret(VitalSignType.HEART_RATE, 110.0) == ClinicalInterpretation.HIGH

    def test_low_hr(self) -> None:
        assert _interpret(VitalSignType.HEART_RATE, 55.0) == ClinicalInterpretation.LOW

    def test_critical_high_hr(self) -> None:
        assert _interpret(VitalSignType.HEART_RATE, 160.0) == ClinicalInterpretation.CRITICAL_HIGH

    def test_critical_low_hr(self) -> None:
        assert _interpret(VitalSignType.HEART_RATE, 25.0) == ClinicalInterpretation.CRITICAL_LOW

    def test_normal_bp(self) -> None:
        assert _interpret(VitalSignType.BLOOD_PRESSURE, 120.0) == ClinicalInterpretation.NORMAL

    def test_high_bp(self) -> None:
        assert _interpret(VitalSignType.BLOOD_PRESSURE, 155.0) == ClinicalInterpretation.HIGH

    def test_critical_high_bp(self) -> None:
        assert _interpret(VitalSignType.BLOOD_PRESSURE, 185.0) == ClinicalInterpretation.CRITICAL_HIGH

    def test_normal_temp(self) -> None:
        assert _interpret(VitalSignType.TEMPERATURE, 98.6) == ClinicalInterpretation.NORMAL

    def test_high_temp(self) -> None:
        assert _interpret(VitalSignType.TEMPERATURE, 100.5) == ClinicalInterpretation.HIGH

    def test_critical_high_temp(self) -> None:
        assert _interpret(VitalSignType.TEMPERATURE, 104.5) == ClinicalInterpretation.CRITICAL_HIGH

    def test_diastolic_normal(self) -> None:
        assert _interpret_diastolic(75.0) == ClinicalInterpretation.NORMAL

    def test_diastolic_high(self) -> None:
        assert _interpret_diastolic(95.0) == ClinicalInterpretation.HIGH

    def test_diastolic_critical_high(self) -> None:
        assert _interpret_diastolic(125.0) == ClinicalInterpretation.CRITICAL_HIGH

    def test_weight_has_no_range(self) -> None:
        # Weight doesn't have clinical ranges (too individual)
        assert _interpret(VitalSignType.WEIGHT, 200.0) == ClinicalInterpretation.NORMAL

    def test_bmi_normal(self) -> None:
        assert _interpret(VitalSignType.BMI, 22.0) == ClinicalInterpretation.NORMAL

    def test_bmi_high(self) -> None:
        assert _interpret(VitalSignType.BMI, 28.0) == ClinicalInterpretation.HIGH

    def test_bmi_critical_high(self) -> None:
        assert _interpret(VitalSignType.BMI, 42.0) == ClinicalInterpretation.CRITICAL_HIGH


class TestValidation:
    """Test physiological range validation."""

    def test_valid_hr(self) -> None:
        assert _is_valid(VitalSignType.HEART_RATE, 72.0) is True

    def test_invalid_hr_too_high(self) -> None:
        assert _is_valid(VitalSignType.HEART_RATE, 400.0) is False

    def test_invalid_hr_too_low(self) -> None:
        assert _is_valid(VitalSignType.HEART_RATE, 5.0) is False

    def test_valid_temp(self) -> None:
        assert _is_valid(VitalSignType.TEMPERATURE, 98.6) is True

    def test_invalid_temp(self) -> None:
        assert _is_valid(VitalSignType.TEMPERATURE, 120.0) is False

    def test_valid_spo2(self) -> None:
        assert _is_valid(VitalSignType.OXYGEN_SATURATION, 98.0) is True

    def test_invalid_spo2_over_100(self) -> None:
        assert _is_valid(VitalSignType.OXYGEN_SATURATION, 105.0) is False

    def test_valid_pain(self) -> None:
        assert _is_valid(VitalSignType.PAIN_SCALE, 5.0) is True

    def test_invalid_pain(self) -> None:
        assert _is_valid(VitalSignType.PAIN_SCALE, 15.0) is False


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------


class TestTrend:
    """Test trend detection from context."""

    def test_improving(self) -> None:
        text = "BP improved to 120/80 mmHg"
        assert _detect_trend(text, 15, 25) == VitalTrend.IMPROVING

    def test_worsening(self) -> None:
        text = "HR worsening, now 120 bpm"
        assert _detect_trend(text, 18, 25) == VitalTrend.WORSENING

    def test_stable(self) -> None:
        text = "Temperature stable at 98.6 F"
        assert _detect_trend(text, 22, 28) == VitalTrend.STABLE

    def test_no_trend(self) -> None:
        text = "BP 120/80 mmHg"
        assert _detect_trend(text, 3, 14) == VitalTrend.UNKNOWN


# ---------------------------------------------------------------------------
# Section awareness
# ---------------------------------------------------------------------------


class TestSectionAwareness:
    """Test vital signs section detection."""

    def test_in_vitals_section(self) -> None:
        text = "Vital Signs:\nBP 120/80\nHR 72\n\nAssessment:\nStable."
        # BP position is inside Vital Signs section
        assert _in_vitals_section(text, 16) is True

    def test_not_in_vitals_section(self) -> None:
        text = "Chief Complaint:\nChest pain\n\nVital Signs:\nBP 120/80"
        assert _in_vitals_section(text, 20) is False

    def test_after_section_ends(self) -> None:
        text = "Vital Signs:\nBP 120/80\n\nASSESSMENT:\nPatient stable."
        # Position in Assessment section (after uppercase header terminator)
        assert _in_vitals_section(text, 42) is False

    def test_confidence_boost_in_section(self, extractor: ClinicalVitalSignsExtractor) -> None:
        text = "Vital Signs:\nHR 72 bpm"
        result = extractor.extract(text)
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) >= 1
        # Should have boosted confidence
        assert hr[0].confidence > 0.88


# ---------------------------------------------------------------------------
# Blood Pressure extraction
# ---------------------------------------------------------------------------


class TestBloodPressure:
    """Test blood pressure extraction patterns."""

    def test_labeled_bp(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 120/80 mmHg")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 1
        assert bp[0].value == 120.0
        assert bp[0].secondary_value == 80.0

    def test_bp_with_colon(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Blood Pressure: 135/85")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 1
        assert bp[0].value == 135.0

    def test_bp_map_calculation(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 120/80")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) >= 1
        assert "mean_arterial_pressure" in bp[0].metadata
        # MAP = (120 + 2*80) / 3 = 93.3
        assert abs(bp[0].metadata["mean_arterial_pressure"] - 93.3) < 0.5

    def test_hypertensive_bp(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 185/110 mmHg")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 1
        assert bp[0].interpretation in (
            ClinicalInterpretation.HIGH,
            ClinicalInterpretation.CRITICAL_HIGH,
        )

    def test_invalid_bp_systolic_below_diastolic(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 70/120 mmHg")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 0

    def test_unit_required_bp(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("120/80 mmHg")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) >= 1


# ---------------------------------------------------------------------------
# Heart Rate extraction
# ---------------------------------------------------------------------------


class TestHeartRate:
    """Test heart rate extraction patterns."""

    def test_hr_labeled(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("HR 72 bpm")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) >= 1
        assert hr[0].value == 72.0

    def test_hr_pulse(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Pulse: 88")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) == 1
        assert hr[0].value == 88.0

    def test_heart_rate_full(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Heart Rate: 65 bpm")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) >= 1
        assert hr[0].value == 65.0

    def test_tachycardia(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("HR 120 bpm")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) >= 1
        assert hr[0].interpretation == ClinicalInterpretation.HIGH

    def test_bradycardia(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("HR 45")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) >= 1
        assert hr[0].interpretation == ClinicalInterpretation.LOW


# ---------------------------------------------------------------------------
# Temperature extraction
# ---------------------------------------------------------------------------


class TestTemperature:
    """Test temperature extraction with unit conversion."""

    def test_temp_fahrenheit(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Temp 98.6 F")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].value == 98.6

    def test_temp_celsius_conversion(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("T: 37.0 C")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert abs(t[0].value - 98.6) < 0.2
        assert t[0].metadata.get("converted") is True

    def test_temp_without_unit(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Temp: 98.6")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].value == 98.6

    def test_fever_detection(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Temperature: 102.5 F")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].interpretation == ClinicalInterpretation.HIGH

    def test_auto_detect_celsius(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """Temperature <= 50 without unit → assumed Celsius."""
        result = extractor.extract("Temp: 37.0")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].metadata.get("converted") is True


# ---------------------------------------------------------------------------
# Respiratory Rate extraction
# ---------------------------------------------------------------------------


class TestRespiratoryRate:
    """Test respiratory rate extraction."""

    def test_rr_labeled(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("RR 18")
        rr = [r for r in result.readings if r.vital_type == VitalSignType.RESPIRATORY_RATE]
        assert len(rr) == 1
        assert rr[0].value == 18.0

    def test_resp_rate_full(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Respiratory Rate: 20 breaths/min")
        rr = [r for r in result.readings if r.vital_type == VitalSignType.RESPIRATORY_RATE]
        assert len(rr) == 1
        assert rr[0].value == 20.0

    def test_tachypnea(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("RR 28")
        rr = [r for r in result.readings if r.vital_type == VitalSignType.RESPIRATORY_RATE]
        assert len(rr) == 1
        assert rr[0].interpretation == ClinicalInterpretation.HIGH


# ---------------------------------------------------------------------------
# Oxygen Saturation extraction
# ---------------------------------------------------------------------------


class TestOxygenSaturation:
    """Test SpO2 extraction."""

    def test_spo2_labeled(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("SpO2 98%")
        sp = [r for r in result.readings if r.vital_type == VitalSignType.OXYGEN_SATURATION]
        assert len(sp) == 1
        assert sp[0].value == 98.0

    def test_o2_sat(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("O2 Sat: 95%")
        sp = [r for r in result.readings if r.vital_type == VitalSignType.OXYGEN_SATURATION]
        assert len(sp) == 1
        assert sp[0].value == 95.0

    def test_on_room_air(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("97% on RA")
        sp = [r for r in result.readings if r.vital_type == VitalSignType.OXYGEN_SATURATION]
        assert len(sp) == 1
        assert sp[0].value == 97.0

    def test_hypoxia(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("SpO2 85%")
        sp = [r for r in result.readings if r.vital_type == VitalSignType.OXYGEN_SATURATION]
        assert len(sp) == 1
        assert sp[0].interpretation in (
            ClinicalInterpretation.LOW,
            ClinicalInterpretation.CRITICAL_LOW,
        )


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


class TestWeight:
    """Test weight extraction with unit conversion."""

    def test_weight_kg(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Weight: 70 kg")
        w = [r for r in result.readings if r.vital_type == VitalSignType.WEIGHT]
        assert len(w) >= 1
        assert w[0].value == 70.0

    def test_weight_lbs_conversion(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Wt: 154 lbs")
        w = [r for r in result.readings if r.vital_type == VitalSignType.WEIGHT]
        assert len(w) == 1
        assert 69.5 <= w[0].value <= 70.0
        assert w[0].metadata.get("converted") is True

    def test_standalone_kg(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Body weight 85 kg")
        w = [r for r in result.readings if r.vital_type == VitalSignType.WEIGHT]
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# Height extraction
# ---------------------------------------------------------------------------


class TestHeight:
    """Test height extraction with unit conversion."""

    def test_height_feet_inches(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Height: 5'10\"")
        h = [r for r in result.readings if r.vital_type == VitalSignType.HEIGHT]
        assert len(h) == 1
        assert 177.0 <= h[0].value <= 178.0
        assert h[0].metadata.get("converted") is True

    def test_height_cm(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Ht: 175 cm")
        h = [r for r in result.readings if r.vital_type == VitalSignType.HEIGHT]
        assert len(h) == 1
        assert h[0].value == 175.0

    def test_height_inches(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Height: 70 inches")
        h = [r for r in result.readings if r.vital_type == VitalSignType.HEIGHT]
        assert len(h) == 1
        assert abs(h[0].value - 177.8) < 0.5


# ---------------------------------------------------------------------------
# BMI extraction
# ---------------------------------------------------------------------------


class TestBMI:
    """Test BMI extraction and calculation."""

    def test_explicit_bmi(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BMI 24.5")
        bmi = [r for r in result.readings if r.vital_type == VitalSignType.BMI]
        assert len(bmi) == 1
        assert bmi[0].value == 24.5

    def test_bmi_with_unit(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BMI: 31.2 kg/m2")
        bmi = [r for r in result.readings if r.vital_type == VitalSignType.BMI]
        assert len(bmi) == 1
        assert bmi[0].value == 31.2

    def test_bmi_interpretation_normal(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BMI 22.0")
        bmi = [r for r in result.readings if r.vital_type == VitalSignType.BMI]
        assert bmi[0].interpretation == ClinicalInterpretation.NORMAL

    def test_bmi_interpretation_high(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BMI 28.5")
        bmi = [r for r in result.readings if r.vital_type == VitalSignType.BMI]
        assert bmi[0].interpretation == ClinicalInterpretation.HIGH

    def test_calculated_bmi(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """BMI should be auto-calculated when weight and height are present."""
        result = extractor.extract("Weight: 70 kg\nHeight: 175 cm")
        bmi = [r for r in result.readings if r.vital_type == VitalSignType.BMI]
        assert len(bmi) == 1
        assert bmi[0].metadata.get("calculated") is True
        assert 22.0 <= bmi[0].value <= 23.0


# ---------------------------------------------------------------------------
# Pain Scale extraction
# ---------------------------------------------------------------------------


class TestPainScale:
    """Test pain scale extraction."""

    def test_pain_score(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Pain: 5/10")
        p = [r for r in result.readings if r.vital_type == VitalSignType.PAIN_SCALE]
        assert len(p) == 1
        assert p[0].value == 5.0

    def test_pain_scale_labeled(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Pain Scale: 7")
        p = [r for r in result.readings if r.vital_type == VitalSignType.PAIN_SCALE]
        assert len(p) == 1
        assert p[0].value == 7.0

    def test_pain_zero(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Pain: 0/10")
        p = [r for r in result.readings if r.vital_type == VitalSignType.PAIN_SCALE]
        assert len(p) == 1
        assert p[0].value == 0.0

    def test_high_pain_interpretation(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Pain: 8/10")
        p = [r for r in result.readings if r.vital_type == VitalSignType.PAIN_SCALE]
        assert len(p) == 1
        assert p[0].interpretation == ClinicalInterpretation.CRITICAL_HIGH


# ---------------------------------------------------------------------------
# Qualitative descriptors
# ---------------------------------------------------------------------------


class TestQualitative:
    """Test qualitative vital sign descriptors."""

    def test_afebrile(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient is afebrile.")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].interpretation == ClinicalInterpretation.NORMAL
        assert t[0].metadata.get("qualitative") is True

    def test_febrile(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient is febrile.")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert len(t) == 1
        assert t[0].interpretation == ClinicalInterpretation.HIGH

    def test_tachycardic(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient appears tachycardic.")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) == 1
        assert hr[0].interpretation == ClinicalInterpretation.HIGH

    def test_bradycardic(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Noted bradycardic rhythm.")
        hr = [r for r in result.readings if r.vital_type == VitalSignType.HEART_RATE]
        assert len(hr) == 1
        assert hr[0].interpretation == ClinicalInterpretation.LOW

    def test_hypotensive(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient is hypotensive.")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 1
        assert bp[0].interpretation == ClinicalInterpretation.LOW

    def test_hypertensive(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient is hypertensive.")
        bp = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp) == 1
        assert bp[0].interpretation == ClinicalInterpretation.HIGH

    def test_hypoxic(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient hypoxic on presentation.")
        sp = [r for r in result.readings if r.vital_type == VitalSignType.OXYGEN_SATURATION]
        assert len(sp) == 1
        assert sp[0].interpretation == ClinicalInterpretation.LOW

    def test_tachypneic(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient tachypneic at rest.")
        rr = [r for r in result.readings if r.vital_type == VitalSignType.RESPIRATORY_RATE]
        assert len(rr) == 1
        assert rr[0].interpretation == ClinicalInterpretation.HIGH

    def test_qualitative_lower_confidence(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient afebrile.")
        t = [r for r in result.readings if r.vital_type == VitalSignType.TEMPERATURE]
        assert t[0].confidence == 0.70


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test overlapping span removal."""

    def test_dedup_keeps_higher_confidence(self) -> None:
        readings = [
            VitalSignReading(
                vital_type=VitalSignType.HEART_RATE, value=72.0, unit="bpm",
                raw_text="HR 72", start=0, end=5, confidence=0.90,
                interpretation=ClinicalInterpretation.NORMAL,
            ),
            VitalSignReading(
                vital_type=VitalSignType.HEART_RATE, value=72.0, unit="bpm",
                raw_text="HR 72 bpm", start=0, end=9, confidence=0.85,
                interpretation=ClinicalInterpretation.NORMAL,
            ),
        ]
        result = ClinicalVitalSignsExtractor._deduplicate(readings)
        assert len(result) == 1
        assert result[0].confidence == 0.90

    def test_dedup_different_types_kept(self) -> None:
        """Readings of different types at same position should both survive."""
        readings = [
            VitalSignReading(
                vital_type=VitalSignType.HEART_RATE, value=72.0, unit="bpm",
                raw_text="72", start=5, end=7, confidence=0.88,
                interpretation=ClinicalInterpretation.NORMAL,
            ),
            VitalSignReading(
                vital_type=VitalSignType.RESPIRATORY_RATE, value=18.0, unit="breaths/min",
                raw_text="18", start=5, end=7, confidence=0.88,
                interpretation=ClinicalInterpretation.NORMAL,
            ),
        ]
        result = ClinicalVitalSignsExtractor._deduplicate(readings)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


class TestBatch:
    """Test batch extraction."""

    def test_batch_returns_correct_count(self, extractor: ClinicalVitalSignsExtractor) -> None:
        texts = ["BP 120/80", "HR 72", "Temp: 98.6 F"]
        results = extractor.extract_batch(texts)
        assert len(results) == 3

    def test_batch_each_has_readings(self, extractor: ClinicalVitalSignsExtractor) -> None:
        texts = ["BP 120/80 mmHg", "HR 72 bpm"]
        results = extractor.extract_batch(texts)
        assert all(len(r.readings) >= 1 for r in results)

    def test_batch_empty_text(self, extractor: ClinicalVitalSignsExtractor) -> None:
        results = extractor.extract_batch(["", "HR 72"])
        assert len(results) == 2
        assert results[0].summary["total"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("")
        assert result.summary["total"] == 0
        assert result.readings == []

    def test_whitespace_only(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("   \n\t  ")
        assert result.summary["total"] == 0

    def test_no_vitals_in_text(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("Patient presents with headache and nausea.")
        # Should have no numeric vitals (possibly qualitative depending on terms)
        numeric = [r for r in result.readings if not r.metadata.get("qualitative")]
        assert len(numeric) == 0

    def test_text_hash_populated(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("HR 72")
        assert len(result.text_hash) == 64  # SHA-256 hex

    def test_extraction_time_positive(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 120/80 mmHg")
        assert result.extraction_time_ms >= 0.0

    def test_min_confidence_filter(self) -> None:
        ext = ClinicalVitalSignsExtractor(min_confidence=0.95)
        result = ext.extract("BP 120/80 mmHg, HR 72 bpm")
        # Most readings have confidence < 0.95, so many should be filtered
        for r in result.readings:
            assert r.confidence >= 0.95

    def test_summary_critical_findings(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("BP 200/120 mmHg")
        assert len(result.summary["critical_findings"]) >= 1

    def test_readings_sorted_by_position(self, extractor: ClinicalVitalSignsExtractor) -> None:
        result = extractor.extract("HR 72 bpm, BP 120/80 mmHg, RR 18")
        positions = [r.start for r in result.readings]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Realistic clinical notes
# ---------------------------------------------------------------------------


class TestRealisticNotes:
    """Test extraction from realistic clinical note fragments."""

    def test_full_vitals_set(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """Standard vital signs documentation."""
        text = (
            "Vital Signs:\n"
            "BP 132/84 mmHg, HR 78 bpm, T 98.4 F, RR 16, SpO2 97% on RA\n"
            "Weight: 82 kg, Height: 5'11\"\n"
            "Pain: 3/10"
        )
        result = extractor.extract(text)
        types_found = {r.vital_type for r in result.readings}
        # Should find most vital types
        assert VitalSignType.BLOOD_PRESSURE in types_found
        assert VitalSignType.HEART_RATE in types_found
        assert VitalSignType.TEMPERATURE in types_found
        assert VitalSignType.RESPIRATORY_RATE in types_found
        assert VitalSignType.OXYGEN_SATURATION in types_found
        assert VitalSignType.PAIN_SCALE in types_found

    def test_critical_presentation(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """Critical vital signs with abnormal interpretations."""
        text = (
            "EMERGENCY DEPARTMENT NOTE\n"
            "Vital Signs:\n"
            "BP 60/30 mmHg, HR 160 bpm, T 105.2 F, RR 32, SpO2 82% on 4L NC\n"
            "Patient appears tachycardic and hypoxic."
        )
        result = extractor.extract(text)
        assert result.summary["total"] >= 5
        assert len(result.summary["critical_findings"]) >= 1

    def test_mixed_qualitative_and_numeric(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """Mix of qualitative and numeric vital signs."""
        text = (
            "Patient is afebrile, hemodynamically stable.\n"
            "BP 118/72 mmHg, HR 68 bpm, RR 14.\n"
            "SpO2 99% on room air. Pain 2/10."
        )
        result = extractor.extract(text)
        qualitative = [r for r in result.readings if r.metadata.get("qualitative")]
        numeric = [r for r in result.readings if not r.metadata.get("qualitative")]
        assert len(qualitative) >= 1
        assert len(numeric) >= 3

    def test_discharge_summary_vitals(self, extractor: ClinicalVitalSignsExtractor) -> None:
        """Vital signs in a discharge summary context."""
        text = (
            "DISCHARGE SUMMARY\n\n"
            "Admission Vitals: BP 165/95 mmHg, HR 102 bpm, T 101.2 F\n"
            "Discharge Vitals: BP 128/78 mmHg, HR 74 bpm, T 98.4 F\n"
            "Blood pressure improved from admission. Heart rate normalized.\n"
            "BMI: 27.3 kg/m2\n"
            "Pain score: 1/10 at discharge."
        )
        result = extractor.extract(text)
        bp_readings = [r for r in result.readings if r.vital_type == VitalSignType.BLOOD_PRESSURE]
        assert len(bp_readings) >= 2  # Admission + discharge
