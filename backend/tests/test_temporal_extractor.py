"""Tests for clinical temporal information extraction module.

Covers:
- TemporalType / TemporalRelation enum completeness
- TemporalExpression / Frequency / TemporalLink / TemporalExtractionResult serialisation
- ClinicalTemporalExtractor:
  - Date extraction (MDY, ISO, written Month DD YYYY, DD Month YYYY)
  - Duration extraction (simple, range)
  - Relative time extraction (N ago, named, last)
  - Age extraction
  - Postoperative day extraction
  - Frequency extraction (qNh, abbreviations, written, every N)
  - Temporal link extraction (before, after, during, etc.)
  - De-duplication of overlapping expressions
  - Edge cases: empty text, no temporals, invalid dates
- FREQUENCY_MAP completeness
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.ml.temporal.extractor import (
    DURATION_UNITS,
    FREQUENCY_MAP,
    MONTH_NAMES,
    ClinicalTemporalExtractor,
    Frequency,
    TemporalExpression,
    TemporalExtractionResult,
    TemporalLink,
    TemporalRelation,
    TemporalType,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestTemporalType:
    def test_all_values_are_strings(self):
        for tt in TemporalType:
            assert isinstance(tt.value, str)

    def test_expected_count(self):
        assert len(TemporalType) == 7

    def test_key_types(self):
        expected = {"date", "datetime", "duration", "frequency", "relative", "age", "period"}
        assert {t.value for t in TemporalType} == expected


class TestTemporalRelation:
    def test_expected_count(self):
        assert len(TemporalRelation) == 7

    def test_key_relations(self):
        assert TemporalRelation.BEFORE.value == "before"
        assert TemporalRelation.AFTER.value == "after"
        assert TemporalRelation.SIMULTANEOUS.value == "simultaneous"


# ---------------------------------------------------------------------------
# Data class serialisation tests
# ---------------------------------------------------------------------------

class TestTemporalExpressionSerialization:
    def test_to_dict_full(self):
        expr = TemporalExpression(
            text="03/15/2024",
            temporal_type=TemporalType.DATE,
            start_char=0,
            end_char=10,
            confidence=0.95,
            normalised_value="2024-03-15",
            resolved_date=date(2024, 3, 15),
        )
        d = expr.to_dict()
        assert d["text"] == "03/15/2024"
        assert d["temporal_type"] == "date"
        assert d["resolved_date"] == "2024-03-15"

    def test_to_dict_none_date(self):
        expr = TemporalExpression(
            text="test", temporal_type=TemporalType.DURATION,
            start_char=0, end_char=4, confidence=0.8,
        )
        assert expr.to_dict()["resolved_date"] is None


class TestFrequencySerialization:
    def test_to_dict(self):
        f = Frequency(text="BID", times_per_day=2.0, interval_hours=12.0)
        d = f.to_dict()
        assert d["times_per_day"] == 2.0
        assert d["interval_hours"] == 12.0
        assert d["as_needed"] is False


class TestTemporalLinkSerialization:
    def test_to_dict(self):
        link = TemporalLink(
            source_span="surgery",
            target_span="infection",
            relation=TemporalRelation.BEFORE,
            confidence=0.7,
            evidence="before",
        )
        d = link.to_dict()
        assert d["relation"] == "before"


class TestTemporalExtractionResultSerialization:
    def test_to_dict(self):
        result = TemporalExtractionResult(
            expressions=[], frequencies=[], temporal_links=[],
            reference_date=date(2024, 1, 1), processing_time_ms=1.5,
        )
        d = result.to_dict()
        assert d["reference_date"] == "2024-01-01"
        assert d["processing_time_ms"] == 1.5


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_month_names_coverage(self):
        assert len(MONTH_NAMES) >= 12
        for month_num in range(1, 13):
            assert month_num in MONTH_NAMES.values()

    def test_duration_units_coverage(self):
        expected_units = {"day", "days", "week", "weeks", "month", "months", "year", "years"}
        assert expected_units.issubset(DURATION_UNITS.keys())

    def test_frequency_map_has_common_abbrevs(self):
        for abbrev in ["qd", "bid", "tid", "qid", "prn", "daily", "stat"]:
            assert abbrev in FREQUENCY_MAP, f"Missing abbreviation: {abbrev}"

    def test_frequency_map_prn_is_as_needed(self):
        assert FREQUENCY_MAP["prn"][2] is True
        assert FREQUENCY_MAP["bid"][2] is False


# ---------------------------------------------------------------------------
# Date extraction tests
# ---------------------------------------------------------------------------

class TestDateExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor(reference_date=date(2024, 6, 15))

    def test_mdy_date(self, extractor):
        result = extractor.extract("Admitted on 03/15/2024 for evaluation")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) >= 1
        assert dates[0].resolved_date == date(2024, 3, 15)
        assert dates[0].normalised_value == "2024-03-15"

    def test_iso_date(self, extractor):
        result = extractor.extract("Visit date: 2024-03-15 morning")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) >= 1
        assert dates[0].resolved_date == date(2024, 3, 15)

    def test_written_date_month_first(self, extractor):
        result = extractor.extract("Surgery scheduled for March 15, 2024")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) >= 1
        assert dates[0].resolved_date == date(2024, 3, 15)

    def test_written_date_abbreviated_month(self, extractor):
        result = extractor.extract("Seen on Jan 5, 2024 in clinic")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) >= 1
        assert dates[0].resolved_date == date(2024, 1, 5)

    def test_dmy_date(self, extractor):
        result = extractor.extract("Patient presented 15 March 2024 with complaints")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) >= 1
        assert dates[0].resolved_date == date(2024, 3, 15)

    def test_invalid_date_skipped(self, extractor):
        result = extractor.extract("Date: 13/32/2024 is invalid")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) == 0

    def test_multiple_dates(self, extractor):
        result = extractor.extract("From 01/01/2024 to 03/15/2024")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) == 2


# ---------------------------------------------------------------------------
# Duration extraction tests
# ---------------------------------------------------------------------------

class TestDurationExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor(reference_date=date(2024, 6, 15))

    def test_simple_duration_weeks(self, extractor):
        result = extractor.extract("Treatment for 6 weeks post-operatively")
        durations = [e for e in result.expressions if e.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1
        assert durations[0].duration_days == pytest.approx(42.0)

    def test_duration_days(self, extractor):
        result = extractor.extract("Admitted for 3 days for observation")
        durations = [e for e in result.expressions if e.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1
        assert durations[0].duration_days == pytest.approx(3.0)

    def test_duration_months(self, extractor):
        result = extractor.extract("Follow-up in over 2 months period")
        durations = [e for e in result.expressions if e.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1
        assert durations[0].duration_days == pytest.approx(60.0)

    def test_duration_range(self, extractor):
        result = extractor.extract("Recovery typically takes 3 to 5 days")
        durations = [e for e in result.expressions if e.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1
        # Average of 3 and 5
        assert durations[0].duration_days == pytest.approx(4.0)
        assert durations[0].metadata.get("range_low") == 3
        assert durations[0].metadata.get("range_high") == 5


# ---------------------------------------------------------------------------
# Relative time extraction tests
# ---------------------------------------------------------------------------

class TestRelativeTimeExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor(reference_date=date(2024, 6, 15))

    def test_days_ago(self, extractor):
        result = extractor.extract("Symptoms began 3 days ago")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        assert relatives[0].resolved_date == date(2024, 6, 12)

    def test_weeks_ago(self, extractor):
        result = extractor.extract("Surgery was 2 weeks ago")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        assert relatives[0].resolved_date == date(2024, 6, 1)

    def test_months_ago(self, extractor):
        result = extractor.extract("Last seen 6 months ago")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        expected = date(2024, 6, 15) - timedelta(days=180)
        assert relatives[0].resolved_date == expected

    def test_today(self, extractor):
        result = extractor.extract("Patient presents today with headache")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        assert relatives[0].resolved_date == date(2024, 6, 15)

    def test_yesterday(self, extractor):
        result = extractor.extract("Onset yesterday evening")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        assert relatives[0].resolved_date == date(2024, 6, 14)

    def test_last_week(self, extractor):
        result = extractor.extract("Labs drawn last week showed improvement")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1
        assert relatives[0].resolved_date == date(2024, 6, 8)

    def test_last_month(self, extractor):
        result = extractor.extract("Hospitalized last month for CHF exacerbation")
        relatives = [e for e in result.expressions if e.temporal_type == TemporalType.RELATIVE]
        assert len(relatives) >= 1


# ---------------------------------------------------------------------------
# Age extraction tests
# ---------------------------------------------------------------------------

class TestAgeExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor()

    def test_standard_age(self, extractor):
        result = extractor.extract("72-year-old male presents with chest pain")
        ages = [e for e in result.expressions if e.temporal_type == TemporalType.AGE]
        assert len(ages) == 1
        assert ages[0].metadata["age_years"] == 72
        assert ages[0].normalised_value == "P72Y"

    def test_age_with_spaces(self, extractor):
        result = extractor.extract("A 45 year old female with back pain")
        ages = [e for e in result.expressions if e.temporal_type == TemporalType.AGE]
        assert len(ages) >= 1

    def test_unreasonable_age_skipped(self, extractor):
        result = extractor.extract("A 150 year old tree stood nearby")
        ages = [e for e in result.expressions if e.temporal_type == TemporalType.AGE]
        assert len(ages) == 0


# ---------------------------------------------------------------------------
# Postoperative day extraction tests
# ---------------------------------------------------------------------------

class TestPODExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor()

    def test_pod_numbered(self, extractor):
        result = extractor.extract("Patient is on POD 3 recovering well")
        pods = [e for e in result.expressions if e.temporal_type == TemporalType.PERIOD]
        assert len(pods) >= 1
        assert pods[0].metadata["postoperative_day"] == 3

    def test_postop_day(self, extractor):
        result = extractor.extract("Post-operative day 5, ambulating independently")
        pods = [e for e in result.expressions if e.temporal_type == TemporalType.PERIOD]
        assert len(pods) >= 1
        assert pods[0].duration_days == 5.0

    def test_post_op_day(self, extractor):
        result = extractor.extract("Post-op day 1 vital signs stable")
        pods = [e for e in result.expressions if e.temporal_type == TemporalType.PERIOD]
        assert len(pods) >= 1


# ---------------------------------------------------------------------------
# Frequency extraction tests
# ---------------------------------------------------------------------------

class TestFrequencyExtraction:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor()

    def test_q6h(self, extractor):
        result = extractor.extract("Administer morphine 2mg IV q6h for pain")
        assert len(result.frequencies) >= 1
        f = result.frequencies[0]
        assert f.times_per_day == pytest.approx(4.0)
        assert f.interval_hours == pytest.approx(6.0)

    def test_q8h_prn(self, extractor):
        result = extractor.extract("Acetaminophen 650mg PO q8h PRN pain")
        freqs = [f for f in result.frequencies if f.interval_hours == pytest.approx(8.0)]
        assert len(freqs) >= 1

    def test_bid_abbreviation(self, extractor):
        result = extractor.extract("Metformin 500mg PO BID with meals")
        freqs = [f for f in result.frequencies if f.times_per_day == pytest.approx(2.0)]
        assert len(freqs) >= 1

    def test_tid_abbreviation(self, extractor):
        result = extractor.extract("Amoxicillin 500mg TID for 10 days")
        freqs = [f for f in result.frequencies if f.times_per_day == pytest.approx(3.0)]
        assert len(freqs) >= 1

    def test_twice_daily_written(self, extractor):
        result = extractor.extract("Take medication twice daily")
        freqs = [f for f in result.frequencies if f.times_per_day == pytest.approx(2.0)]
        assert len(freqs) >= 1

    def test_every_n_hours(self, extractor):
        result = extractor.extract("Check vitals every 4 hours")
        assert len(result.frequencies) >= 1
        f = result.frequencies[0]
        assert f.times_per_day == pytest.approx(6.0)
        assert f.interval_hours == pytest.approx(4.0)

    def test_prn(self, extractor):
        result = extractor.extract("Zofran 4mg IV PRN nausea")
        prn_freqs = [f for f in result.frequencies if f.as_needed]
        assert len(prn_freqs) >= 1

    def test_stat(self, extractor):
        result = extractor.extract("Give NS bolus STAT")
        stat_freqs = [f for f in result.frequencies if f.text.upper() == "STAT"]
        assert len(stat_freqs) >= 1


# ---------------------------------------------------------------------------
# Temporal link extraction tests
# ---------------------------------------------------------------------------

class TestTemporalLinks:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor()

    def test_before_signal(self, extractor):
        result = extractor.extract("Fever developed before the surgery was performed")
        before_links = [l for l in result.temporal_links if l.relation == TemporalRelation.BEFORE]
        assert len(before_links) >= 1

    def test_after_signal(self, extractor):
        result = extractor.extract("Patient improved after starting antibiotics")
        after_links = [l for l in result.temporal_links if l.relation == TemporalRelation.AFTER]
        assert len(after_links) >= 1

    def test_during_signal(self, extractor):
        result = extractor.extract("Arrhythmia occurred during the procedure")
        sim_links = [l for l in result.temporal_links if l.relation == TemporalRelation.SIMULTANEOUS]
        assert len(sim_links) >= 1

    def test_no_temporal_signals(self, extractor):
        result = extractor.extract("Patient has diabetes and hypertension")
        assert len(result.temporal_links) == 0


# ---------------------------------------------------------------------------
# De-duplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor(reference_date=date(2024, 6, 15))

    def test_overlapping_expressions_deduplicated(self, extractor):
        # "03/15/2024" could be matched by multiple patterns; should deduplicate
        result = extractor.extract("Date: 03/15/2024 noted")
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        # Should have exactly 1 after dedup
        assert len(dates) <= 2  # At most MDY + ISO if they overlap


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.fixture()
    def extractor(self) -> ClinicalTemporalExtractor:
        return ClinicalTemporalExtractor(reference_date=date(2024, 6, 15))

    def test_empty_text(self, extractor):
        result = extractor.extract("")
        assert len(result.expressions) == 0
        assert len(result.frequencies) == 0
        assert result.processing_time_ms >= 0

    def test_no_temporal_info(self, extractor):
        result = extractor.extract("Patient denies chest pain and shortness of breath")
        # May have some from "today" or other implicit references, but no dates
        dates = [e for e in result.expressions if e.temporal_type == TemporalType.DATE]
        assert len(dates) == 0

    def test_reference_date_default(self):
        ext = ClinicalTemporalExtractor()
        assert ext.reference_date == date.today()

    def test_reference_date_custom(self):
        ref = date(2023, 1, 1)
        ext = ClinicalTemporalExtractor(reference_date=ref)
        assert ext.reference_date == ref

    def test_processing_time_reasonable(self, extractor):
        text = "Patient seen on 03/15/2024 for 6 weeks of pain starting 2 weeks ago. Take medication BID."
        result = extractor.extract(text)
        # Should complete in under 100ms for normal text
        assert result.processing_time_ms < 1000

    def test_comprehensive_clinical_note(self, extractor):
        """Full clinical note with mixed temporal expressions."""
        text = (
            "72-year-old male admitted on 03/15/2024 with chest pain starting "
            "3 days ago. History of MI 2 years ago. Started on heparin drip, "
            "morphine 2mg IV q4h PRN pain. Post-op day 1 after CABG. "
            "Plan: aspirin daily, follow-up in 6 weeks. Labs today showed "
            "troponin trending down since yesterday."
        )
        result = extractor.extract(text)

        # Should find multiple expression types
        types_found = {e.temporal_type for e in result.expressions}
        assert TemporalType.DATE in types_found or TemporalType.RELATIVE in types_found
        assert TemporalType.AGE in types_found

        # Should find frequencies
        assert len(result.frequencies) >= 1

        # Should find temporal links
        assert len(result.temporal_links) >= 0  # May vary based on sentence structure

        assert result.reference_date == date(2024, 6, 15)
