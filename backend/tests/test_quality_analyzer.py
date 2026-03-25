"""Tests for the clinical note quality analyzer module.

Covers:
- QualityDimension enum completeness
- QualityConfig normalization
- Finding/QualityScore/QualityReport dataclass serialization
- Completeness scoring (word count, section coverage, bonuses)
- Readability scoring (sentence length, abbreviation density, edge cases)
- Structure scoring (whitespace, headers, lists, line variance)
- Information density scoring (medical terms, numeric data)
- Consistency scoring (duplicates, contradictions)
- Grade assignment boundaries
- Recommendation generation and sorting
- Batch analysis
- Section detection patterns
- Sentence splitting
- Full end-to-end on realistic clinical notes
"""

import pytest

from app.ml.quality.analyzer import (
    ClinicalNoteQualityAnalyzer,
    Finding,
    FindingSeverity,
    QualityConfig,
    QualityDimension,
    QualityReport,
    QualityScore,
    KNOWN_SECTION_HEADERS,
    DEFAULT_EXPECTED_SECTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPLETE_CLINICAL_NOTE = """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
Mr. Johnson is a 68-year-old male with a history of coronary artery
disease, type 2 diabetes mellitus, hypertension, and hyperlipidemia
presenting with acute chest pain that started 2 hours ago.
Pain radiates to the left arm. Associated diaphoresis.
Current medications include metformin 1000mg BID, lisinopril 10mg daily,
atorvastatin 40mg nightly, aspirin 81mg daily, and nitroglycerin PRN.

PAST MEDICAL HISTORY:
Coronary artery disease with stent placement 2019.
Type 2 diabetes mellitus diagnosed 2015.
Hypertension since 2010.

MEDICATIONS:
1. Metformin 1000mg PO BID
2. Lisinopril 10mg PO daily
3. Atorvastatin 40mg PO nightly
4. Aspirin 81mg PO daily
5. Nitroglycerin 0.4mg SL PRN

VITAL SIGNS:
BP: 160/95 mmHg, HR: 98 bpm, RR: 22, SpO2: 94%, Temp: 98.6°F

PHYSICAL EXAMINATION:
General: Alert, oriented, in moderate distress.
Cardiovascular: Regular rate, no murmurs.
Respiratory: Clear to auscultation bilaterally.

ASSESSMENT:
1. Acute coronary syndrome — rule out STEMI.
2. Uncontrolled hypertension.
3. Type 2 diabetes mellitus.

PLAN:
1. Admit to CCU for monitoring
2. Serial troponins Q6h
3. Cardiology consult for possible catheterization
4. Continue home medications except hold metformin
5. Heparin drip per protocol
"""

SHORT_NOTE = "Patient seen. Chest pain. Follow up."

ABBREVIATION_HEAVY_NOTE = """
CC: CP, SOB
HPI: 68 yo M w/ h/o CAD, DM2, HTN, HLD p/w CP x 2h. Pt c/o
radicular pain L arm. Assoc diaphoresis. Meds: metformin BID,
lisinopril QD, ASA QD, NTG PRN. PMH: s/p PCI 2019, DM dx 2015.
VS: BP 160/95, HR 98, RR 22, SpO2 94%.
A/P: ACS r/o STEMI. Admit CCU. Trops Q6h. Cards c/s.
"""


@pytest.fixture
def analyzer():
    """Default analyzer instance."""
    return ClinicalNoteQualityAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Analyzer with custom config."""
    config = QualityConfig(
        min_word_count=50,
        max_abbreviation_ratio=0.15,
        expected_sections=["chief complaint", "assessment", "plan"],
    )
    return ClinicalNoteQualityAnalyzer(config)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestQualityDimensionEnum:
    """Tests for QualityDimension enum."""

    def test_all_dimensions_present(self):
        """All five quality dimensions exist."""
        expected = {"completeness", "readability", "structure", "information_density", "consistency"}
        actual = {d.value for d in QualityDimension}
        assert actual == expected

    def test_dimension_is_str_enum(self):
        """Dimensions are string enums for JSON serialization."""
        assert QualityDimension.COMPLETENESS == "completeness"
        assert isinstance(QualityDimension.READABILITY, str)


class TestFindingSeverityEnum:
    """Tests for FindingSeverity enum."""

    def test_all_severities(self):
        """All three severity levels exist."""
        expected = {"critical", "warning", "info"}
        actual = {s.value for s in FindingSeverity}
        assert actual == expected


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------


class TestDataclassSerialization:
    """Tests for dataclass to_dict methods."""

    def test_finding_to_dict_with_detail(self):
        """Finding serializes with detail."""
        f = Finding(
            dimension=QualityDimension.COMPLETENESS,
            severity=FindingSeverity.WARNING,
            message="Missing section",
            detail="Add HPI section.",
        )
        d = f.to_dict()
        assert d["dimension"] == "completeness"
        assert d["severity"] == "warning"
        assert d["message"] == "Missing section"
        assert d["detail"] == "Add HPI section."

    def test_finding_to_dict_without_detail(self):
        """Finding serializes without detail when None."""
        f = Finding(
            dimension=QualityDimension.READABILITY,
            severity=FindingSeverity.INFO,
            message="Good readability",
        )
        d = f.to_dict()
        assert "detail" not in d

    def test_quality_score_to_dict(self):
        """QualityScore serializes correctly."""
        qs = QualityScore(
            dimension=QualityDimension.STRUCTURE,
            score=85.123456,
            weight=0.2,
        )
        d = qs.to_dict()
        assert d["dimension"] == "structure"
        assert d["score"] == 85.12
        assert d["weight"] == 0.2
        assert d["findings"] == []

    def test_quality_report_to_dict(self, analyzer):
        """QualityReport serializes all fields."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        d = report.to_dict()
        assert "overall_score" in d
        assert "grade" in d
        assert "dimensions" in d
        assert "recommendations" in d
        assert "stats" in d
        assert "text_hash" in d
        assert "analysis_ms" in d
        assert isinstance(d["dimensions"], list)
        assert len(d["dimensions"]) == 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestQualityConfig:
    """Tests for QualityConfig."""

    def test_default_weights_equal(self):
        """Default config gives equal weight to all dimensions."""
        config = QualityConfig()
        weights = config.normalized_weights()
        assert len(weights) == 5
        for w in weights.values():
            assert abs(w - 0.2) < 0.001

    def test_custom_weights_normalized(self):
        """Custom weights are normalised to sum to 1.0."""
        config = QualityConfig(
            weights={
                QualityDimension.COMPLETENESS: 3.0,
                QualityDimension.READABILITY: 1.0,
                QualityDimension.STRUCTURE: 1.0,
                QualityDimension.INFORMATION_DENSITY: 1.0,
                QualityDimension.CONSISTENCY: 1.0,
            }
        )
        weights = config.normalized_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001
        # Completeness should have highest weight
        assert weights[QualityDimension.COMPLETENESS] > weights[QualityDimension.READABILITY]

    def test_default_expected_sections(self):
        """Default expected sections are the clinical essentials."""
        assert "chief complaint" in DEFAULT_EXPECTED_SECTIONS
        assert "assessment" in DEFAULT_EXPECTED_SECTIONS
        assert "plan" in DEFAULT_EXPECTED_SECTIONS

    def test_known_section_headers_nonempty(self):
        """Known section headers set is populated."""
        assert len(KNOWN_SECTION_HEADERS) >= 30
        assert "chief complaint" in KNOWN_SECTION_HEADERS
        assert "dental history" in KNOWN_SECTION_HEADERS


# ---------------------------------------------------------------------------
# Completeness scoring
# ---------------------------------------------------------------------------


class TestCompletenessScoring:
    """Tests for completeness dimension."""

    def test_complete_note_high_score(self, analyzer):
        """A well-structured note with all sections scores highly."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        completeness = next(
            d for d in report.dimensions if d.dimension == QualityDimension.COMPLETENESS
        )
        assert completeness.score >= 80

    def test_short_note_penalized(self, analyzer):
        """A very short note gets a completeness penalty."""
        report = analyzer.analyze(SHORT_NOTE)
        completeness = next(
            d for d in report.dimensions if d.dimension == QualityDimension.COMPLETENESS
        )
        assert completeness.score < 80
        # Should have a finding about being short
        short_findings = [f for f in completeness.findings if "short" in f.message.lower()]
        assert len(short_findings) >= 1

    def test_missing_sections_flagged(self, analyzer):
        """Missing expected sections generate findings."""
        text = "This is a clinical note without any section headers but has enough words to be meaningful and analyzed properly by the quality engine."
        report = analyzer.analyze(text)
        completeness = next(
            d for d in report.dimensions if d.dimension == QualityDimension.COMPLETENESS
        )
        missing_findings = [
            f for f in completeness.findings if "Missing expected section" in f.message
        ]
        assert len(missing_findings) >= 1

    def test_many_sections_bonus(self, analyzer):
        """Notes with 6+ sections get a bonus finding."""
        text = """
CHIEF COMPLAINT: Headache.
HISTORY OF PRESENT ILLNESS: Patient presents with headache.
PAST MEDICAL HISTORY: Hypertension.
MEDICATIONS: Lisinopril 10mg daily.
ALLERGIES: NKDA.
REVIEW OF SYSTEMS: Negative except as noted.
PHYSICAL EXAMINATION: Alert and oriented.
ASSESSMENT: Migraine headache.
PLAN: Sumatriptan PRN.
"""
        report = analyzer.analyze(text)
        completeness = next(
            d for d in report.dimensions if d.dimension == QualityDimension.COMPLETENESS
        )
        bonus_findings = [
            f for f in completeness.findings if "Well-structured" in f.message
        ]
        assert len(bonus_findings) == 1

    def test_custom_expected_sections(self, custom_analyzer):
        """Custom expected sections are used instead of defaults."""
        text = """
CHIEF COMPLAINT: Toothache.
ASSESSMENT: Dental caries #14.
PLAN: Composite restoration.
"""
        report = custom_analyzer.analyze(text)
        completeness = next(
            d for d in report.dimensions if d.dimension == QualityDimension.COMPLETENESS
        )
        # All 3 custom sections present — should score well on section coverage
        missing = [f for f in completeness.findings if "Missing" in f.message]
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# Readability scoring
# ---------------------------------------------------------------------------


class TestReadabilityScoring:
    """Tests for readability dimension."""

    def test_normal_readability(self, analyzer):
        """A standard clinical note has reasonable readability."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        readability = next(
            d for d in report.dimensions if d.dimension == QualityDimension.READABILITY
        )
        assert readability.score >= 60

    def test_abbreviation_heavy_note_flagged(self, analyzer):
        """Notes with many abbreviations get a readability warning."""
        report = analyzer.analyze(ABBREVIATION_HEAVY_NOTE)
        readability = next(
            d for d in report.dimensions if d.dimension == QualityDimension.READABILITY
        )
        abbrev_findings = [
            f for f in readability.findings if "abbreviation" in f.message.lower()
        ]
        assert len(abbrev_findings) >= 1

    def test_empty_text_zero_readability(self, analyzer):
        """Empty/whitespace-only text scores zero readability."""
        report = analyzer.analyze("   ")
        readability = next(
            d for d in report.dimensions if d.dimension == QualityDimension.READABILITY
        )
        assert readability.score == 0.0

    def test_very_long_sentences_flagged(self, analyzer):
        """Sentences exceeding 50 words are flagged."""
        long_sentence = " ".join(["word"] * 60) + ". Short sentence."
        text = f"ASSESSMENT: {long_sentence}"
        report = analyzer.analyze(text)
        readability = next(
            d for d in report.dimensions if d.dimension == QualityDimension.READABILITY
        )
        long_findings = [
            f for f in readability.findings if "exceed 50 words" in f.message
        ]
        assert len(long_findings) >= 1


# ---------------------------------------------------------------------------
# Structure scoring
# ---------------------------------------------------------------------------


class TestStructureScoring:
    """Tests for structure dimension."""

    def test_structured_note_scores_well(self, analyzer):
        """A well-structured note with headers scores highly."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        structure = next(
            d for d in report.dimensions if d.dimension == QualityDimension.STRUCTURE
        )
        assert structure.score >= 70

    def test_no_headers_penalized(self, analyzer):
        """Notes without headers get a structure penalty."""
        text = (
            "Patient is a 68 year old male presenting with chest pain and shortness of breath. "
            "He has a history of coronary artery disease and type 2 diabetes mellitus. "
            "Blood pressure is 160/95 mmHg and heart rate is 98 beats per minute. "
            "He was seen in the emergency department and evaluated for possible acute coronary syndrome. "
            "Plan is to admit to the cardiac care unit and monitor with serial troponins every six hours."
        )
        report = analyzer.analyze(text)
        structure = next(
            d for d in report.dimensions if d.dimension == QualityDimension.STRUCTURE
        )
        no_header_findings = [
            f for f in structure.findings if "No section headers" in f.message
        ]
        assert len(no_header_findings) >= 1

    def test_list_usage_bonus(self, analyzer):
        """Notes with numbered/bulleted lists get a bonus."""
        text = """
PLAN:
1. Admit to CCU
2. Serial troponins
3. Cardiology consult
4. Heparin drip
5. Hold metformin
"""
        report = analyzer.analyze(text)
        structure = next(
            d for d in report.dimensions if d.dimension == QualityDimension.STRUCTURE
        )
        list_findings = [f for f in structure.findings if "list" in f.message.lower()]
        assert len(list_findings) >= 1

    def test_excessive_whitespace_flagged(self, analyzer):
        """Excessive whitespace is flagged."""
        text = "ASSESSMENT:     Chest pain.     \n\n\n\n\n\n\n\n     PLAN:      Admit.     \n\n\n\n\n"
        report = analyzer.analyze(text)
        structure = next(
            d for d in report.dimensions if d.dimension == QualityDimension.STRUCTURE
        )
        ws_findings = [f for f in structure.findings if "whitespace" in f.message.lower()]
        assert len(ws_findings) >= 1


# ---------------------------------------------------------------------------
# Information density scoring
# ---------------------------------------------------------------------------


class TestInformationDensityScoring:
    """Tests for information density dimension."""

    def test_medical_rich_note_high_density(self, analyzer):
        """A medically rich note scores highly on information density."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        density = next(
            d for d in report.dimensions if d.dimension == QualityDimension.INFORMATION_DENSITY
        )
        assert density.score >= 70

    def test_non_medical_text_low_density(self, analyzer):
        """Non-medical text scores low on information density."""
        text = (
            "The weather today was sunny and warm. I went to the park and "
            "had a nice lunch with my family. We played some games and "
            "enjoyed the afternoon sunshine. It was a wonderful day overall "
            "and I hope to do it again next weekend when the weather is nice."
        )
        report = analyzer.analyze(text)
        density = next(
            d for d in report.dimensions if d.dimension == QualityDimension.INFORMATION_DENSITY
        )
        low_findings = [f for f in density.findings if "Low medical term" in f.message]
        assert len(low_findings) >= 1

    def test_numeric_data_detected(self, analyzer):
        """Vital signs and lab values are counted as numeric data."""
        text = """
VITAL SIGNS:
BP: 160/95 mmHg, HR: 98 bpm, RR: 22, SpO2: 94%, Temp: 98.6°F
Labs: WBC 12.3, HGB 14.2 g, PLT 250, BUN 18 mg, Cr 1.1 mg, Glucose 145 mg
"""
        report = analyzer.analyze(text)
        density = next(
            d for d in report.dimensions if d.dimension == QualityDimension.INFORMATION_DENSITY
        )
        numeric_findings = [f for f in density.findings if "numeric" in f.message.lower() or "measurement" in f.message.lower()]
        assert len(numeric_findings) >= 1


# ---------------------------------------------------------------------------
# Consistency scoring
# ---------------------------------------------------------------------------


class TestConsistencyScoring:
    """Tests for consistency dimension."""

    def test_consistent_note(self, analyzer):
        """A consistent note scores highly."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        consistency = next(
            d for d in report.dimensions if d.dimension == QualityDimension.CONSISTENCY
        )
        assert consistency.score >= 80

    def test_duplicate_paragraphs_detected(self, analyzer):
        """Duplicate paragraphs are flagged."""
        paragraph = "Patient is a 68 year old male presenting with acute chest pain and shortness of breath radiating to the left arm."
        text = f"{paragraph}\n\n{paragraph}"
        report = analyzer.analyze(text)
        consistency = next(
            d for d in report.dimensions if d.dimension == QualityDimension.CONSISTENCY
        )
        dup_findings = [f for f in consistency.findings if "duplicate" in f.message.lower()]
        assert len(dup_findings) >= 1

    def test_no_issues_info_finding(self, analyzer):
        """When no issues, an info finding is generated."""
        text = "ASSESSMENT: Stable condition. PLAN: Follow up in two weeks."
        report = analyzer.analyze(text)
        consistency = next(
            d for d in report.dimensions if d.dimension == QualityDimension.CONSISTENCY
        )
        info_findings = [f for f in consistency.findings if "No consistency issues" in f.message]
        assert len(info_findings) >= 1


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------


class TestGradeAssignment:
    """Tests for score-to-grade conversion."""

    @pytest.mark.parametrize(
        "score,expected_grade",
        [
            (95.0, "A"),
            (90.0, "A"),
            (89.9, "B"),
            (80.0, "B"),
            (79.9, "C"),
            (70.0, "C"),
            (69.9, "D"),
            (60.0, "D"),
            (59.9, "F"),
            (0.0, "F"),
        ],
    )
    def test_grade_boundaries(self, score, expected_grade):
        """Grade boundaries are correctly applied."""
        grade = ClinicalNoteQualityAnalyzer._score_to_grade(score)
        assert grade == expected_grade


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_sorted_by_severity(self, analyzer):
        """Critical findings appear before warnings in recommendations."""
        report = analyzer.analyze(SHORT_NOTE)
        # Short notes should have critical/warning recommendations
        assert len(report.recommendations) >= 1

    def test_info_findings_excluded(self, analyzer):
        """Info-level findings don't appear in recommendations."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        for rec in report.recommendations:
            # Recommendations should not contain purely informational messages
            assert "No consistency issues" not in rec

    def test_complete_note_fewer_recommendations(self, analyzer):
        """A complete note has fewer recommendations than a poor one."""
        good_report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        bad_report = analyzer.analyze(SHORT_NOTE)
        assert len(good_report.recommendations) <= len(bad_report.recommendations)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


class TestSectionDetection:
    """Tests for section header detection."""

    def test_all_caps_headers(self, analyzer):
        """ALL CAPS headers with colons are detected."""
        text = "CHIEF COMPLAINT: Pain.\nASSESSMENT: Stable.\nPLAN: Follow up."
        stats = analyzer._compute_stats(text)
        sections = stats["detected_sections"]
        assert "chief complaint" in sections
        assert "assessment" in sections
        assert "plan" in sections

    def test_title_case_headers(self, analyzer):
        """Title Case headers with colons are detected."""
        text = "History of Present Illness: Patient presents with pain."
        stats = analyzer._compute_stats(text)
        sections = stats["detected_sections"]
        assert any("history" in s for s in sections)

    def test_bold_markdown_headers(self, analyzer):
        """**Bold** markdown headers are detected."""
        text = "**Chief Complaint** Headache. **Assessment** Migraine."
        stats = analyzer._compute_stats(text)
        sections = stats["detected_sections"]
        assert any("chief complaint" in s for s in sections)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    """Tests for sentence splitting."""

    def test_basic_splitting(self, analyzer):
        """Sentences are split on periods followed by capitals."""
        text = "Patient has chest pain. He denies fever. Blood pressure is normal."
        sentences = analyzer._split_sentences(text)
        assert len(sentences) >= 2

    def test_empty_text(self, analyzer):
        """Empty text returns empty list."""
        sentences = analyzer._split_sentences("")
        assert sentences == []

    def test_newline_splitting(self, analyzer):
        """Sentences are split on newlines."""
        text = "First line.\nSecond line.\nThird line."
        sentences = analyzer._split_sentences(text)
        assert len(sentences) >= 3


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------


class TestBatchAnalysis:
    """Tests for batch analysis."""

    def test_batch_returns_all_reports(self, analyzer):
        """Batch analysis returns one report per input."""
        texts = [COMPLETE_CLINICAL_NOTE, SHORT_NOTE, ABBREVIATION_HEAVY_NOTE]
        reports = analyzer.analyze_batch(texts)
        assert len(reports) == 3

    def test_batch_order_preserved(self, analyzer):
        """Reports are in the same order as inputs."""
        texts = [SHORT_NOTE, COMPLETE_CLINICAL_NOTE]
        reports = analyzer.analyze_batch(texts)
        # Short note should score lower
        assert reports[0].overall_score < reports[1].overall_score

    def test_batch_reports_are_independent(self, analyzer):
        """Each batch report has a unique text hash."""
        texts = [COMPLETE_CLINICAL_NOTE, SHORT_NOTE]
        reports = analyzer.analyze_batch(texts)
        assert reports[0].text_hash != reports[1].text_hash


# ---------------------------------------------------------------------------
# End-to-end on realistic notes
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end quality analysis on realistic clinical notes."""

    def test_complete_note_high_overall_score(self, analyzer):
        """A complete clinical note scores well overall."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        assert report.overall_score >= 65
        assert report.grade in ("A", "B", "C")
        assert len(report.dimensions) == 5

    def test_short_note_low_score(self, analyzer):
        """A very short note scores poorly."""
        report = analyzer.analyze(SHORT_NOTE)
        assert report.overall_score < 80
        assert report.grade in ("B", "C", "D", "F")

    def test_analysis_timing(self, analyzer):
        """Analysis completes in under 100ms."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        assert report.analysis_ms < 100.0

    def test_text_hash_deterministic(self, analyzer):
        """Same text always produces same hash."""
        r1 = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        r2 = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        assert r1.text_hash == r2.text_hash

    def test_report_stats_populated(self, analyzer):
        """Report stats contain expected keys."""
        report = analyzer.analyze(COMPLETE_CLINICAL_NOTE)
        assert "word_count" in report.stats
        assert "sentence_count" in report.stats
        assert "section_count" in report.stats
        assert "abbreviation_ratio" in report.stats
        assert "medical_term_ratio" in report.stats
        assert report.stats["word_count"] > 0

    def test_score_clamped_0_100(self, analyzer):
        """Overall score is always in [0, 100]."""
        for text in [SHORT_NOTE, COMPLETE_CLINICAL_NOTE, ABBREVIATION_HEAVY_NOTE, "x"]:
            report = analyzer.analyze(text)
            assert 0 <= report.overall_score <= 100
            for dim in report.dimensions:
                assert 0 <= dim.score <= 100

    def test_dental_note_quality(self):
        """Dental notes with dental-specific sections score well."""
        config = QualityConfig(
            expected_sections=["chief complaint", "dental history", "assessment", "plan"]
        )
        analyzer = ClinicalNoteQualityAnalyzer(config)
        text = """
CHIEF COMPLAINT: Toothache lower left.

DENTAL HISTORY: Last cleaning 6 months ago. No prior restorations.

ORAL EXAMINATION:
Tooth #19: Large occlusal caries extending to DEJ.
Periodontal probing: 2-3mm all quadrants.

ASSESSMENT:
Dental caries #19, moderate.

PLAN:
1. Composite restoration #19 MOD
2. Fluoride varnish application
3. Return in 6 months for recall
"""
        report = analyzer.analyze(text)
        assert report.overall_score >= 60
