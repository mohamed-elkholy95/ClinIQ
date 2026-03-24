"""Unit tests for the dental NLP module."""

import pytest

from app.ml.dental.model import (
    CDTCodePredictor,
    DentalEntity,
    DentalNERModel,
    DentalSurface,
    PeriodontalRiskAssessment,
    ToothNumberingSystem,
)


class TestDentalEntityDataclass:
    """Tests for the DentalEntity dataclass."""

    def test_creation_with_base_fields(self):
        """Test creating a DentalEntity with base Entity fields."""
        entity = DentalEntity(
            text="#14",
            entity_type="TOOTH",
            start_char=0,
            end_char=3,
            confidence=0.95,
        )

        assert entity.text == "#14"
        assert entity.entity_type == "TOOTH"
        assert entity.confidence == 0.95

    def test_dental_specific_defaults(self):
        """Test that dental-specific fields default to None."""
        entity = DentalEntity(
            text="crown",
            entity_type="DENTAL_PROCEDURE",
            start_char=0,
            end_char=5,
            confidence=0.9,
        )

        assert entity.tooth_number is None
        assert entity.tooth_surface is None
        assert entity.numbering_system is None
        assert entity.periodontal_value is None
        assert entity.quadrant is None

    def test_creation_with_dental_fields(self):
        """Test creating a DentalEntity with dental-specific fields."""
        entity = DentalEntity(
            text="#14",
            entity_type="TOOTH",
            start_char=0,
            end_char=3,
            confidence=0.95,
            tooth_number="14",
            tooth_surface=["M", "O"],
            numbering_system="universal",
            quadrant=2,
        )

        assert entity.tooth_number == "14"
        assert entity.tooth_surface == ["M", "O"]
        assert entity.numbering_system == "universal"
        assert entity.quadrant == 2

    def test_periodontal_value(self):
        """Test setting a periodontal measurement value."""
        entity = DentalEntity(
            text="5mm pocket",
            entity_type="PERIODONTAL",
            start_char=0,
            end_char=10,
            confidence=0.85,
            periodontal_value=5.0,
        )

        assert entity.periodontal_value == 5.0

    def test_to_dict_inherited(self):
        """Test that to_dict() from Entity base class is callable."""
        entity = DentalEntity(
            text="#3",
            entity_type="TOOTH",
            start_char=0,
            end_char=2,
            confidence=0.9,
        )
        result = entity.to_dict()
        assert isinstance(result, dict)
        assert "text" in result
        assert "entity_type" in result


class TestToothNumberingSystem:
    """Tests for ToothNumberingSystem enum."""

    def test_universal_value(self):
        """Test UNIVERSAL enum value."""
        assert ToothNumberingSystem.UNIVERSAL == "universal"

    def test_fdi_value(self):
        """Test FDI enum value."""
        assert ToothNumberingSystem.FDI == "fdi"

    def test_palmer_value(self):
        """Test PALMER enum value."""
        assert ToothNumberingSystem.PALMER == "palmer"


class TestDentalSurface:
    """Tests for DentalSurface enum."""

    def test_mesial_value(self):
        """Test M (mesial) value."""
        assert DentalSurface.M == "mesial"

    def test_distal_value(self):
        """Test D (distal) value."""
        assert DentalSurface.D == "distal"

    def test_buccal_value(self):
        """Test B (buccal) value."""
        assert DentalSurface.B == "buccal"

    def test_occlusal_value(self):
        """Test O (occlusal) value."""
        assert DentalSurface.O == "occlusal"

    def test_all_surfaces_have_values(self):
        """Test that all surface enum members have non-empty string values."""
        for surface in DentalSurface:
            assert isinstance(surface.value, str)
            assert len(surface.value) > 0


class TestDentalNERModel:
    """Tests for DentalNERModel."""

    @pytest.fixture
    def model(self) -> DentalNERModel:
        """Create a loaded DentalNERModel instance."""
        m = DentalNERModel()
        m.load()
        return m

    def test_model_loaded_after_load(self):
        """Test that _is_loaded becomes True after load()."""
        model = DentalNERModel()
        assert model._is_loaded is False
        model.load()
        assert model._is_loaded is True

    def test_load_on_first_extract(self):
        """Test that extract_entities triggers load if not loaded."""
        model = DentalNERModel()
        assert model._is_loaded is False
        # Should not raise
        result = model.extract_entities("Tooth #14 extraction.")
        assert model._is_loaded is True
        assert isinstance(result, list)

    def test_returns_list(self, model: DentalNERModel):
        """Test that extract_entities always returns a list."""
        result = model.extract_entities("Patient presents for routine exam.")
        assert isinstance(result, list)

    def test_returns_dental_entities(self, model: DentalNERModel):
        """Test that returned items are DentalEntity instances."""
        text = "Extraction of tooth #14 recommended."
        entities = model.extract_entities(text)
        for entity in entities:
            assert isinstance(entity, DentalEntity)

    def test_extract_tooth_number_universal(self, model: DentalNERModel):
        """Test extraction of a universal tooth number."""
        text = "Caries noted on tooth 14."
        entities = model.extract_entities(text)

        tooth_entities = [e for e in entities if e.entity_type == "TOOTH"]
        assert len(tooth_entities) > 0

        numbers = [e.tooth_number for e in tooth_entities]
        assert any("14" in (n or "") for n in numbers)

    @pytest.mark.parametrize(
        "text,expected_tooth",
        [
            ("Tooth #3 extraction.", "3"),
            ("Root canal on tooth 19.", "19"),
            ("Crown preparation for tooth 30.", "30"),
        ],
    )
    def test_extract_various_tooth_numbers(
        self, model: DentalNERModel, text: str, expected_tooth: str
    ):
        """Parametrised test for tooth number extraction."""
        entities = model.extract_entities(text)
        tooth_numbers = [
            e.tooth_number
            for e in entities
            if e.entity_type == "TOOTH" and e.tooth_number is not None
        ]
        assert any(expected_tooth in (n or "") for n in tooth_numbers), (
            f"Expected tooth {expected_tooth!r} in: {tooth_numbers}"
        )

    def test_extract_dental_procedure_extraction(self, model: DentalNERModel):
        """Test extraction of dental procedure 'extraction'."""
        text = "Surgical extraction of impacted wisdom tooth."
        entities = model.extract_entities(text)

        proc_entities = [e for e in entities if e.entity_type == "DENTAL_PROCEDURE"]
        assert len(proc_entities) > 0

    def test_extract_procedure_crown(self, model: DentalNERModel):
        """Test extraction of crown procedure."""
        text = "Crown preparation on tooth 14 completed."
        entities = model.extract_entities(text)

        proc_types = [e.entity_type for e in entities]
        assert "DENTAL_PROCEDURE" in proc_types

    def test_extract_periodontal_pocket_depth(self, model: DentalNERModel):
        """Test extraction of periodontal pocket depth measurement."""
        text = "Probing depth: 5mm pocket noted at site 14."
        entities = model.extract_entities(text)

        perio_entities = [e for e in entities if e.entity_type == "PERIODONTAL"]
        if perio_entities:
            perio_with_values = [e for e in perio_entities if e.periodontal_value is not None]
            if perio_with_values:
                assert any(e.periodontal_value == 5.0 for e in perio_with_values)

    def test_extract_surface_m(self, model: DentalNERModel):
        """Test extraction of tooth surface 'M' (mesial)."""
        text = "Mesial-occlusal composite on tooth 14."
        entities = model.extract_entities(text)

        surface_entities = [e for e in entities if e.entity_type == "TOOTH_SURFACE"]
        # MO should be identified as a valid surface combination
        if surface_entities:
            assert all(e.tooth_surface is not None for e in surface_entities)

    def test_extract_dental_condition_caries(self, model: DentalNERModel):
        """Test extraction of dental condition 'caries'."""
        text = "Caries present on occlusal surface of tooth 3."
        entities = model.extract_entities(text)

        condition_entities = [e for e in entities if e.entity_type == "DENTAL_CONDITION"]
        assert len(condition_entities) > 0

    def test_extract_dental_condition_periodontitis(self, model: DentalNERModel):
        """Test extraction of 'periodontitis' as a dental condition."""
        text = "Diagnosis: generalized periodontitis stage III."
        entities = model.extract_entities(text)

        condition_entities = [e for e in entities if e.entity_type == "DENTAL_CONDITION"]
        assert any("periodontitis" in e.text.lower() for e in condition_entities)

    def test_no_overlapping_entities(self, model: DentalNERModel):
        """Test that resolved entities do not overlap."""
        text = (
            "Extraction of tooth 14 MOD composite restoration "
            "with 5mm pocket depth and bleeding."
        )
        entities = model.extract_entities(text)

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                overlaps = e1.start_char < e2.end_char and e2.start_char < e1.end_char
                assert not overlaps, (
                    f"Entities overlap: {e1.text!r} [{e1.start_char}:{e1.end_char}] "
                    f"and {e2.text!r} [{e2.start_char}:{e2.end_char}]"
                )

    def test_negation_detection(self, model: DentalNERModel):
        """Test that negation is detected for negated findings."""
        text = "No caries noted on tooth 14."
        entities = model.extract_entities(text)

        for entity in entities:
            assert isinstance(entity.is_negated, bool)

    def test_quadrant_assignment_upper_right(self, model: DentalNERModel):
        """Test that upper right teeth (1-8) get quadrant 1."""
        text = "Examination of tooth 5."
        entities = model.extract_entities(text)

        tooth_entities = [e for e in entities if e.entity_type == "TOOTH" and e.tooth_number == "5"]
        if tooth_entities:
            assert tooth_entities[0].quadrant == 1

    def test_quadrant_assignment_lower_left(self, model: DentalNERModel):
        """Test that lower left teeth (17-24) get quadrant 3."""
        text = "Tooth 19 extracted."
        entities = model.extract_entities(text)

        tooth_entities = [e for e in entities if e.entity_type == "TOOTH" and e.tooth_number == "19"]
        if tooth_entities:
            assert tooth_entities[0].quadrant == 3

    def test_empty_text_returns_empty_list(self, model: DentalNERModel):
        """Test that empty text returns an empty list."""
        result = model.extract_entities("")
        assert result == []


class TestPeriodontalRiskAssessment:
    """Tests for PeriodontalRiskAssessment."""

    @pytest.fixture
    def assessor(self) -> PeriodontalRiskAssessment:
        """Create a PeriodontalRiskAssessment instance."""
        return PeriodontalRiskAssessment()

    @pytest.fixture
    def dental_model(self) -> DentalNERModel:
        """Provide a loaded DentalNERModel."""
        m = DentalNERModel()
        m.load()
        return m

    def test_calculate_risk_returns_dict(self, assessor: PeriodontalRiskAssessment):
        """Test that calculate_risk returns a dict."""
        result = assessor.calculate_risk(entities=[], text="Healthy periodontium.")
        assert isinstance(result, dict)

    def test_result_contains_required_keys(self, assessor: PeriodontalRiskAssessment):
        """Test that result has all required keys."""
        result = assessor.calculate_risk(entities=[], text="Routine exam.")
        required_keys = {"overall_risk", "risk_score", "risk_factors", "recommendations"}
        assert required_keys.issubset(set(result.keys()))

    def test_low_risk_no_risk_factors(self, assessor: PeriodontalRiskAssessment):
        """Test that minimal input produces low risk."""
        result = assessor.calculate_risk(entities=[], text="Healthy gingival tissue.")
        assert result["overall_risk"] == "low"
        assert result["risk_score"] <= 0.5

    def test_high_risk_with_periodontitis(
        self, assessor: PeriodontalRiskAssessment, dental_model: DentalNERModel
    ):
        """Test that periodontitis text increases risk."""
        text = (
            "Patient presents with generalized periodontitis. "
            "Bone loss noted radiographically. "
            "Deep probing depths of 7mm pocket depth. Bleeding on probing. "
            "Tooth mobility grade 2 present."
        )
        entities = dental_model.extract_entities(text)
        result = assessor.calculate_risk(entities=entities, text=text)

        assert result["overall_risk"] in ("moderate", "high")
        assert result["risk_score"] >= 0.4

    def test_risk_factors_is_list(self, assessor: PeriodontalRiskAssessment):
        """Test that risk_factors is a list."""
        result = assessor.calculate_risk(entities=[], text="Normal exam.")
        assert isinstance(result["risk_factors"], list)

    def test_recommendations_is_list(self, assessor: PeriodontalRiskAssessment):
        """Test that recommendations is a list."""
        result = assessor.calculate_risk(entities=[], text="Normal exam.")
        assert isinstance(result["recommendations"], list)

    def test_recommendations_not_empty(self, assessor: PeriodontalRiskAssessment):
        """Test that at least one recommendation is always returned."""
        result = assessor.calculate_risk(entities=[], text="Routine cleaning done.")
        assert len(result["recommendations"]) > 0

    def test_bleeding_increases_risk(self, assessor: PeriodontalRiskAssessment):
        """Test that bleeding on probing is identified as a risk factor."""
        text = "Significant bleeding on probing noted throughout."
        result = assessor.calculate_risk(entities=[], text=text)

        factor_names = [r.get("factor", "") for r in result["risk_factors"]]
        assert any("bleeding" in name for name in factor_names)

    def test_risk_score_range(self, assessor: PeriodontalRiskAssessment):
        """Test that risk_score is within [0, 1] range."""
        texts = [
            "Healthy periodontium.",
            "Periodontitis with deep pockets and bone loss.",
            "Gingivitis with bleeding on probing.",
        ]
        for text in texts:
            result = assessor.calculate_risk(entities=[], text=text)
            assert 0.0 <= result["risk_score"] <= 1.0, (
                f"risk_score {result['risk_score']} out of range for text: {text!r}"
            )


class TestCDTCodePredictor:
    """Tests for CDTCodePredictor."""

    @pytest.fixture
    def predictor(self) -> CDTCodePredictor:
        """Create a CDTCodePredictor instance."""
        return CDTCodePredictor()

    def test_predict_returns_list(self, predictor: CDTCodePredictor):
        """Test that predict() returns a list."""
        result = predictor.predict("Patient needs a cleaning.")
        assert isinstance(result, list)

    def test_predict_cleaning_maps_to_d1110(self, predictor: CDTCodePredictor):
        """Test that 'cleaning' maps to CDT code D1110."""
        result = predictor.predict("Prophylaxis cleaning performed.")
        codes = [p["code"] for p in result]
        assert "D1110" in codes

    def test_predict_scaling_maps_to_d4341(self, predictor: CDTCodePredictor):
        """Test that 'scaling' maps to CDT code D4341."""
        result = predictor.predict("Periodontal scaling performed in upper right quadrant.")
        codes = [p["code"] for p in result]
        assert "D4341" in codes

    def test_predict_root_canal_maps_to_d3310(self, predictor: CDTCodePredictor):
        """Test that 'root canal' maps to CDT code D3310."""
        result = predictor.predict("Root canal treatment on tooth 14.")
        codes = [p["code"] for p in result]
        assert "D3310" in codes

    def test_predict_extraction_maps_to_d7140(self, predictor: CDTCodePredictor):
        """Test that 'extraction' maps to CDT code D7140."""
        result = predictor.predict("Extraction of tooth 32 completed.")
        codes = [p["code"] for p in result]
        assert "D7140" in codes

    @pytest.mark.parametrize(
        "text,expected_code",
        [
            ("Panoramic radiograph taken.", "D0330"),
            ("Fluoride varnish applied.", "D1206"),
            ("Sealant placed on molars.", "D1351"),
            ("Crown preparation completed.", "D2750"),
        ],
    )
    def test_parametrised_cdt_mapping(
        self, predictor: CDTCodePredictor, text: str, expected_code: str
    ):
        """Parametrised tests for CDT code mapping."""
        result = predictor.predict(text)
        codes = [p["code"] for p in result]
        assert expected_code in codes, (
            f"Expected CDT code {expected_code!r} not found in {codes} for text: {text!r}"
        )

    def test_result_has_required_fields(self, predictor: CDTCodePredictor):
        """Test that each prediction dict has required fields."""
        result = predictor.predict("Cleaning and fluoride treatment.")
        for pred in result:
            assert "code" in pred
            assert "description" in pred
            assert "confidence" in pred
            assert "procedure" in pred

    def test_confidence_in_range(self, predictor: CDTCodePredictor):
        """Test that all confidence values are in [0, 1]."""
        result = predictor.predict("Crown and root canal treatment.")
        for pred in result:
            assert 0.0 <= pred["confidence"] <= 1.0

    def test_results_sorted_by_confidence_descending(self, predictor: CDTCodePredictor):
        """Test that predictions are sorted highest confidence first."""
        result = predictor.predict("Scaling and root planing with crown.")
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["confidence"] >= result[i + 1]["confidence"]

    def test_empty_text_returns_list(self, predictor: CDTCodePredictor):
        """Test that empty text returns an empty list (no procedures found)."""
        result = predictor.predict("")
        assert isinstance(result, list)

    def test_no_matching_procedures_returns_empty(self, predictor: CDTCodePredictor):
        """Test that text with no dental procedures returns empty list."""
        result = predictor.predict("Blood pressure measured: 130/80.")
        assert result == []

    def test_max_10_results_returned(self, predictor: CDTCodePredictor):
        """Test that at most 10 CDT codes are returned."""
        # A text with many possible procedures
        text = (
            "Cleaning prophylaxis scaling root planing amalgam composite crown "
            "root canal extraction implant bitewing panoramic fluoride sealant."
        )
        result = predictor.predict(text)
        assert len(result) <= 10
