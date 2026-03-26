"""ClinIQ SDK data models.

Typed dataclasses representing API response objects for all ClinIQ
endpoints.  Each model provides a ``from_dict`` class method for safe
deserialization from raw JSON dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core analysis models
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """Extracted medical entity."""

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


@dataclass
class ICDPrediction:
    """ICD-10 code prediction."""

    code: str
    description: str | None
    confidence: float
    chapter: str | None = None
    contributing_text: list[str] | None = None


@dataclass
class Summary:
    """Clinical text summary."""

    summary: str
    key_findings: list[str] = field(default_factory=list)
    detail_level: str = "standard"
    word_count: int = 0


@dataclass
class RiskFactor:
    """Risk assessment factor."""

    name: str
    score: float
    weight: float
    category: str
    description: str = ""


@dataclass
class RiskAssessment:
    """Patient risk assessment."""

    overall_score: float
    risk_level: str
    factors: list[RiskFactor] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Full pipeline analysis result."""

    entities: list[Entity] = field(default_factory=list)
    icd_predictions: list[ICDPrediction] = field(default_factory=list)
    summary: Summary | None = None
    risk_assessment: RiskAssessment | None = None
    processing_time_ms: float = 0.0
    model_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        """Create from API response dict."""
        entities = [Entity(**e) for e in (data.get("entities") or [])]
        icd_preds = [ICDPrediction(**p) for p in (data.get("icd_predictions") or [])]
        summary = None
        if data.get("summary"):
            summary = Summary(**data["summary"])
        risk = None
        if data.get("risk_assessment"):
            ra = data["risk_assessment"]
            factors = [RiskFactor(**f) for f in (ra.get("factors") or [])]
            risk = RiskAssessment(
                overall_score=ra["overall_score"],
                risk_level=ra["risk_level"],
                factors=factors,
                recommendations=ra.get("recommendations", []),
            )
        return cls(
            entities=entities,
            icd_predictions=icd_preds,
            summary=summary,
            risk_assessment=risk,
            processing_time_ms=data.get("processing_time_ms", 0),
            model_versions=data.get("model_versions", {}),
        )


@dataclass
class BatchJob:
    """Batch processing job status."""

    job_id: str
    status: str
    total_documents: int
    processed_documents: int = 0
    failed_documents: int = 0
    progress: float = 0.0
    result_file: str | None = None


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


@dataclass
class ClassificationScore:
    """Single document type score."""

    document_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Document classification result."""

    predicted_type: str
    scores: list[ClassificationScore] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        """Create from API response dict."""
        scores = [
            ClassificationScore(**s) for s in (data.get("scores") or [])
        ]
        return cls(
            predicted_type=data.get("predicted_type", "unknown"),
            scores=scores,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Medication extraction
# ---------------------------------------------------------------------------


@dataclass
class Medication:
    """Extracted medication with components."""

    drug_name: str
    generic_name: str | None = None
    dosage: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    prn: bool = False
    status: str = "active"
    confidence: float = 0.0


@dataclass
class MedicationResult:
    """Medication extraction result."""

    medication_count: int
    medications: list[Medication] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MedicationResult:
        """Create from API response dict."""
        meds = [Medication(**m) for m in (data.get("medications") or [])]
        return cls(
            medication_count=data.get("medication_count", len(meds)),
            medications=meds,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Allergy extraction
# ---------------------------------------------------------------------------


@dataclass
class Allergy:
    """Extracted allergen with reactions."""

    allergen: str
    category: str = "unknown"
    reactions: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "unknown"
    status: str = "active"
    confidence: float = 0.0


@dataclass
class AllergyResult:
    """Allergy extraction result."""

    allergy_count: int
    no_known_allergies: bool = False
    allergies: list[Allergy] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AllergyResult:
        """Create from API response dict."""
        allergies = [Allergy(**a) for a in (data.get("allergies") or [])]
        return cls(
            allergy_count=data.get("allergy_count", len(allergies)),
            no_known_allergies=data.get("no_known_allergies", False),
            allergies=allergies,
        )


# ---------------------------------------------------------------------------
# Vital signs extraction
# ---------------------------------------------------------------------------


@dataclass
class VitalSign:
    """Extracted vital sign measurement."""

    vital_type: str
    value: float
    unit: str
    interpretation: str = "normal"
    confidence: float = 0.0


@dataclass
class VitalSignResult:
    """Vital signs extraction result."""

    vital_count: int
    vitals: list[VitalSign] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VitalSignResult:
        """Create from API response dict."""
        vitals = [VitalSign(**v) for v in (data.get("vitals") or [])]
        return cls(
            vital_count=data.get("vital_count", len(vitals)),
            vitals=vitals,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """Parsed document section."""

    category: str
    header: str
    header_normalised: str = ""
    header_start: int = 0
    header_end: int = 0
    body_end: int = 0
    confidence: float = 0.0


@dataclass
class SectionResult:
    """Section parsing result."""

    section_count: int
    sections: list[Section] = field(default_factory=list)
    categories_found: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionResult:
        """Create from API response dict."""
        sections = [Section(**s) for s in (data.get("sections") or [])]
        return cls(
            section_count=data.get("section_count", len(sections)),
            sections=sections,
            categories_found=data.get("categories_found", []),
        )


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------


@dataclass
class AbbreviationMatch:
    """Detected abbreviation with expansion."""

    abbreviation: str
    expansion: str
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    domain: str = "general"
    is_ambiguous: bool = False


@dataclass
class AbbreviationResult:
    """Abbreviation expansion result."""

    total_found: int
    expanded_text: str = ""
    matches: list[AbbreviationMatch] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbbreviationResult:
        """Create from API response dict."""
        matches = [AbbreviationMatch(**m) for m in (data.get("matches") or [])]
        return cls(
            total_found=data.get("total_found", len(matches)),
            expanded_text=data.get("expanded_text", ""),
            matches=matches,
        )


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------


@dataclass
class QualityDimension:
    """Quality score for a single dimension."""

    dimension: str
    score: float
    weight: float
    finding_count: int = 0


@dataclass
class QualityReport:
    """Clinical note quality report."""

    overall_score: float
    grade: str
    dimensions: list[QualityDimension] = field(default_factory=list)
    recommendation_count: int = 0
    top_recommendations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        """Create from API response dict."""
        dims = [QualityDimension(**d) for d in (data.get("dimensions") or [])]
        return cls(
            overall_score=data.get("overall_score", 0),
            grade=data.get("grade", "F"),
            dimensions=dims,
            recommendation_count=data.get("recommendation_count", 0),
            top_recommendations=data.get("top_recommendations", []),
        )


# ---------------------------------------------------------------------------
# Assertion detection
# ---------------------------------------------------------------------------


@dataclass
class AssertionResult:
    """Assertion status for an entity."""

    entity_text: str
    entity_type: str = ""
    status: str = "present"
    confidence: float = 0.0
    trigger_text: str | None = None


# ---------------------------------------------------------------------------
# Concept normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Normalized entity with ontology codes."""

    entity_text: str
    entity_type: str = ""
    cui: str | None = None
    preferred_term: str | None = None
    match_type: str = "exact"
    confidence: float = 0.0
    snomed_code: str | None = None
    rxnorm_code: str | None = None
    icd10_code: str | None = None
    loinc_code: str | None = None


# ---------------------------------------------------------------------------
# SDoH extraction
# ---------------------------------------------------------------------------


@dataclass
class SDoHExtraction:
    """Social Determinant of Health extraction."""

    domain: str
    text: str
    sentiment: str = "adverse"
    confidence: float = 0.0
    z_codes: list[str] = field(default_factory=list)


@dataclass
class SDoHResult:
    """SDoH extraction result."""

    extraction_count: int
    adverse_count: int = 0
    protective_count: int = 0
    domain_summary: dict[str, int] = field(default_factory=dict)
    extractions: list[SDoHExtraction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SDoHResult:
        """Create from API response dict."""
        extractions = [SDoHExtraction(**e) for e in (data.get("extractions") or [])]
        return cls(
            extraction_count=data.get("extraction_count", len(extractions)),
            adverse_count=data.get("adverse_count", 0),
            protective_count=data.get("protective_count", 0),
            domain_summary=data.get("domain_summary", {}),
            extractions=extractions,
        )


# ---------------------------------------------------------------------------
# Comorbidity scoring
# ---------------------------------------------------------------------------


@dataclass
class MatchedCategory:
    """Matched comorbidity category."""

    category: str
    weight: int
    source: str = ""
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class ComorbidityResult:
    """Charlson Comorbidity Index result."""

    raw_score: int
    age_adjusted_score: int | None = None
    risk_group: str = "low"
    ten_year_mortality: float = 0.0
    category_count: int = 0
    matched_categories: list[MatchedCategory] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComorbidityResult:
        """Create from API response dict."""
        cats = [MatchedCategory(**c) for c in (data.get("matched_categories") or [])]
        return cls(
            raw_score=data.get("raw_score", 0),
            age_adjusted_score=data.get("age_adjusted_score"),
            risk_group=data.get("risk_group", "low"),
            ten_year_mortality=data.get("ten_year_mortality", 0),
            category_count=data.get("category_count", len(cats)),
            matched_categories=cats,
        )


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------


@dataclass
class Relation:
    """Extracted clinical relation."""

    subject: str
    subject_type: str
    object: str
    object_type: str
    relation_type: str
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class RelationResult:
    """Relation extraction result."""

    relation_count: int
    pair_count: int = 0
    relations: list[Relation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationResult:
        """Create from API response dict."""
        rels = [Relation(**r) for r in (data.get("relations") or [])]
        return cls(
            relation_count=data.get("relation_count", len(rels)),
            pair_count=data.get("pair_count", 0),
            relations=rels,
        )


# ---------------------------------------------------------------------------
# Enhanced pipeline result
# ---------------------------------------------------------------------------


@dataclass
class EnhancedAnalysisResult:
    """Full enhanced pipeline analysis result."""

    base_result: AnalysisResult | None = None
    classification: dict[str, Any] | None = None
    sections: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    deidentification: dict[str, Any] | None = None
    abbreviations: dict[str, Any] | None = None
    medications: dict[str, Any] | None = None
    allergies: dict[str, Any] | None = None
    vitals: dict[str, Any] | None = None
    temporal: dict[str, Any] | None = None
    assertions: list[dict[str, Any]] | None = None
    normalization: list[dict[str, Any]] | None = None
    sdoh: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    comorbidity: dict[str, Any] | None = None
    processing_time_ms: float = 0.0
    component_errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhancedAnalysisResult:
        """Create from API response dict."""
        base = None
        if data.get("base_result"):
            base = AnalysisResult.from_dict(data["base_result"])
        return cls(
            base_result=base,
            classification=data.get("classification"),
            sections=data.get("sections"),
            quality=data.get("quality"),
            deidentification=data.get("deidentification"),
            abbreviations=data.get("abbreviations"),
            medications=data.get("medications"),
            allergies=data.get("allergies"),
            vitals=data.get("vitals"),
            temporal=data.get("temporal"),
            assertions=data.get("assertions"),
            normalization=data.get("normalization"),
            sdoh=data.get("sdoh"),
            relations=data.get("relations"),
            comorbidity=data.get("comorbidity"),
            processing_time_ms=data.get("processing_time_ms", 0),
            component_errors=data.get("component_errors", {}),
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


@dataclass
class ClassificationEvalResult:
    """Binary classification evaluation result."""

    mcc: float
    tp: int
    fp: int
    fn: int
    tn: int
    calibration: dict[str, Any] | None = None
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationEvalResult:
        """Create from API response dict."""
        return cls(
            mcc=data.get("mcc", 0.0),
            tp=data.get("tp", 0),
            fp=data.get("fp", 0),
            fn=data.get("fn", 0),
            tn=data.get("tn", 0),
            calibration=data.get("calibration"),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class KappaResult:
    """Cohen's Kappa inter-annotator agreement result."""

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_items: int
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KappaResult:
        """Create from API response dict."""
        return cls(
            kappa=data.get("kappa", 0.0),
            observed_agreement=data.get("observed_agreement", 0.0),
            expected_agreement=data.get("expected_agreement", 0.0),
            n_items=data.get("n_items", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class NEREvalResult:
    """NER partial span matching evaluation result."""

    exact_f1: float
    partial_f1: float
    type_weighted_f1: float
    mean_overlap: float
    n_gold: int
    n_pred: int
    n_exact_matches: int
    n_partial_matches: int
    n_unmatched_pred: int
    n_unmatched_gold: int
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NEREvalResult:
        """Create from API response dict."""
        return cls(
            exact_f1=data.get("exact_f1", 0.0),
            partial_f1=data.get("partial_f1", 0.0),
            type_weighted_f1=data.get("type_weighted_f1", 0.0),
            mean_overlap=data.get("mean_overlap", 0.0),
            n_gold=data.get("n_gold", 0),
            n_pred=data.get("n_pred", 0),
            n_exact_matches=data.get("n_exact_matches", 0),
            n_partial_matches=data.get("n_partial_matches", 0),
            n_unmatched_pred=data.get("n_unmatched_pred", 0),
            n_unmatched_gold=data.get("n_unmatched_gold", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class ROUGEScores:
    """ROUGE score for a single variant (1, 2, or L)."""

    precision: float
    recall: float
    f1: float


@dataclass
class ROUGEEvalResult:
    """ROUGE summarisation evaluation result."""

    rouge1: ROUGEScores
    rouge2: ROUGEScores
    rougeL: ROUGEScores
    reference_length: int
    hypothesis_length: int
    length_ratio: float
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ROUGEEvalResult:
        """Create from API response dict."""
        def _scores(d: dict) -> ROUGEScores:
            return ROUGEScores(
                precision=d.get("precision", 0.0),
                recall=d.get("recall", 0.0),
                f1=d.get("f1", 0.0),
            )
        return cls(
            rouge1=_scores(data.get("rouge1", {})),
            rouge2=_scores(data.get("rouge2", {})),
            rougeL=_scores(data.get("rougeL", {})),
            reference_length=data.get("reference_length", 0),
            hypothesis_length=data.get("hypothesis_length", 0),
            length_ratio=data.get("length_ratio", 0.0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class ICDEvalResult:
    """Hierarchical ICD-10 evaluation result."""

    full_code_accuracy: float
    block_accuracy: float
    chapter_accuracy: float
    n_samples: int
    full_code_matches: int
    block_matches: int
    chapter_matches: int
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ICDEvalResult:
        """Create from API response dict."""
        return cls(
            full_code_accuracy=data.get("full_code_accuracy", 0.0),
            block_accuracy=data.get("block_accuracy", 0.0),
            chapter_accuracy=data.get("chapter_accuracy", 0.0),
            n_samples=data.get("n_samples", 0),
            full_code_matches=data.get("full_code_matches", 0),
            block_matches=data.get("block_matches", 0),
            chapter_matches=data.get("chapter_matches", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


@dataclass
class AUPRCResult:
    """Area Under Precision-Recall Curve result."""

    label: str
    auprc: float
    n_positive: int
    n_total: int
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AUPRCResult:
        """Create from API response dict."""
        return cls(
            label=data.get("label", "positive"),
            auprc=data.get("auprc", 0.0),
            n_positive=data.get("n_positive", 0),
            n_total=data.get("n_total", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Conversation memory models
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurnResult:
    """Result of adding a conversation turn."""

    session_id: str
    turn_id: int
    turn_count: int


@dataclass
class ConversationContext:
    """Aggregated conversation context."""

    session_id: str
    turn_count: int
    unique_entities: list[str] = field(default_factory=list)
    unique_icd_codes: list[str] = field(default_factory=list)
    overall_risk_trend: list[float] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        """Create from API response dict."""
        return cls(
            session_id=data.get("session_id", ""),
            turn_count=data.get("turn_count", 0),
            unique_entities=data.get("unique_entities", []),
            unique_icd_codes=data.get("unique_icd_codes", []),
            overall_risk_trend=data.get("overall_risk_trend", []),
            context=data.get("context", []),
        )


@dataclass
class ConversationStats:
    """Memory usage statistics."""

    active_sessions: int
    total_turns: int
    max_turns_per_session: int
    session_ttl_seconds: float
    max_sessions: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationStats:
        """Create from API response dict."""
        return cls(
            active_sessions=data.get("active_sessions", 0),
            total_turns=data.get("total_turns", 0),
            max_turns_per_session=data.get("max_turns_per_session", 50),
            session_ttl_seconds=data.get("session_ttl_seconds", 7200.0),
            max_sessions=data.get("max_sessions", 5000),
        )


@dataclass
class ConversationSessionInfo:
    """Summary of an active conversation session."""

    session_id: str
    turn_count: int
    oldest_turn_id: int
    newest_turn_id: int
    last_access: str


@dataclass
class SearchHit:
    """Single search result."""

    document_id: str
    score: float
    snippet: str = ""
    title: str = ""


@dataclass
class SearchResult:
    """Document search result."""

    hits: list[SearchHit] = field(default_factory=list)
    total: int = 0
    query_expansion: dict[str, Any] | None = None
    reranked: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create from API response dict."""
        hits = [SearchHit(**h) for h in (data.get("hits") or [])]
        return cls(
            hits=hits,
            total=data.get("total", len(hits)),
            query_expansion=data.get("query_expansion"),
            reranked=data.get("reranked", False),
        )
