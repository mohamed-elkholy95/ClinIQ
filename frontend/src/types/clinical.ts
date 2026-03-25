/**
 * Type definitions for ClinIQ's specialized clinical NLP modules.
 *
 * These types mirror the backend Pydantic/dataclass schemas for the 14
 * clinical extraction modules added in post-PRD sessions 20–34.  Each
 * module has its own request/response pair, enabling the frontend to
 * consume any endpoint with full type safety.
 */

// ─── Medication Extraction ───────────────────────────────────

export type MedicationStatus =
  | 'active'
  | 'discontinued'
  | 'held'
  | 'new'
  | 'changed'
  | 'allergic'
  | 'unknown';

export interface MedicationResult {
  drug_name: string;
  generic_name: string | null;
  brand_names: string[];
  dosage: string | null;
  route: string | null;
  frequency: string | null;
  duration: string | null;
  indication: string | null;
  prn: boolean;
  status: MedicationStatus;
  confidence: number;
  start_char: number;
  end_char: number;
}

export interface MedicationExtractionResponse {
  medications: MedicationResult[];
  count: number;
  processing_time_ms: number;
}

// ─── Allergy Extraction ──────────────────────────────────────

export type AllergyCategory = 'drug' | 'food' | 'environmental';

export type AllergySeverity =
  | 'mild'
  | 'moderate'
  | 'severe'
  | 'life_threatening'
  | 'unknown';

export type AssertionStatus =
  | 'present'
  | 'absent'
  | 'possible'
  | 'conditional'
  | 'hypothetical'
  | 'family'
  | 'historical'
  | 'tolerated';

export interface AllergyResult {
  allergen: string;
  canonical_name: string;
  category: AllergyCategory;
  reactions: string[];
  severity: AllergySeverity;
  assertion_status: AssertionStatus;
  confidence: number;
  start_char: number;
  end_char: number;
}

export interface AllergyExtractionResponse {
  allergies: AllergyResult[];
  nkda_detected: boolean;
  count: number;
  processing_time_ms: number;
}

// ─── Vital Signs Extraction ──────────────────────────────────

export type VitalType =
  | 'blood_pressure'
  | 'heart_rate'
  | 'temperature'
  | 'respiratory_rate'
  | 'oxygen_saturation'
  | 'weight'
  | 'height'
  | 'bmi'
  | 'pain_scale';

export type ClinicalInterpretation =
  | 'normal'
  | 'low'
  | 'high'
  | 'critical_low'
  | 'critical_high';

export interface VitalSignResult {
  type: VitalType;
  value: number;
  unit: string;
  interpretation: ClinicalInterpretation;
  trend: string | null;
  confidence: number;
  start_char: number;
  end_char: number;
  /** Diastolic value for blood_pressure type. */
  diastolic?: number;
  /** Mean Arterial Pressure for blood_pressure type. */
  map?: number;
}

export interface VitalExtractionResponse {
  vitals: VitalSignResult[];
  count: number;
  processing_time_ms: number;
}

// ─── Section Parsing ─────────────────────────────────────────

export interface SectionResult {
  category: string;
  header_text: string;
  body_text: string;
  header_start: number;
  header_end: number;
  body_end: number;
  confidence: number;
}

export interface SectionParseResponse {
  sections: SectionResult[];
  count: number;
  processing_time_ms: number;
}

// ─── Abbreviation Expansion ──────────────────────────────────

export interface AbbreviationResult {
  abbreviation: string;
  expansion: string;
  domain: string;
  is_ambiguous: boolean;
  confidence: number;
  start_char: number;
  end_char: number;
}

export interface AbbreviationExpansionResponse {
  abbreviations: AbbreviationResult[];
  expanded_text: string | null;
  count: number;
  processing_time_ms: number;
}

// ─── De-identification ───────────────────────────────────────

export type PHIType =
  | 'NAME'
  | 'DATE'
  | 'PHONE'
  | 'EMAIL'
  | 'SSN'
  | 'MRN'
  | 'URL'
  | 'IP_ADDRESS'
  | 'ZIP_CODE'
  | 'AGE_OVER_90'
  | 'ACCOUNT_NUMBER'
  | 'LICENSE_NUMBER';

export type ReplacementStrategy = 'redact' | 'mask' | 'surrogate';

export interface PHIDetection {
  type: PHIType;
  text: string;
  replacement: string;
  start_char: number;
  end_char: number;
  confidence: number;
}

export interface DeidentifyResponse {
  deidentified_text: string;
  detections: PHIDetection[];
  count: number;
  strategy: ReplacementStrategy;
  processing_time_ms: number;
}

// ─── Document Classification ─────────────────────────────────

export type DocumentType =
  | 'discharge_summary'
  | 'progress_note'
  | 'history_physical'
  | 'operative_note'
  | 'consultation_note'
  | 'radiology_report'
  | 'pathology_report'
  | 'laboratory_report'
  | 'nursing_note'
  | 'emergency_note'
  | 'dental_note'
  | 'prescription'
  | 'referral'
  | 'unknown';

export interface ClassificationScore {
  document_type: DocumentType;
  score: number;
  evidence: string[];
}

export interface ClassificationResponse {
  predicted_type: DocumentType;
  scores: ClassificationScore[];
  confidence: number;
  processing_time_ms: number;
}

// ─── Quality Analysis ────────────────────────────────────────

export type QualityGrade = 'A' | 'B' | 'C' | 'D' | 'F';
export type FindingSeverity = 'critical' | 'warning' | 'info';

export interface QualityFinding {
  message: string;
  severity: FindingSeverity;
  dimension: string;
}

export interface QualityDimensionScore {
  dimension: string;
  score: number;
  weight: number;
  findings: QualityFinding[];
}

export interface QualityReport {
  overall_score: number;
  grade: QualityGrade;
  dimensions: QualityDimensionScore[];
  recommendations: string[];
  processing_time_ms?: number;
  analysis_time_ms?: number;
}

// ─── SDoH Extraction ─────────────────────────────────────────

export type SDoHDomain =
  | 'housing'
  | 'employment'
  | 'education'
  | 'food_security'
  | 'transportation'
  | 'social_support'
  | 'substance_use'
  | 'financial';

export type SDoHSentiment = 'adverse' | 'protective' | 'neutral';

export interface SDoHFinding {
  domain: SDoHDomain;
  trigger_text: string;
  /** Raw matched text from the note (may include surrounding context). */
  matched_text?: string;
  sentiment: SDoHSentiment;
  z_code: string | null;
  confidence: number;
  start_char: number;
  end_char: number;
}

export interface SDoHExtractionResponse {
  findings: SDoHFinding[];
  domains_detected: SDoHDomain[];
  adverse_count: number;
  protective_count: number;
  processing_time_ms: number;
}

// ─── Comorbidity (Charlson) ──────────────────────────────────

export type RiskGroup = 'low' | 'mild' | 'moderate' | 'severe';

export interface ComorbidityCategory {
  name: string;
  weight: number;
  detected: boolean;
  source: 'icd_code' | 'text_extraction' | 'both';
  /** Human-readable description of the disease category. */
  description?: string;
  /** ICD-10-CM codes that matched this category. */
  matched_codes?: string[];
  /** Confidence score for text-based detection. */
  confidence?: number;
}

export interface ComorbidityResult {
  /** Raw CCI score (sum of category weights). */
  total_score: number;
  /** Alias for total_score used by some backend serialisations. */
  score?: number;
  age_adjusted_score: number | null;
  risk_group: RiskGroup;
  /** 10-year mortality estimate via Charlson exponential survival. */
  estimated_mortality: number | null;
  ten_year_mortality?: number;
  categories: ComorbidityCategory[];
  processing_time_ms?: number;
}

// ─── Concept Normalization ───────────────────────────────────

export type MatchType = 'exact' | 'alias' | 'fuzzy' | 'none';

export interface NormalizationResult {
  input_text: string;
  matched: boolean;
  match_type: MatchType;
  preferred_term: string | null;
  cui: string | null;
  codes: Record<string, string>;
  confidence: number;
}

// ─── Assertion Detection ─────────────────────────────────────

export type AssertionType =
  | 'present'
  | 'absent'
  | 'possible'
  | 'conditional'
  | 'hypothetical'
  | 'family';

export interface AssertionResult {
  status: AssertionType;
  trigger_text: string | null;
  confidence: number;
  scope_start: number;
  scope_end: number;
}

// ─── Relation Extraction ─────────────────────────────────────

export type RelationType =
  | 'treats'
  | 'causes'
  | 'diagnoses'
  | 'contraindicates'
  | 'administered_for'
  | 'dosage_of'
  | 'location_of'
  | 'result_of'
  | 'worsens'
  | 'prevents'
  | 'monitors'
  | 'side_effect_of';

export interface RelationResult {
  subject: string;
  object: string;
  relation_type: RelationType;
  confidence: number;
  evidence: string;
}

export interface RelationExtractionResponse {
  relations: RelationResult[];
  count: number;
  processing_time_ms: number;
}

// ─── Temporal Extraction ─────────────────────────────────────

export type TemporalType =
  | 'date'
  | 'duration'
  | 'relative_time'
  | 'age'
  | 'postoperative_day'
  | 'frequency';

export interface TemporalExpression {
  type: TemporalType;
  text: string;
  normalized_value: string | null;
  confidence: number;
  start_char: number;
  end_char: number;
}

export interface TemporalExtractionResponse {
  expressions: TemporalExpression[];
  count: number;
  processing_time_ms: number;
}

// ─── Enhanced Pipeline (all-in-one) ──────────────────────────

export interface EnhancedAnalysisConfig {
  enable_classification?: boolean;
  enable_sections?: boolean;
  enable_quality?: boolean;
  enable_deidentification?: boolean;
  enable_abbreviations?: boolean;
  enable_medications?: boolean;
  enable_allergies?: boolean;
  enable_vitals?: boolean;
  enable_temporal?: boolean;
  enable_assertions?: boolean;
  enable_normalization?: boolean;
  enable_sdoh?: boolean;
  enable_relations?: boolean;
  enable_comorbidity?: boolean;
}

export interface EnhancedAnalysisResponse {
  classification: ClassificationResponse | null;
  sections: SectionParseResponse | null;
  quality: QualityReport | null;
  deidentification: DeidentifyResponse | null;
  abbreviations: AbbreviationExpansionResponse | null;
  medications: MedicationExtractionResponse | null;
  allergies: AllergyExtractionResponse | null;
  vitals: VitalExtractionResponse | null;
  temporal: TemporalExtractionResponse | null;
  sdoh: SDoHExtractionResponse | null;
  comorbidity: ComorbidityResult | null;
  relations: RelationExtractionResponse | null;
  component_errors: Record<string, string>;
  processing_time_ms: number;
}

// ─── Search ──────────────────────────────────────────────────

export interface SearchHit {
  document_id: string;
  score: number;
  snippet: string;
  title: string | null;
}

export interface SearchResponse {
  hits: SearchHit[];
  total: number;
  query_expansion?: {
    original_query: string;
    expanded_terms: string[];
  };
  reranked: boolean;
  processing_time_ms: number;
}

// ─── Drift Monitoring ────────────────────────────────────────

export type DriftStatus = 'stable' | 'warning' | 'drifted';

export interface DriftStatusResponse {
  overall_status: DriftStatus;
  text_distribution_psi: number;
  model_drift: Record<string, DriftStatus>;
  last_updated: string;
}

// ─── Module Catalogue ────────────────────────────────────────

export interface ModuleInfo {
  name: string;
  description: string;
  enabled_by_default: boolean;
}
