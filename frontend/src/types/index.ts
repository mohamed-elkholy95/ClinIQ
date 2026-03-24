// ─── Entity Types ────────────────────────────────────────────
export type EntityType =
  | 'disease'
  | 'medication'
  | 'procedure'
  | 'anatomy'
  | 'symptom'
  | 'lab_value';

export interface Entity {
  id: string;
  text: string;
  type: EntityType;
  start: number;
  end: number;
  confidence: number;
  normalized?: string;
  cui?: string; // UMLS Concept Unique Identifier
}

// ─── ICD-10 Types ────────────────────────────────────────────
export interface ICDPrediction {
  code: string;
  description: string;
  confidence: number;
  chapter: string;
  category: string;
  evidence: string[];
}

// ─── Risk Assessment ─────────────────────────────────────────
export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

export interface RiskFactor {
  name: string;
  score: number;
  category: string;
  description: string;
}

export interface RiskAssessment {
  overall_score: number;
  level: RiskLevel;
  factors: RiskFactor[];
  recommendations: string[];
  category_scores: Record<string, number>;
}

// ─── Clinical Summary ────────────────────────────────────────
export type SummaryDetail = 'brief' | 'standard' | 'detailed';

export interface ClinicalSummary {
  summary: string;
  key_findings: string[];
  detail_level: SummaryDetail;
  word_count: number;
  generated_at: string;
}

// ─── Analysis Result (composite) ─────────────────────────────
export interface AnalysisResult {
  id: string;
  text: string;
  entities: Entity[];
  icd_predictions: ICDPrediction[];
  summary: ClinicalSummary;
  risk_assessment: RiskAssessment;
  processing_time_ms: number;
  created_at: string;
}

// ─── Document ────────────────────────────────────────────────
export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Document {
  id: string;
  title: string;
  text: string;
  status: DocumentStatus;
  result?: AnalysisResult;
  created_at: string;
  updated_at: string;
}

// ─── Batch Job ───────────────────────────────────────────────
export type BatchStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface BatchJob {
  id: string;
  name: string;
  status: BatchStatus;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  started_at: string;
  completed_at?: string;
  error?: string;
}

// ─── User ────────────────────────────────────────────────────
export type UserRole = 'admin' | 'analyst' | 'viewer';

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  avatar_url?: string;
  last_login?: string;
}

// ─── Model ───────────────────────────────────────────────────
export type ModelStatus = 'active' | 'inactive' | 'training' | 'failed';

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  type: string;
  status: ModelStatus;
  accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  last_trained?: string;
  deployed_at?: string;
  description: string;
}

// ─── Timeline ────────────────────────────────────────────────
export interface TimelineEvent {
  id: string;
  date: string;
  type: EntityType;
  description: string;
  entities: Entity[];
  source_text: string;
}

// ─── Dashboard Stats ─────────────────────────────────────────
export interface DashboardStats {
  total_documents: number;
  total_entities: number;
  avg_risk_score: number;
  documents_today: number;
  processing_volume: { date: string; count: number }[];
  entity_distribution: { type: EntityType; count: number }[];
  recent_activity: {
    id: string;
    action: string;
    document_title: string;
    timestamp: string;
  }[];
}
