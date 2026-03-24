import type {
  AnalysisResult,
  Entity,
  ICDPrediction,
  ClinicalSummary,
  RiskAssessment,
  Document,
  BatchJob,
  ModelInfo,
  DashboardStats,
  SummaryDetail,
  User,
} from './index';

// ─── Request Types ───────────────────────────────────────────

export interface AnalyzeRequest {
  text: string;
  options?: {
    extract_entities?: boolean;
    predict_icd?: boolean;
    generate_summary?: boolean;
    assess_risk?: boolean;
  };
}

export interface NERRequest {
  text: string;
  entity_types?: string[];
  min_confidence?: number;
}

export interface ICDRequest {
  text: string;
  max_predictions?: number;
  min_confidence?: number;
}

export interface SummarizeRequest {
  text: string;
  detail_level?: SummaryDetail;
}

export interface RiskRequest {
  text: string;
  patient_context?: Record<string, unknown>;
}

export interface BatchRequest {
  name: string;
  documents: { title: string; text: string }[];
}

export interface LoginRequest {
  email: string;
  password: string;
}

// ─── Response Types ──────────────────────────────────────────

export interface ApiResponse<T> {
  data: T;
  status: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export type AnalyzeResponse = ApiResponse<AnalysisResult>;
export type NERResponse = ApiResponse<{ entities: Entity[] }>;
export type ICDResponse = ApiResponse<{ predictions: ICDPrediction[] }>;
export type SummarizeResponse = ApiResponse<ClinicalSummary>;
export type RiskResponse = ApiResponse<RiskAssessment>;
export type BatchResponse = ApiResponse<BatchJob>;
export type DocumentListResponse = PaginatedResponse<Document>;
export type DocumentResponse = ApiResponse<Document>;
export type ModelListResponse = ApiResponse<ModelInfo[]>;
export type ModelResponse = ApiResponse<ModelInfo>;
export type DashboardResponse = ApiResponse<DashboardStats>;
export type LoginResponse = ApiResponse<{ token: string; user: User }>;
export type UserResponse = ApiResponse<User>;
