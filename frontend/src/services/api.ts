import axios from 'axios';
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  NERRequest,
  NERResponse,
  ICDRequest,
  ICDResponse,
  SummarizeRequest,
  SummarizeResponse,
  RiskRequest,
  RiskResponse,
  BatchRequest,
  BatchResponse,
  DocumentListResponse,
  DocumentResponse,
  ModelListResponse,
  DashboardResponse,
  LoginRequest,
  LoginResponse,
  UserResponse,
} from '../types/api';

// ─── Axios Instance ──────────────────────────────────────────

const client = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000,
});

// ─── Request Interceptor ────────────────────────────────────

client.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('cliniq_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ─── Response Interceptor ───────────────────────────────────

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('cliniq_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ─── Analysis Endpoints ─────────────────────────────────────

export const analyzeText = async (data: AnalyzeRequest): Promise<AnalyzeResponse> => {
  const response = await client.post<AnalyzeResponse>('/analyze', data);
  return response.data;
};

export const extractEntities = async (data: NERRequest): Promise<NERResponse> => {
  const response = await client.post<NERResponse>('/ner', data);
  return response.data;
};

export const predictICD = async (data: ICDRequest): Promise<ICDResponse> => {
  const response = await client.post<ICDResponse>('/icd/predict', data);
  return response.data;
};

export const summarizeText = async (data: SummarizeRequest): Promise<SummarizeResponse> => {
  const response = await client.post<SummarizeResponse>('/summarize', data);
  return response.data;
};

export const assessRisk = async (data: RiskRequest): Promise<RiskResponse> => {
  const response = await client.post<RiskResponse>('/risk/assess', data);
  return response.data;
};

// ─── Batch Endpoints ─────────────────────────────────────────

export const createBatchJob = async (data: BatchRequest): Promise<BatchResponse> => {
  const response = await client.post<BatchResponse>('/batch', data);
  return response.data;
};

export const getBatchJob = async (id: string): Promise<BatchResponse> => {
  const response = await client.get<BatchResponse>(`/batch/${id}`);
  return response.data;
};

// ─── Document Endpoints ──────────────────────────────────────

export const getDocuments = async (
  page = 1,
  perPage = 20
): Promise<DocumentListResponse> => {
  const response = await client.get<DocumentListResponse>('/documents', {
    params: { page, per_page: perPage },
  });
  return response.data;
};

export const getDocument = async (id: string): Promise<DocumentResponse> => {
  const response = await client.get<DocumentResponse>(`/documents/${id}`);
  return response.data;
};

// ─── Model Endpoints ─────────────────────────────────────────

export const getModels = async (): Promise<ModelListResponse> => {
  const response = await client.get<ModelListResponse>('/models');
  return response.data;
};

// ─── Dashboard Endpoints ─────────────────────────────────────

export const getDashboardStats = async (): Promise<DashboardResponse> => {
  const response = await client.get<DashboardResponse>('/dashboard/stats');
  return response.data;
};

// ─── Auth Endpoints ──────────────────────────────────────────

export const login = async (data: LoginRequest): Promise<LoginResponse> => {
  const response = await client.post<LoginResponse>('/auth/login', data);
  return response.data;
};

export const getCurrentUser = async (): Promise<UserResponse> => {
  const response = await client.get<UserResponse>('/auth/me');
  return response.data;
};

export const logout = async (): Promise<void> => {
  await client.post('/auth/logout');
  localStorage.removeItem('cliniq_token');
};

export default client;
