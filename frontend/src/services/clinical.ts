/**
 * API service layer for ClinIQ's specialized clinical NLP endpoints.
 *
 * This module provides typed functions for all 31 backend endpoint groups,
 * complementing the core analysis endpoints in ``api.ts``.  Every function
 * returns a strongly-typed promise matching the backend's response schema.
 *
 * Design decisions:
 * - All functions accept plain objects (not FormData) since the backend
 *   uses JSON everywhere.
 * - ``client`` is imported from the core API module so auth interceptors
 *   and base URL are shared.
 * - Batch endpoints accept an array and handle the wrapping internally.
 */

import client from './api';
import type {
  AddTurnRequest,
  AddTurnResponse,
  ContextRequest,
  ContextResponse,
  ConversationStats,
  SessionsListResponse,
  MedicationExtractionResponse,
  AllergyExtractionResponse,
  VitalExtractionResponse,
  SectionParseResponse,
  AbbreviationExpansionResponse,
  DeidentifyResponse,
  ReplacementStrategy,
  ClassificationResponse,
  QualityReport,
  SDoHExtractionResponse,
  ComorbidityResult,
  NormalizationResult,
  AssertionResult,
  RelationExtractionResponse,
  TemporalExtractionResponse,
  EnhancedAnalysisConfig,
  EnhancedAnalysisResponse,
  SearchResponse,
  DriftStatusResponse,
  ModuleInfo,
  ClassificationEvalRequest,
  ClassificationEvalResponse,
  KappaRequest,
  KappaResponse,
  NEREvalRequest,
  NEREvalResponse,
  ROUGERequest,
  ROUGEResponse,
  ICDEvalRequest,
  ICDEvalResponse,
  AUPRCRequest,
  AUPRCResponse,
  EvaluationMetricsCatalogue,
} from '../types/clinical';

// ─── Medication Extraction ───────────────────────────────────

export const extractMedications = async (
  text: string,
  options?: { min_confidence?: number; include_generics?: boolean }
): Promise<MedicationExtractionResponse> => {
  const response = await client.post('/medications', { text, ...options });
  return response.data;
};

export const lookupMedication = async (
  drugName: string
): Promise<{ drug_name: string; generic_name: string; brands: string[] }> => {
  const response = await client.get(`/medications/lookup/${encodeURIComponent(drugName)}`);
  return response.data;
};

// ─── Allergy Extraction ──────────────────────────────────────

export const extractAllergies = async (
  text: string,
  options?: { min_confidence?: number }
): Promise<AllergyExtractionResponse> => {
  const response = await client.post('/allergies', { text, ...options });
  return response.data;
};

export const getAllergyDictionaryStats = async (): Promise<{
  total_allergens: number;
  categories: Record<string, number>;
}> => {
  const response = await client.get('/allergies/dictionary/stats');
  return response.data;
};

// ─── Vital Signs ─────────────────────────────────────────────

export const extractVitals = async (
  text: string,
  options?: { min_confidence?: number }
): Promise<VitalExtractionResponse> => {
  const response = await client.post('/vitals', { text, ...options });
  return response.data;
};

export const getVitalTypes = async (): Promise<{
  types: { name: string; unit: string }[];
}> => {
  const response = await client.get('/vitals/types');
  return response.data;
};

export const getVitalRanges = async (): Promise<Record<string, {
  normal_min: number;
  normal_max: number;
  unit: string;
}>> => {
  const response = await client.get('/vitals/ranges');
  return response.data;
};

// ─── Section Parsing ─────────────────────────────────────────

export const parseSections = async (
  text: string,
  options?: { min_confidence?: number }
): Promise<SectionParseResponse> => {
  const response = await client.post('/sections', { text, ...options });
  return response.data;
};

export const getSectionCategories = async (): Promise<{
  categories: { name: string; description: string }[];
}> => {
  const response = await client.get('/sections/categories');
  return response.data;
};

// ─── Abbreviation Expansion ──────────────────────────────────

export const expandAbbreviations = async (
  text: string,
  options?: { min_confidence?: number; expand_in_place?: boolean; domain?: string }
): Promise<AbbreviationExpansionResponse> => {
  const response = await client.post('/abbreviations', { text, ...options });
  return response.data;
};

export const lookupAbbreviation = async (
  abbr: string
): Promise<{
  abbreviation: string;
  expansions: { expansion: string; domain: string; is_ambiguous: boolean }[];
}> => {
  const response = await client.get(`/abbreviations/lookup/${encodeURIComponent(abbr)}`);
  return response.data;
};

// ─── De-identification ───────────────────────────────────────

export const deidentifyText = async (
  text: string,
  options?: {
    strategy?: ReplacementStrategy;
    phi_types?: string[];
    min_confidence?: number;
  }
): Promise<DeidentifyResponse> => {
  const response = await client.post('/deidentify', { text, ...options });
  return response.data;
};

// ─── Document Classification ─────────────────────────────────

export const classifyDocument = async (
  text: string,
  options?: { min_confidence?: number; top_k?: number }
): Promise<ClassificationResponse> => {
  const response = await client.post('/classify', { text, ...options });
  return response.data;
};

export const getDocumentTypes = async (): Promise<{
  types: { name: string; description: string }[];
}> => {
  const response = await client.get('/classify/types');
  return response.data;
};

// ─── Quality Analysis ────────────────────────────────────────

export const analyzeQuality = async (
  text: string,
  options?: { expected_sections?: string[] }
): Promise<QualityReport> => {
  const response = await client.post('/quality', { text, ...options });
  return response.data;
};

export const getQualityDimensions = async (): Promise<{
  dimensions: { name: string; description: string; weight: number }[];
}> => {
  const response = await client.get('/quality/dimensions');
  return response.data;
};

// ─── SDoH Extraction ─────────────────────────────────────────

export const extractSDoH = async (
  text: string,
  options?: { min_confidence?: number }
): Promise<SDoHExtractionResponse> => {
  const response = await client.post('/sdoh', { text, ...options });
  return response.data;
};

export const getSDoHDomains = async (): Promise<{
  domains: { name: string; description: string; z_codes: string[] }[];
}> => {
  const response = await client.get('/sdoh/domains');
  return response.data;
};

// ─── Comorbidity (Charlson) ──────────────────────────────────

export const calculateComorbidity = async (
  options: {
    icd_codes?: string[];
    text?: string;
    age?: number;
    age_adjust?: boolean;
  }
): Promise<ComorbidityResult> => {
  const response = await client.post('/comorbidity', options);
  return response.data;
};

export const getComorbidityCategories = async (): Promise<{
  categories: { name: string; weight: number; description: string }[];
}> => {
  const response = await client.get('/comorbidity/categories');
  return response.data;
};

// ─── Concept Normalization ───────────────────────────────────

export const normalizeConcept = async (
  text: string,
  options?: { entity_type?: string; min_similarity?: number; enable_fuzzy?: boolean }
): Promise<NormalizationResult> => {
  const response = await client.post('/normalize', { text, ...options });
  return response.data;
};

// ─── Assertion Detection ─────────────────────────────────────

export const detectAssertion = async (
  text: string,
  entityStart: number,
  entityEnd: number
): Promise<AssertionResult> => {
  const response = await client.post('/assertions', {
    text,
    entity_start: entityStart,
    entity_end: entityEnd,
  });
  return response.data;
};

export const getAssertionStatuses = async (): Promise<{
  statuses: { name: string; description: string }[];
}> => {
  const response = await client.get('/assertions/statuses');
  return response.data;
};

// ─── Relation Extraction ─────────────────────────────────────

export const extractRelations = async (
  text: string,
  options?: { relation_types?: string[]; min_confidence?: number }
): Promise<RelationExtractionResponse> => {
  const response = await client.post('/relations', { text, ...options });
  return response.data;
};

// ─── Temporal Extraction ─────────────────────────────────────

export const extractTemporal = async (
  text: string,
  options?: { reference_date?: string }
): Promise<TemporalExtractionResponse> => {
  const response = await client.post('/temporal', { text, ...options });
  return response.data;
};

// ─── Enhanced Analysis (all-in-one) ──────────────────────────

export const analyzeEnhanced = async (
  text: string,
  config?: EnhancedAnalysisConfig
): Promise<EnhancedAnalysisResponse> => {
  const response = await client.post('/analyze/enhanced', { text, config });
  return response.data;
};

export const getEnhancedModules = async (): Promise<ModuleInfo[]> => {
  const response = await client.get('/analyze/enhanced/modules');
  return response.data;
};

// ─── Search ──────────────────────────────────────────────────

export const searchDocuments = async (
  query: string,
  options?: {
    top_k?: number;
    min_score?: number;
    alpha?: number;
    expand_query?: boolean;
    rerank?: boolean;
  }
): Promise<SearchResponse> => {
  const response = await client.post('/search', { query, ...options });
  return response.data;
};

// ─── Drift Monitoring ────────────────────────────────────────

export const getDriftStatus = async (): Promise<DriftStatusResponse> => {
  const response = await client.get('/drift/status');
  return response.data;
};

// ─── Streaming Analysis ──────────────────────────────────────

/**
 * Opens a Server-Sent Events connection for real-time pipeline analysis.
 *
 * Each pipeline stage emits its own event (ner, icd, summary, risk)
 * as it completes, allowing the UI to show incremental progress.
 *
 * @param text - Clinical document text
 * @param onEvent - Callback fired for each SSE event
// ─── Conversation Memory ─────────────────────────────────────

/** Record a completed analysis turn in a session's conversation history. */
export const addConversationTurn = async (
  data: AddTurnRequest,
): Promise<AddTurnResponse> => {
  const resp = await client.post('/conversation/turns', data);
  return resp.data;
};

/** Retrieve aggregated context from a session's recent history. */
export const getConversationContext = async (
  data: ContextRequest,
): Promise<ContextResponse> => {
  const resp = await client.post('/conversation/context', data);
  return resp.data;
};

/** Clear a session's conversation history. */
export const clearConversationSession = async (
  sessionId: string,
): Promise<{ session_id: string; status: string }> => {
  const resp = await client.delete(`/conversation/${sessionId}`);
  return resp.data;
};

/** Get conversation memory usage statistics. */
export const getConversationStats = async (): Promise<ConversationStats> => {
  const resp = await client.get('/conversation/stats');
  return resp.data;
};

/** List all active conversation sessions. */
export const listConversationSessions =
  async (): Promise<SessionsListResponse> => {
    const resp = await client.get('/conversation/sessions');
    return resp.data;
  };

// ─── Streaming Analysis ──────────────────────────────────────

 * @param onError - Callback for connection errors
 * @returns AbortController to cancel the stream
 */
export const analyzeStream = (
  text: string,
  onEvent: (event: { stage: string; data: unknown }) => void,
  onError?: (error: Event) => void
): AbortController => {
  const controller = new AbortController();

  // Use fetch with ReadableStream for SSE since EventSource doesn't
  // support POST bodies.  This is the standard pattern for POST-based SSE.
  const baseURL = client.defaults.baseURL || '/api/v1';
  const token = localStorage.getItem('cliniq_token');

  fetch(`${baseURL}/analyze/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ text }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok || !response.body) {
        throw new Error(`Stream failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6));
              onEvent(parsed);
            } catch {
              // Skip malformed events
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError' && onError) {
        onError(err);
      }
    });

  return controller;
};

// ─── Evaluation ──────────────────────────────────────────────

/** Evaluate binary classification predictions (MCC + confusion matrix + calibration). */
export const evaluateClassification = async (
  request: ClassificationEvalRequest
): Promise<ClassificationEvalResponse> => {
  const resp = await client.post('/evaluate/classification', request);
  return resp.data;
};

/** Compute inter-annotator agreement (Cohen's Kappa). */
export const evaluateAgreement = async (
  request: KappaRequest
): Promise<KappaResponse> => {
  const resp = await client.post('/evaluate/agreement', request);
  return resp.data;
};

/** Evaluate NER with exact and partial span matching. */
export const evaluateNER = async (
  request: NEREvalRequest
): Promise<NEREvalResponse> => {
  const resp = await client.post('/evaluate/ner', request);
  return resp.data;
};

/** Evaluate summarisation with full ROUGE-1/2/L. */
export const evaluateROUGE = async (
  request: ROUGERequest
): Promise<ROUGEResponse> => {
  const resp = await client.post('/evaluate/rouge', request);
  return resp.data;
};

/** Evaluate ICD-10 predictions at chapter/block/full-code levels. */
export const evaluateICD = async (
  request: ICDEvalRequest
): Promise<ICDEvalResponse> => {
  const resp = await client.post('/evaluate/icd', request);
  return resp.data;
};

/** Compute Area Under Precision-Recall Curve for binary predictions. */
export const evaluateAUPRC = async (
  request: AUPRCRequest
): Promise<AUPRCResponse> => {
  const resp = await client.post('/evaluate/auprc', request);
  return resp.data;
};

/** List available evaluation metrics. */
export const listEvaluationMetrics =
  async (): Promise<EvaluationMetricsCatalogue> => {
    const resp = await client.get('/evaluate/metrics');
    return resp.data;
  };
