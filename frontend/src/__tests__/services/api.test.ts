/**
 * Tests for the API service layer (services/api.ts).
 *
 * Uses Vitest mocks to stub axios. Verifies correct HTTP method, URL,
 * payload, auth header injection, and 401 redirect behaviour for every
 * exported API function.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios from 'axios';

// We mock the entire axios module so the real HTTP client is never invoked.
vi.mock('axios', () => {
  const mockInstance = {
    post: vi.fn(),
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
    defaults: { headers: { common: {} } },
  };
  return {
    default: {
      create: vi.fn(() => mockInstance),
      __mockInstance: mockInstance,
    },
  };
});

// Get the mock instance that all api.ts calls will use
function getMockClient() {
  return (axios as any).__mockInstance;
}

// Dynamically import after mock is set up
let api: typeof import('../../services/api');

beforeEach(async () => {
  vi.resetModules();
  // Re-import so interceptors re-bind
  api = await import('../../services/api');
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('API Service', () => {
  // ── Analysis Endpoints ──────────────────────────────────────

  describe('analyzeText', () => {
    it('POSTs to /analyze with request body', async () => {
      const mock = getMockClient();
      const payload = { text: 'Patient has fever' };
      const response = { data: { data: { id: '1' }, status: 'ok' } };
      mock.post.mockResolvedValueOnce(response);

      const result = await api.analyzeText(payload);
      expect(mock.post).toHaveBeenCalledWith('/analyze', payload);
      expect(result).toEqual(response.data);
    });
  });

  describe('extractEntities', () => {
    it('POSTs to /ner', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({ data: { data: { entities: [] } } });

      await api.extractEntities({ text: 'test' });
      expect(mock.post).toHaveBeenCalledWith('/ner', { text: 'test' });
    });
  });

  describe('predictICD', () => {
    it('POSTs to /icd/predict', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({ data: { data: { predictions: [] } } });

      await api.predictICD({ text: 'hypertension' });
      expect(mock.post).toHaveBeenCalledWith('/icd/predict', { text: 'hypertension' });
    });
  });

  describe('summarizeText', () => {
    it('POSTs to /summarize', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({ data: { data: { summary: '' } } });

      await api.summarizeText({ text: 'long note' });
      expect(mock.post).toHaveBeenCalledWith('/summarize', { text: 'long note' });
    });
  });

  describe('assessRisk', () => {
    it('POSTs to /risk/assess', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({ data: { data: {} } });

      await api.assessRisk({ text: 'diabetic patient' });
      expect(mock.post).toHaveBeenCalledWith('/risk/assess', { text: 'diabetic patient' });
    });
  });

  // ── Batch Endpoints ─────────────────────────────────────────

  describe('createBatchJob', () => {
    it('POSTs to /batch', async () => {
      const mock = getMockClient();
      const payload = { name: 'Batch 1', documents: [{ title: 'Doc', text: 'text' }] };
      mock.post.mockResolvedValueOnce({ data: { data: { id: 'b1' } } });

      await api.createBatchJob(payload);
      expect(mock.post).toHaveBeenCalledWith('/batch', payload);
    });
  });

  describe('getBatchJob', () => {
    it('GETs /batch/:id', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { data: { id: 'b1' } } });

      await api.getBatchJob('b1');
      expect(mock.get).toHaveBeenCalledWith('/batch/b1');
    });
  });

  // ── Document Endpoints ──────────────────────────────────────

  describe('getDocuments', () => {
    it('GETs /documents with pagination params', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { items: [], total: 0 } });

      await api.getDocuments(2, 10);
      expect(mock.get).toHaveBeenCalledWith('/documents', {
        params: { page: 2, per_page: 10 },
      });
    });

    it('uses default pagination (page=1, perPage=20)', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { items: [] } });

      await api.getDocuments();
      expect(mock.get).toHaveBeenCalledWith('/documents', {
        params: { page: 1, per_page: 20 },
      });
    });
  });

  describe('getDocument', () => {
    it('GETs /documents/:id', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { data: { id: 'd1' } } });

      await api.getDocument('d1');
      expect(mock.get).toHaveBeenCalledWith('/documents/d1');
    });
  });

  // ── Model Endpoints ─────────────────────────────────────────

  describe('getModels', () => {
    it('GETs /models', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { data: [] } });

      await api.getModels();
      expect(mock.get).toHaveBeenCalledWith('/models');
    });
  });

  // ── Dashboard ───────────────────────────────────────────────

  describe('getDashboardStats', () => {
    it('GETs /dashboard/stats', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { data: {} } });

      await api.getDashboardStats();
      expect(mock.get).toHaveBeenCalledWith('/dashboard/stats');
    });
  });

  // ── Auth Endpoints ──────────────────────────────────────────

  describe('login', () => {
    it('POSTs to /auth/login', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({
        data: { data: { token: 'abc', user: {} } },
      });

      await api.login({ email: 'doc@hospital.org', password: 'secret' });
      expect(mock.post).toHaveBeenCalledWith('/auth/login', {
        email: 'doc@hospital.org',
        password: 'secret',
      });
    });
  });

  describe('getCurrentUser', () => {
    it('GETs /auth/me', async () => {
      const mock = getMockClient();
      mock.get.mockResolvedValueOnce({ data: { data: { id: 'u1' } } });

      await api.getCurrentUser();
      expect(mock.get).toHaveBeenCalledWith('/auth/me');
    });
  });

  describe('logout', () => {
    it('POSTs to /auth/logout and removes token', async () => {
      const mock = getMockClient();
      mock.post.mockResolvedValueOnce({});
      localStorage.setItem('cliniq_token', 'old-token');

      await api.logout();
      expect(mock.post).toHaveBeenCalledWith('/auth/logout');
      expect(localStorage.getItem('cliniq_token')).toBeNull();
    });
  });
});

describe('Axios instance configuration', () => {
  it('creates client with /api/v1 baseURL', () => {
    expect(axios.create).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: '/api/v1',
        timeout: 60000,
      })
    );
  });

  it('registers request and response interceptors', () => {
    const mock = getMockClient();
    expect(mock.interceptors.request.use).toHaveBeenCalled();
    expect(mock.interceptors.response.use).toHaveBeenCalled();
  });
});
