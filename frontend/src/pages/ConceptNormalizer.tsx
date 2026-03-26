/**
 * ConceptNormalizer page — Map clinical terms to standardized medical
 * concepts (CUI codes) with exact, alias, and fuzzy matching.
 *
 * Uses the concept normalization module to resolve free-text medical
 * terms to their canonical forms, CUI identifiers, and coding system
 * mappings (ICD-10, SNOMED, RxNorm, LOINC).  Supports batch input
 * with match-type indicators and confidence scores.
 */

import { useState, useCallback } from 'react';

// ─── Types ───────────────────────────────────────────────────

interface NormResult {
  input_text: string;
  matched: boolean;
  match_type: 'exact' | 'alias' | 'fuzzy' | 'none';
  preferred_term: string | null;
  cui: string | null;
  codes: Record<string, string>;
  confidence: number;
}

// ─── Sample Terms ────────────────────────────────────────────

const SAMPLE_BATCHES = [
  {
    label: 'Common Conditions',
    terms: [
      'heart attack',
      'high blood pressure',
      'diabetes',
      'broken arm',
      'pneumonia',
      'kidney failure',
      'stroke',
      'asthma',
      'depression',
      'osteoarthritis',
    ],
  },
  {
    label: 'Drug Names',
    terms: [
      'Tylenol',
      'ibuprofen',
      'metformin',
      'lisinopril',
      'Lipitor',
      'omeprazole',
      'Plavix',
      'aspirin',
      'amoxicillin',
      'warfarin',
    ],
  },
  {
    label: 'Lab Tests & Procedures',
    terms: [
      'CBC',
      'hemoglobin A1c',
      'chest X-ray',
      'MRI brain',
      'echocardiogram',
      'colonoscopy',
      'basic metabolic panel',
      'blood glucose',
      'troponin',
      'urinalysis',
    ],
  },
];

// ─── Match Type Styling ──────────────────────────────────────

const MATCH_STYLES: Record<string, { color: string; label: string; icon: string }> = {
  exact: {
    color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    label: 'Exact Match',
    icon: '✅',
  },
  alias: {
    color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    label: 'Alias Match',
    icon: '🔗',
  },
  fuzzy: {
    color: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    label: 'Fuzzy Match',
    icon: '🔍',
  },
  none: {
    color: 'bg-gray-100 text-gray-600 dark:bg-gray-800/30 dark:text-gray-400',
    label: 'No Match',
    icon: '❌',
  },
};

// ─── Code System Colors ──────────────────────────────────────

const CODE_SYSTEM_COLORS: Record<string, string> = {
  'ICD-10': 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  SNOMED: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
  RxNorm: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
  LOINC: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300',
  CPT: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
  CDT: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300',
};

// ─── Result Card Component ───────────────────────────────────

function ResultCard({ result }: { result: NormResult }) {
  const matchStyle = MATCH_STYLES[result.match_type] ?? MATCH_STYLES.none;
  const codeEntries = Object.entries(result.codes ?? {});

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        {/* Input term */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Input Term</p>
          <p className="text-base font-semibold text-gray-900 dark:text-white truncate">
            {result.input_text}
          </p>
        </div>
        {/* Match badge */}
        <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium shrink-0 ${matchStyle.color}`}>
          {matchStyle.icon} {matchStyle.label}
        </span>
      </div>

      {result.matched && (
        <div className="mt-3 space-y-2">
          {/* Preferred term */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400 w-20 shrink-0">
              Canonical:
            </span>
            <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
              {result.preferred_term}
            </span>
          </div>

          {/* CUI */}
          {result.cui && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 w-20 shrink-0">
                CUI:
              </span>
              <code className="text-xs bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded font-mono text-gray-800 dark:text-gray-200">
                {result.cui}
              </code>
            </div>
          )}

          {/* Confidence */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400 w-20 shrink-0">
              Confidence:
            </span>
            <div className="flex items-center gap-2 flex-1">
              <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden max-w-32">
                <div
                  className={`h-full rounded-full ${
                    result.confidence >= 0.9
                      ? 'bg-green-500'
                      : result.confidence >= 0.7
                        ? 'bg-blue-500'
                        : result.confidence >= 0.5
                          ? 'bg-amber-500'
                          : 'bg-red-500'
                  }`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                {(result.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Code mappings */}
          {codeEntries.length > 0 && (
            <div className="flex items-start gap-2">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 w-20 shrink-0 pt-0.5">
                Codes:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {codeEntries.map(([system, code]) => (
                  <span
                    key={system}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${CODE_SYSTEM_COLORS[system] ?? 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'}`}
                  >
                    <span className="font-bold">{system}:</span> {code}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!result.matched && (
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400 italic">
          No matching concept found in the normalization dictionary.
        </p>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function ConceptNormalizer() {
  const [singleTerm, setSingleTerm] = useState('');
  const [batchTerms, setBatchTerms] = useState('');
  const [results, setResults] = useState<NormResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const normalizeSingle = useCallback(async () => {
    if (!singleTerm.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/normalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: singleTerm.trim() }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults([data]);
      setProcessingTime(data.processing_time_ms ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Normalization failed');
    } finally {
      setLoading(false);
    }
  }, [singleTerm]);

  const normalizeBatch = useCallback(async () => {
    const terms = batchTerms
      .split('\n')
      .map((t) => t.trim())
      .filter(Boolean);
    if (terms.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/normalize/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: terms }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults(data.results ?? []);
      setProcessingTime(data.processing_time_ms ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Normalization failed');
    } finally {
      setLoading(false);
    }
  }, [batchTerms]);

  const loadSampleBatch = (idx: number) => {
    setBatchTerms(SAMPLE_BATCHES[idx].terms.join('\n'));
    setMode('batch');
    setResults([]);
    setError(null);
  };

  const handleAnalyze = mode === 'single' ? normalizeSingle : normalizeBatch;
  const matchedCount = results.filter((r) => r.matched).length;
  const exactCount = results.filter((r) => r.match_type === 'exact').length;
  const aliasCount = results.filter((r) => r.match_type === 'alias').length;
  const fuzzyCount = results.filter((r) => r.match_type === 'fuzzy').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          🔬 Concept Normalizer
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Map clinical terms to standardized concepts (CUI, ICD-10, SNOMED, RxNorm, LOINC)
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
        {/* Mode Toggle */}
        <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1 w-fit">
          <button
            onClick={() => { setMode('single'); setResults([]); setError(null); }}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              mode === 'single'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            Single Term
          </button>
          <button
            onClick={() => { setMode('batch'); setResults([]); setError(null); }}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              mode === 'batch'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            Batch Input
          </button>
        </div>

        {/* Sample Batch Buttons */}
        {mode === 'batch' && (
          <div className="flex flex-wrap gap-2">
            {SAMPLE_BATCHES.map((batch, idx) => (
              <button
                key={batch.label}
                onClick={() => loadSampleBatch(idx)}
                className="px-3 py-1.5 text-xs font-medium rounded-md bg-teal-50 text-teal-700 hover:bg-teal-100 dark:bg-teal-900/30 dark:text-teal-300 dark:hover:bg-teal-900/50 transition-colors"
              >
                📋 {batch.label}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        {mode === 'single' ? (
          <div className="flex gap-3">
            <input
              type="text"
              value={singleTerm}
              onChange={(e) => setSingleTerm(e.target.value)}
              placeholder='Enter a clinical term (e.g., "heart attack", "Tylenol", "CBC")'
              className="flex-1 rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-sm focus:border-teal-500 focus:ring-teal-500"
              disabled={loading}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
          </div>
        ) : (
          <div>
            <textarea
              value={batchTerms}
              onChange={(e) => setBatchTerms(e.target.value)}
              rows={8}
              placeholder="Enter one term per line..."
              className="w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-sm focus:border-teal-500 focus:ring-teal-500 font-mono text-sm"
              disabled={loading}
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {batchTerms.split('\n').filter((t) => t.trim()).length} terms
            </p>
          </div>
        )}

        {/* Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={loading || (mode === 'single' ? !singleTerm.trim() : !batchTerms.trim())}
          className="px-4 py-2 bg-teal-600 text-white rounded-md hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Normalizing...
            </span>
          ) : (
            '🔬 Normalize'
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-700 dark:text-red-300">❌ {error}</p>
          </div>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          {/* Stats Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-teal-600 dark:text-teal-400">{results.length}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Total Terms</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">{exactCount}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Exact</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{aliasCount}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Alias</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">{fuzzyCount}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Fuzzy</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                {matchedCount > 0 ? ((matchedCount / results.length) * 100).toFixed(0) : 0}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Match Rate</p>
            </div>
          </div>

          {/* Result Cards */}
          <div className="grid gap-3 sm:grid-cols-2">
            {results.map((result, idx) => (
              <ResultCard key={`${result.input_text}-${idx}`} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
