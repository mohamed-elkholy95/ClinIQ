/**
 * Deidentification page — Remove PHI from clinical text per HIPAA Safe Harbor.
 *
 * Provides a split-pane view showing the original text alongside the
 * de-identified output with highlighted PHI detections.  Supports all
 * three replacement strategies (redact, mask, surrogate) and allows
 * filtering by PHI type.
 */

import { useState, useCallback, useMemo } from 'react';
import type {
  DeidentifyResponse,
  PHIDetection,
  PHIType,
  ReplacementStrategy,
} from '../types/clinical';

// ─── PHI Type Metadata ───────────────────────────────────────

const PHI_TYPE_INFO: Record<
  PHIType,
  { label: string; color: string; icon: string }
> = {
  NAME: { label: 'Name', color: 'bg-rose-100 text-rose-800', icon: '👤' },
  DATE: { label: 'Date', color: 'bg-amber-100 text-amber-800', icon: '📅' },
  PHONE: { label: 'Phone', color: 'bg-blue-100 text-blue-800', icon: '📞' },
  EMAIL: { label: 'Email', color: 'bg-violet-100 text-violet-800', icon: '✉️' },
  SSN: { label: 'SSN', color: 'bg-red-200 text-red-900', icon: '🔒' },
  MRN: { label: 'MRN', color: 'bg-orange-100 text-orange-800', icon: '🏥' },
  URL: { label: 'URL', color: 'bg-cyan-100 text-cyan-800', icon: '🔗' },
  IP_ADDRESS: { label: 'IP Address', color: 'bg-slate-100 text-slate-800', icon: '🌐' },
  ZIP_CODE: { label: 'ZIP Code', color: 'bg-emerald-100 text-emerald-800', icon: '📍' },
  AGE_OVER_90: { label: 'Age ≥ 90', color: 'bg-pink-100 text-pink-800', icon: '🎂' },
  ACCOUNT_NUMBER: { label: 'Account #', color: 'bg-indigo-100 text-indigo-800', icon: '💳' },
  LICENSE_NUMBER: { label: 'License #', color: 'bg-teal-100 text-teal-800', icon: '🪪' },
};

// ─── Sample Clinical Note ────────────────────────────────────

const SAMPLE_NOTE = `PATIENT: John Smith
DOB: 03/15/1960
MRN: MRN-2024-78432
SSN: 123-45-6789

ATTENDING PHYSICIAN: Dr. Sarah Johnson
DATE OF SERVICE: 01/15/2025

CHIEF COMPLAINT: 65-year-old male presenting with chest pain.

Mr. Smith was seen at the emergency department at Memorial Hospital.
He reports crushing substernal chest pain radiating to the left arm,
onset 2 hours prior to arrival. Patient called 911 from his home at
123 Oak Street, Springfield, IL 62704.

Contact: (555) 867-5309, email: john.smith@email.com
Insurance: Account #: 9876543210
NPI: 1234567890

ASSESSMENT: Acute STEMI. Patient transferred to cath lab for PCI.
Follow-up scheduled with Dr. Michael Chen at Cardiology Associates.

Report accessed from IP 192.168.1.42 via patient portal at
https://portal.memorialhospital.com/records/78432`;

// ─── Strategy Description ────────────────────────────────────

const STRATEGY_INFO: Record<ReplacementStrategy, { label: string; description: string; example: string }> = {
  redact: {
    label: 'Redact',
    description: 'Replace PHI with bracketed type labels',
    example: 'John Smith → [NAME]',
  },
  mask: {
    label: 'Mask',
    description: 'Replace characters with asterisks (length-preserving)',
    example: 'John Smith → **********',
  },
  surrogate: {
    label: 'Surrogate',
    description: 'Replace with realistic synthetic values',
    example: 'John Smith → Robert Davis',
  },
};

// ─── Detection Badge ─────────────────────────────────────────

function DetectionBadge({ detection }: { detection: PHIDetection }) {
  const info = PHI_TYPE_INFO[detection.type] || {
    label: detection.type,
    color: 'bg-gray-100 text-gray-800',
    icon: '❓',
  };

  return (
    <div className="flex items-center justify-between rounded-lg border bg-white p-3 text-sm hover:shadow-sm transition-shadow">
      <div className="flex items-center gap-3">
        <span className="text-lg">{info.icon}</span>
        <div>
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${info.color}`}>
            {info.label}
          </span>
          <p className="mt-1 font-mono text-xs text-red-600 line-through">{detection.text}</p>
          <p className="font-mono text-xs text-green-700">{detection.replacement}</p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-xs text-gray-500">{Math.round(detection.confidence * 100)}%</p>
        <p className="text-xs text-gray-400">
          {detection.start_char}–{detection.end_char}
        </p>
      </div>
    </div>
  );
}

// ─── Main Page Component ─────────────────────────────────────

export function Deidentification() {
  const [text, setText] = useState('');
  const [strategy, setStrategy] = useState<ReplacementStrategy>('redact');
  const [minConfidence, setMinConfidence] = useState(0.7);
  const [results, setResults] = useState<DeidentifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTypes, setSelectedTypes] = useState<Set<PHIType>>(new Set());

  const handleDeidentify = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/deidentify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          strategy,
          min_confidence: minConfidence,
          ...(selectedTypes.size > 0 ? { phi_types: Array.from(selectedTypes) } : {}),
        }),
      });
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data: DeidentifyResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'De-identification failed');
    } finally {
      setLoading(false);
    }
  }, [text, strategy, minConfidence, selectedTypes]);

  const toggleType = (type: PHIType) => {
    setSelectedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  // Count detections by type for summary
  const typeCounts = useMemo(() => {
    if (!results) return new Map<PHIType, number>();
    const counts = new Map<PHIType, number>();
    for (const d of results.detections) {
      counts.set(d.type, (counts.get(d.type) || 0) + 1);
    }
    return counts;
  }, [results]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          🔒 PHI De-identification
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Remove Protected Health Information per HIPAA Safe Harbor rules.
          Covers all 18 identifier categories including names, dates, SSNs, MRNs,
          phone numbers, emails, and more.
        </p>
      </div>

      {/* Strategy Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {(Object.entries(STRATEGY_INFO) as [ReplacementStrategy, typeof STRATEGY_INFO[ReplacementStrategy]][]).map(
          ([key, info]) => (
            <button
              key={key}
              onClick={() => {
                setStrategy(key);
                setResults(null);
              }}
              className={`rounded-lg border-2 p-4 text-left transition-all ${
                strategy === key
                  ? 'border-indigo-500 bg-indigo-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
            >
              <h3 className="text-sm font-semibold text-gray-900">{info.label}</h3>
              <p className="mt-1 text-xs text-gray-500">{info.description}</p>
              <p className="mt-2 text-xs font-mono text-gray-600">{info.example}</p>
            </button>
          )
        )}
      </div>

      {/* PHI Type Filter */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Filter PHI types (empty = detect all):
        </label>
        <div className="flex flex-wrap gap-2">
          {(Object.keys(PHI_TYPE_INFO) as PHIType[]).map((type) => {
            const info = PHI_TYPE_INFO[type];
            const isSelected = selectedTypes.has(type);
            return (
              <button
                key={type}
                onClick={() => toggleType(type)}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                  isSelected
                    ? `${info.color} ring-2 ring-offset-1 ring-indigo-300`
                    : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                }`}
              >
                {info.icon} {info.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Confidence Slider */}
      <div className="flex items-center gap-4">
        <label className="text-sm font-medium text-gray-700">
          Min confidence: {Math.round(minConfidence * 100)}%
        </label>
        <input
          type="range"
          min={0.5}
          max={1.0}
          step={0.05}
          value={minConfidence}
          onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
          className="w-48 accent-indigo-600"
        />
      </div>

      {/* Input */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">Input Text</label>
          <button
            onClick={() => {
              setText(SAMPLE_NOTE);
              setResults(null);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            Load sample note
          </button>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste clinical text containing PHI to de-identify..."
          rows={10}
          className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <div className="mt-2 flex justify-end">
          <button
            onClick={handleDeidentify}
            disabled={!text.trim() || loading}
            className="px-6 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'De-identifying…' : '🔒 De-identify'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">PHI Detected</p>
              <p className="mt-1 text-2xl font-bold text-red-600">{results.count}</p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">PHI Types</p>
              <p className="mt-1 text-2xl font-bold text-gray-900">{typeCounts.size}</p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Strategy</p>
              <p className="mt-1 text-2xl font-bold text-indigo-600 capitalize">{results.strategy}</p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Processing Time</p>
              <p className="mt-1 text-2xl font-bold text-gray-600">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
            </div>
          </div>

          {/* Type Distribution */}
          {typeCounts.size > 0 && (
            <div className="bg-white rounded-lg border p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Detection Distribution</h3>
              <div className="flex flex-wrap gap-2">
                {Array.from(typeCounts.entries())
                  .sort((a, b) => b[1] - a[1])
                  .map(([type, count]) => {
                    const info = PHI_TYPE_INFO[type];
                    return (
                      <span
                        key={type}
                        className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${info?.color || 'bg-gray-100 text-gray-700'}`}
                      >
                        {info?.icon} {info?.label || type}: {count}
                      </span>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Split Pane: Original vs De-identified */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                🔴 Original Text
              </h3>
              <div className="rounded-lg border bg-red-50 p-4 text-sm font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                {text}
              </div>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                🟢 De-identified Text
              </h3>
              <div className="rounded-lg border bg-green-50 p-4 text-sm font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                {results.deidentified_text}
              </div>
            </div>
          </div>

          {/* Detection List */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              PHI Detections ({results.count})
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {results.detections.map((detection, i) => (
                <DetectionBadge key={i} detection={detection} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
