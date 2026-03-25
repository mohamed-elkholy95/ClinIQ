/**
 * MedicationExtractor page — Extract structured medication data from clinical text.
 *
 * Demonstrates the rule-based medication extraction pipeline that identifies
 * drug names, dosages, routes, frequencies, durations, PRN status, and
 * medication status from free-text clinical notes.  Results are displayed
 * in a sortable, filterable table with inline confidence indicators.
 */

import { useState, useCallback } from 'react';
import type { MedicationResult, MedicationExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Discharge Medications',
    text: `DISCHARGE MEDICATIONS:
1. Metformin 500mg PO BID for diabetes
2. Lisinopril 10mg PO daily for hypertension
3. Atorvastatin 40mg PO at bedtime for hyperlipidemia
4. Aspirin 81mg PO daily
5. Metoprolol succinate 25mg PO daily
6. Omeprazole 20mg PO daily for GERD
7. Albuterol 2 puffs inhaled q4-6h PRN for shortness of breath
8. Acetaminophen 650mg PO q6h PRN for pain

DISCONTINUED:
- Hydrochlorothiazide 25mg — held due to hyponatremia
- Glipizide 5mg — changed to metformin`,
  },
  {
    label: 'Progress Note',
    text: `Patient is currently on Lipitor 80mg daily, recently started Eliquis 5mg BID for new-onset atrial fibrillation. Continue home medications including amlodipine 5mg daily, gabapentin 300mg TID for neuropathic pain, and sertraline 100mg daily. Insulin glargine 20 units SQ at bedtime. PRN tramadol 50mg q6h for breakthrough pain. Discussed discontinuing naproxen due to GI risk with anticoagulation.`,
  },
  {
    label: 'Dental Note',
    text: `Post-extraction instructions given. Prescribed amoxicillin 500mg PO TID for 7 days for prophylaxis. Ibuprofen 600mg PO q6h PRN for pain for 5 days. Chlorhexidine 0.12% rinse BID starting tomorrow. Patient takes warfarin 5mg daily — INR checked, 2.3. Also on metformin 1000mg BID and lisinopril 20mg daily.`,
  },
];

// ─── Status Badge Component ──────────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-green-100 text-green-800',
  new: 'bg-blue-100 text-blue-800',
  discontinued: 'bg-red-100 text-red-800',
  held: 'bg-yellow-100 text-yellow-800',
  changed: 'bg-purple-100 text-purple-800',
  allergic: 'bg-red-200 text-red-900',
  unknown: 'bg-gray-100 text-gray-600',
};

function StatusBadge({ status }: { status: string }) {
  const colorClass = STATUS_COLORS[status] || STATUS_COLORS.unknown;
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>
      {status}
    </span>
  );
}

// ─── Confidence Bar ──────────────────────────────────────────

function ConfidenceIndicator({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2" title={`${pct}% confidence`}>
      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500 tabular-nums">{pct}%</span>
    </div>
  );
}

// ─── Main Page Component ─────────────────────────────────────

export function MedicationExtractor() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<MedicationExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<keyof MedicationResult>('confidence');
  const [sortAsc, setSortAsc] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      // Simulated API call — in production, use:
      // const data = await extractMedications(text);
      const response = await fetch('/api/v1/medications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data: MedicationExtractionResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed');
    } finally {
      setLoading(false);
    }
  }, [text]);

  const loadSample = useCallback((sample: typeof SAMPLE_NOTES[number]) => {
    setText(sample.text);
    setResults(null);
    setError(null);
  }, []);

  // ── Sort & Filter ──────────────────────────────────────────

  const sortedMeds = results
    ? [...results.medications]
        .filter((m) => statusFilter === 'all' || m.status === statusFilter)
        .sort((a, b) => {
          const aVal = a[sortField];
          const bVal = b[sortField];
          if (typeof aVal === 'string' && typeof bVal === 'string') {
            return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
          }
          return sortAsc
            ? (aVal as number) - (bVal as number)
            : (bVal as number) - (aVal as number);
        })
    : [];

  const toggleSort = (field: keyof MedicationResult) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const uniqueStatuses = results
    ? Array.from(new Set(results.medications.map((m) => m.status)))
    : [];

  // ── Render ─────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Medication Extractor</h1>
        <p className="mt-1 text-sm text-gray-500">
          Extract structured medication data from clinical text — drug names, dosages,
          routes, frequencies, durations, PRN status, and clinical status.
        </p>
      </div>

      {/* Sample Buttons */}
      <div className="flex flex-wrap gap-2">
        <span className="text-sm font-medium text-gray-700 self-center">Load sample:</span>
        {SAMPLE_NOTES.map((sample) => (
          <button
            key={sample.label}
            onClick={() => loadSample(sample)}
            className="px-3 py-1.5 text-sm bg-indigo-50 text-indigo-700 rounded-md hover:bg-indigo-100 transition-colors"
          >
            {sample.label}
          </button>
        ))}
      </div>

      {/* Input */}
      <div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste clinical text containing medication information..."
          rows={8}
          className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">
            {text.length > 0 ? `${text.split(/\s+/).filter(Boolean).length} words` : ''}
          </span>
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || loading}
            className="px-6 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Extracting…
              </span>
            ) : (
              'Extract Medications'
            )}
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
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Total Medications</p>
              <p className="mt-1 text-2xl font-bold text-gray-900">{results.count}</p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Active</p>
              <p className="mt-1 text-2xl font-bold text-green-600">
                {results.medications.filter((m) => m.status === 'active').length}
              </p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">PRN</p>
              <p className="mt-1 text-2xl font-bold text-blue-600">
                {results.medications.filter((m) => m.prn).length}
              </p>
            </div>
            <div className="bg-white rounded-lg border p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Processing Time</p>
              <p className="mt-1 text-2xl font-bold text-gray-600">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
            </div>
          </div>

          {/* Filter */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Filter by status:</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-1.5 text-sm"
            >
              <option value="all">All ({results.count})</option>
              {uniqueStatuses.map((s) => (
                <option key={s} value={s}>
                  {s} ({results.medications.filter((m) => m.status === s).length})
                </option>
              ))}
            </select>
          </div>

          {/* Table */}
          <div className="overflow-x-auto bg-white rounded-lg border">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {[
                    { key: 'drug_name', label: 'Medication' },
                    { key: 'dosage', label: 'Dosage' },
                    { key: 'route', label: 'Route' },
                    { key: 'frequency', label: 'Frequency' },
                    { key: 'status', label: 'Status' },
                    { key: 'confidence', label: 'Confidence' },
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      onClick={() => toggleSort(key as keyof MedicationResult)}
                      className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    >
                      {label}
                      {sortField === key && (
                        <span className="ml-1">{sortAsc ? '↑' : '↓'}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {sortedMeds.map((med, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{med.drug_name}</div>
                      {med.generic_name && med.generic_name !== med.drug_name && (
                        <div className="text-xs text-gray-500">
                          Generic: {med.generic_name}
                        </div>
                      )}
                      {med.indication && (
                        <div className="text-xs text-indigo-600">for {med.indication}</div>
                      )}
                      {med.prn && (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700 mt-1">
                          PRN
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{med.dosage || '—'}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{med.route || '—'}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {med.frequency || '—'}
                      {med.duration && (
                        <div className="text-xs text-gray-500">{med.duration}</div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={med.status} />
                    </td>
                    <td className="px-4 py-3">
                      <ConfidenceIndicator value={med.confidence} />
                    </td>
                  </tr>
                ))}
                {sortedMeds.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-sm text-gray-500">
                      No medications match the current filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
