/**
 * TemporalExtractor page — Extract and normalise temporal expressions
 * from clinical free text.
 *
 * Detects 6 temporal expression types (date, duration, relative_time,
 * age, postoperative_day, frequency) with normalised ISO-8601 values
 * and character-offset spans for highlighting.  Results are displayed
 * in a sortable table with type-coloured badges and confidence bars.
 */

import { useState, useCallback } from 'react';
import type { TemporalExpression, TemporalExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Discharge Summary',
    text: `DISCHARGE SUMMARY
Date of Admission: March 15, 2025
Date of Discharge: March 22, 2025
Hospital Day 7.

HOSPITAL COURSE:
Patient is a 67-year-old male admitted on 3/15/2025 with acute myocardial infarction.
PCI was performed on hospital day 1.  On postoperative day 2, patient developed
atrial fibrillation which resolved within 48 hours with amiodarone drip.
Echocardiogram on 3/18/2025 showed EF 35%.  Patient was stable for 3 days prior
to discharge.

MEDICATIONS AT DISCHARGE:
- Aspirin 81 mg daily
- Metoprolol 25 mg twice daily
- Lisinopril 10 mg once daily
- Atorvastatin 80 mg at bedtime

FOLLOW-UP:
- Cardiology clinic in 2 weeks (by April 5, 2025)
- Repeat echocardiogram in 3 months
- Labs (BMP, CBC) in 1 week
- Cardiac rehab starting next Monday`,
  },
  {
    label: 'Progress Note',
    text: `PROGRESS NOTE — 01/10/2025
S: Patient reports improving pain since yesterday.  States the headache
started 5 days ago and has been occurring every 4-6 hours.  Last dose
of acetaminophen was 3 hours ago.  Patient has had migraines since age 14.
Family history of stroke — mother had CVA at age 72 in 2019.

O: T 98.6°F, HR 78, BP 132/84.  Patient appears comfortable today.
Post-procedure day 1 — wound site clean and dry.

A/P:
1. Migraine — continue current regimen.  If no improvement in 48 hours,
   consider triptans.
2. Post-procedure monitoring — recheck wound in 72 hours.
3. Return to clinic in 10 days for suture removal.
4. Annual wellness visit due in 6 months (July 2025).`,
  },
  {
    label: 'Dental Treatment Plan',
    text: `TREATMENT PLAN — 02/20/2025
Patient: 45-year-old female, last dental visit was 18 months ago (August 2023).
Chief complaint: tooth pain for the past 2 weeks.

FINDINGS:
Periapical abscess #19 identified on PA radiograph taken today.
Tooth has had a crown since 2018 (approximately 7 years).
Patient reports intermittent pain starting around February 6, 2025.

PLAN:
Phase 1 — Root canal therapy #19, scheduled for next Thursday (02/27/2025).
  Prescribe amoxicillin 500 mg three times daily for 7 days starting today.
  Ibuprofen 600 mg every 6 hours as needed for pain.
Phase 2 — Post and core buildup, 2 weeks after RCT completion.
Phase 3 — New PFM crown, 3-4 weeks after post placement.
  
Total estimated treatment time: 6-8 weeks.
Recall appointment in 6 months.`,
  },
];

// ─── Type Badge Colours ──────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  date: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  duration: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
  relative_time: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
  age: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  postoperative_day: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  frequency: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
};

const TYPE_LABELS: Record<string, string> = {
  date: '📅 Date',
  duration: '⏱️ Duration',
  relative_time: '🔄 Relative',
  age: '🎂 Age',
  postoperative_day: '🏥 Post-Op Day',
  frequency: '🔁 Frequency',
};

// ─── Component ───────────────────────────────────────────────

export function TemporalExtractor() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<TemporalExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { extractTemporal } = await import('../services/clinical');
      const data = await extractTemporal(text);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [text]);

  const loadSample = (idx: number) => {
    setText(SAMPLE_NOTES[idx].text);
    setResults(null);
    setError(null);
    setActiveFilter(null);
  };

  // Compute type counts for filter bar
  const typeCounts: Record<string, number> = {};
  if (results) {
    for (const expr of results.expressions) {
      typeCounts[expr.type] = (typeCounts[expr.type] || 0) + 1;
    }
  }

  const activeTypes = Object.keys(typeCounts);

  const filteredExpressions = results
    ? activeFilter
      ? results.expressions.filter((e) => e.type === activeFilter)
      : results.expressions
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Temporal Expression Extractor
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Extract dates, durations, relative times, ages, post-operative days, and
          frequencies from clinical notes.
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-3">
          <label
            htmlFor="temporal-input"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Clinical Text
          </label>
          <span className="text-xs text-gray-400">{wordCount} words</span>
        </div>

        <textarea
          id="temporal-input"
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a clinical note to extract temporal expressions..."
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-3 text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />

        {/* Sample Notes */}
        <div className="mt-3 flex flex-wrap gap-2">
          {SAMPLE_NOTES.map((sample, idx) => (
            <button
              key={sample.label}
              onClick={() => loadSample(idx)}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {sample.label}
            </button>
          ))}
        </div>

        {/* Analyze Button */}
        <div className="mt-4">
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Extracting…' : 'Extract Temporal Expressions'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-blue-600">{results.count}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Total Expressions</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-purple-600">{activeTypes.length}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Types Found</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-green-600">
                {typeCounts['date'] || 0}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Dates</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-gray-600 dark:text-gray-300">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Processing Time</p>
            </div>
          </div>

          {/* Type Filter Bar */}
          {activeTypes.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setActiveFilter(null)}
                className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                  activeFilter === null
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200'
                }`}
              >
                All ({results.count})
              </button>
              {activeTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => setActiveFilter(activeFilter === type ? null : type)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                    activeFilter === type
                      ? 'bg-blue-600 text-white'
                      : TYPE_COLORS[type] || 'bg-gray-100 text-gray-700'
                  }`}
                >
                  {TYPE_LABELS[type] || type} ({typeCounts[type]})
                </button>
              ))}
            </div>
          )}

          {/* Results Table */}
          {filteredExpressions.length > 0 ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Text
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Normalised Value
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Position
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {filteredExpressions.map((expr, idx) => (
                      <tr
                        key={`${expr.start_char}-${expr.end_char}-${idx}`}
                        className="hover:bg-gray-50 dark:hover:bg-gray-700/50"
                      >
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              TYPE_COLORS[expr.type] || 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            {TYPE_LABELS[expr.type] || expr.type}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-white font-mono">
                          &ldquo;{expr.text}&rdquo;
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {expr.normalized_value ? (
                            <code className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-blue-700 dark:text-blue-300 text-xs">
                              {expr.normalized_value}
                            </code>
                          ) : (
                            <span className="text-gray-400 text-xs">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400 font-mono">
                          [{expr.start_char}:{expr.end_char}]
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 rounded-full"
                                style={{ width: `${expr.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {(expr.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
              <p className="text-gray-500 dark:text-gray-400">
                No temporal expressions found matching the current filter.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
