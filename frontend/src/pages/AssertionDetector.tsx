/**
 * AssertionDetector page — Classify entity assertion status using
 * ConText/NegEx-inspired negation detection.
 *
 * Allows users to paste clinical text, highlight an entity span, and
 * detect whether the entity is present, absent (negated), possible,
 * conditional, hypothetical, or family history.  Supports batch entity
 * analysis with colour-coded status badges and confidence scoring.
 */

import { useState, useCallback } from 'react';
import type { AssertionResult } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'H&P Note',
    text: `HISTORY OF PRESENT ILLNESS:
Patient is a 58-year-old male presenting with chest pain and shortness of breath.
Patient denies nausea, vomiting, or diaphoresis.
No fever or chills.  No cough or hemoptysis.
Family history of coronary artery disease — father had MI at age 55.
Mother had diabetes mellitus.

ASSESSMENT:
1. Possible acute coronary syndrome — will obtain troponins and ECG.
2. Rule out pulmonary embolism.
3. Hypertension — stable on current medications.
4. If symptoms worsen, consider cardiac catheterisation.`,
    entities: [
      { text: 'chest pain', start: 72, end: 82 },
      { text: 'shortness of breath', start: 87, end: 106 },
      { text: 'nausea', start: 123, end: 129 },
      { text: 'vomiting', start: 131, end: 139 },
      { text: 'fever', start: 160, end: 165 },
      { text: 'coronary artery disease', start: 218, end: 241 },
      { text: 'MI', start: 261, end: 263 },
      { text: 'diabetes mellitus', start: 285, end: 302 },
      { text: 'acute coronary syndrome', start: 329, end: 352 },
      { text: 'pulmonary embolism', start: 395, end: 413 },
      { text: 'cardiac catheterisation', start: 494, end: 517 },
    ],
  },
  {
    label: 'Discharge Summary',
    text: `DISCHARGE DIAGNOSIS:
1. Community-acquired pneumonia, resolved.
2. Type 2 diabetes mellitus, controlled on metformin.
3. History of deep vein thrombosis (2019), currently on warfarin.

HOSPITAL COURSE:
Patient was admitted with pneumonia and treated with IV antibiotics for 5 days.
No evidence of empyema on imaging.  Patient does not have COPD.
Will start physical therapy if patient develops deconditioning.
Should symptoms return, consider readmission.
Patient tolerated oral diet without dysphagia.`,
    entities: [
      { text: 'pneumonia', start: 24, end: 33 },
      { text: 'Type 2 diabetes mellitus', start: 45, end: 69 },
      { text: 'deep vein thrombosis', start: 101, end: 121 },
      { text: 'empyema', start: 262, end: 269 },
      { text: 'COPD', start: 300, end: 304 },
      { text: 'deconditioning', start: 352, end: 366 },
      { text: 'dysphagia', start: 432, end: 441 },
    ],
  },
  {
    label: 'Dental Progress Note',
    text: `DENTAL PROGRESS NOTE:
Patient presents for evaluation of tooth #30.
No pain reported at this time.
Patient denies sensitivity to hot or cold.
Possible crack detected on distal surface.
Family history of periodontal disease — mother lost teeth by age 50.
If infection develops, will prescribe amoxicillin.
No caries detected on bitewing radiographs.
Plan for crown preparation next visit.`,
    entities: [
      { text: 'pain', start: 67, end: 71 },
      { text: 'sensitivity', start: 107, end: 118 },
      { text: 'crack', start: 144, end: 149 },
      { text: 'periodontal disease', start: 198, end: 217 },
      { text: 'infection', start: 258, end: 267 },
      { text: 'caries', start: 307, end: 313 },
    ],
  },
];

// ─── Status Styling ──────────────────────────────────────────

const STATUS_STYLES: Record<string, { label: string; color: string; icon: string }> = {
  present: {
    label: 'Present',
    color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    icon: '✅',
  },
  absent: {
    label: 'Absent (Negated)',
    color: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
    icon: '❌',
  },
  possible: {
    label: 'Possible',
    color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
    icon: '❓',
  },
  conditional: {
    label: 'Conditional',
    color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
    icon: '⚡',
  },
  hypothetical: {
    label: 'Hypothetical',
    color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    icon: '💭',
  },
  family: {
    label: 'Family History',
    color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
    icon: '👨‍👩‍👧',
  },
};

// ─── Types ───────────────────────────────────────────────────

interface EntityInput {
  text: string;
  start: number;
  end: number;
}

interface EntityAssertionResult {
  entity: EntityInput;
  assertion: AssertionResult;
}

// ─── Component ───────────────────────────────────────────────

export function AssertionDetector() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<EntityAssertionResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSampleIdx, setActiveSampleIdx] = useState<number | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  // Use pre-defined entities from sample notes
  const currentEntities: EntityInput[] =
    activeSampleIdx !== null ? SAMPLE_NOTES[activeSampleIdx].entities : [];

  const handleAnalyze = useCallback(async () => {
    if (!text.trim() || currentEntities.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const { detectAssertion } = await import('../services/clinical');
      const promises = currentEntities.map(async (entity) => {
        const assertion = await detectAssertion(text, entity.start, entity.end);
        return { entity, assertion };
      });
      const data = await Promise.all(promises);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [text, currentEntities]);

  const loadSample = (idx: number) => {
    setText(SAMPLE_NOTES[idx].text);
    setActiveSampleIdx(idx);
    setResults(null);
    setError(null);
  };

  // Compute status counts
  const statusCounts: Record<string, number> = {};
  if (results) {
    for (const r of results) {
      statusCounts[r.assertion.status] = (statusCounts[r.assertion.status] || 0) + 1;
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Assertion Detector
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Classify clinical entity assertions as present, absent, possible,
          conditional, hypothetical, or family history using ConText/NegEx rules.
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-3">
          <label
            htmlFor="assertion-input"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Clinical Text
          </label>
          <span className="text-xs text-gray-400">{wordCount} words</span>
        </div>

        <textarea
          id="assertion-input"
          rows={8}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setActiveSampleIdx(null);
          }}
          placeholder="Paste a clinical note, then load a sample to use pre-defined entities..."
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-3 text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />

        {/* Sample Notes */}
        <div className="mt-3 flex flex-wrap gap-2">
          {SAMPLE_NOTES.map((sample, idx) => (
            <button
              key={sample.label}
              onClick={() => loadSample(idx)}
              className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                activeSampleIdx === idx
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {sample.label}
            </button>
          ))}
        </div>

        {/* Entity Count */}
        {currentEntities.length > 0 && (
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            {currentEntities.length} entities will be analysed for assertion status.
          </p>
        )}

        {/* Analyze Button */}
        <div className="mt-4">
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim() || currentEntities.length === 0}
            className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Analysing…' : 'Detect Assertions'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Status Legend */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Assertion Status Legend
        </h3>
        <div className="flex flex-wrap gap-3">
          {Object.entries(STATUS_STYLES).map(([key, style]) => (
            <div
              key={key}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${style.color}`}
            >
              <span>{style.icon}</span>
              <span>{style.label}</span>
              {results && statusCounts[key] ? (
                <span className="ml-1 font-bold">({statusCounts[key]})</span>
              ) : null}
            </div>
          ))}
        </div>
      </div>

      {/* Results */}
      {results && results.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Assertion Results ({results.length} entities)
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Entity
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Trigger
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {results.map((r, idx) => {
                  const style = STATUS_STYLES[r.assertion.status] || STATUS_STYLES.present;
                  return (
                    <tr
                      key={`${r.entity.start}-${r.entity.end}-${idx}`}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700/50"
                    >
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-white font-medium">
                        {r.entity.text}
                        <span className="ml-2 text-xs text-gray-400 font-mono">
                          [{r.entity.start}:{r.entity.end}]
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${style.color}`}
                        >
                          {style.icon} {style.label}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-300">
                        {r.assertion.trigger_text ? (
                          <code className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-xs">
                            {r.assertion.trigger_text}
                          </code>
                        ) : (
                          <span className="text-gray-400 text-xs">default (present)</span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-blue-500 rounded-full"
                              style={{ width: `${r.assertion.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {(r.assertion.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
