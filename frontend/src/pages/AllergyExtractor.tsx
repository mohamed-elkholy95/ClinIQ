/**
 * AllergyExtractor page — Identify drug, food, and environmental allergies
 * from clinical free text.
 *
 * Uses the rule-based allergy extraction module (~150 allergen entries,
 * 30+ reaction patterns) to detect allergens, reactions, severity, and
 * assertion status (present/historical/tolerated).  Includes NKDA
 * detection and confidence-scored results displayed in a grouped table.
 */

import { useState, useCallback } from 'react';
import type { AllergyResult, AllergyExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Allergy List',
    text: `ALLERGIES:
1. Penicillin — anaphylaxis (confirmed 2018)
2. Sulfa drugs — rash, hives
3. Codeine — nausea, vomiting
4. Shellfish — urticaria, throat swelling
5. Latex — contact dermatitis
6. Aspirin — angioedema (severe)
7. Peanuts — anaphylactic shock

NKFA (No Known Food Allergies) — patient denies additional food sensitivities.
Patient tolerates cephalosporins without issue.`,
  },
  {
    label: 'H&P Note',
    text: `PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes, GERD
ALLERGIES: Patient reports allergy to amoxicillin causing rash. 
History of sulfonamide allergy with Stevens-Johnson syndrome.
Denies allergy to contrast dye. Previously allergic to eggs as a child — outgrown.
No known drug allergies to NSAIDs. Mother had penicillin allergy.
Patient is allergic to ibuprofen — GI upset.
Reports bee sting allergy — severe swelling, carries EpiPen.`,
  },
  {
    label: 'Dental Pre-Op',
    text: `PRE-OPERATIVE ASSESSMENT:
Allergies reviewed with patient:
- Lidocaine: reports dizziness and palpitations (possible vasovagal, not true allergy)
- Erythromycin: severe GI upset, nausea
- Nickel: contact dermatitis from jewelry
- Adhesive tape: skin irritation and blistering
Patient is not allergic to amoxicillin, clindamycin, or ibuprofen.
No known food allergies. NKDA confirmed in chart.
Note: Patient takes warfarin — coordinate with cardiologist for INR.`,
  },
];

// ─── Category & Severity Styling ─────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  drug: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
  food: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
  environmental: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
};

const SEVERITY_COLORS: Record<string, string> = {
  'life-threatening': 'bg-red-600 text-white',
  severe: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  moderate: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
  mild: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  unknown: 'bg-gray-100 text-gray-800 dark:bg-gray-800/30 dark:text-gray-300',
};

const ASSERTION_LABELS: Record<string, { label: string; color: string }> = {
  present: { label: 'Confirmed', color: 'text-red-600 dark:text-red-400' },
  historical: { label: 'Historical', color: 'text-amber-600 dark:text-amber-400' },
  tolerated: { label: 'Tolerated', color: 'text-green-600 dark:text-green-400' },
  negated: { label: 'Negated', color: 'text-gray-500 dark:text-gray-400' },
  family: { label: 'Family Hx', color: 'text-blue-600 dark:text-blue-400' },
};

// ─── Badge Components ────────────────────────────────────────

function CategoryBadge({ category }: { category: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${CATEGORY_COLORS[category] ?? CATEGORY_COLORS.drug}`}
    >
      {category === 'drug' ? '💊' : category === 'food' ? '🍽️' : '🌿'}{' '}
      {category}
    </span>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${SEVERITY_COLORS[severity] ?? SEVERITY_COLORS.unknown}`}
    >
      {severity === 'life-threatening' ? '⚠️ ' : ''}
      {severity}
    </span>
  );
}

// ─── Confidence Bar ──────────────────────────────────────────

function ConfidenceIndicator({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 90
      ? 'bg-green-500'
      : pct >= 70
        ? 'bg-blue-500'
        : pct >= 50
          ? 'bg-amber-500'
          : 'bg-red-500';

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-text-muted w-8 text-right">{pct}%</span>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function AllergyExtractor() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<AllergyExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/allergies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, min_confidence: minConfidence }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data: AllergyExtractionResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [text, minConfidence]);

  const loadSample = (idx: number) => {
    setText(SAMPLE_NOTES[idx].text);
    setResults(null);
    setError(null);
  };

  // Filter results by category
  const filteredAllergies =
    results?.allergies?.filter(
      (a: AllergyResult) => categoryFilter === 'all' || a.category === categoryFilter
    ) ?? [];

  // Group by category for summary cards
  const categoryCounts = (results?.allergies ?? []).reduce(
    (acc: Record<string, number>, a: AllergyResult) => {
      acc[a.category] = (acc[a.category] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const lifeThreatening = (results?.allergies ?? []).filter(
    (a: AllergyResult) => a.severity === 'life-threatening'
  ).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Allergy Extraction</h1>
        <p className="mt-1 text-text-secondary">
          Identify drug, food, and environmental allergies with reactions, severity, and assertion status.
        </p>
      </div>

      {/* Input Section */}
      <div className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-primary">Clinical Note</h2>
          <div className="flex gap-2">
            {SAMPLE_NOTES.map((s, i) => (
              <button
                key={i}
                onClick={() => loadSample(i)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste clinical note here..."
          rows={8}
          className="w-full px-4 py-3 rounded-lg border border-border bg-surface text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 resize-y font-mono text-sm"
        />

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <label className="text-xs text-text-muted">
              Min confidence: <strong>{Math.round(minConfidence * 100)}%</strong>
            </label>
            <input
              type="range"
              min={0.3}
              max={1}
              step={0.05}
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="w-32 accent-primary-500"
            />
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-text-muted">
              {text.split(/\s+/).filter(Boolean).length} words
            </span>
            <button
              onClick={handleAnalyze}
              disabled={!text.trim() || loading}
              className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-500 text-white hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing...' : 'Extract Allergies'}
            </button>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="card p-4 border-red-300 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-text-primary">
                {results.allergies?.length ?? 0}
              </div>
              <div className="text-xs text-text-muted mt-1">Total Detected</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-purple-600">
                {categoryCounts.drug ?? 0}
              </div>
              <div className="text-xs text-text-muted mt-1">💊 Drug</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-orange-600">
                {categoryCounts.food ?? 0}
              </div>
              <div className="text-xs text-text-muted mt-1">🍽️ Food</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-teal-600">
                {categoryCounts.environmental ?? 0}
              </div>
              <div className="text-xs text-text-muted mt-1">🌿 Environmental</div>
            </div>
            <div className="card p-4 text-center">
              <div className={`text-2xl font-bold ${lifeThreatening > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {lifeThreatening}
              </div>
              <div className="text-xs text-text-muted mt-1">⚠️ Life-threatening</div>
            </div>
          </div>

          {/* NKDA indicator */}
          {results.nkda_detected && (
            <div className="card p-4 bg-green-50 dark:bg-green-900/20 border-green-300 text-green-700 dark:text-green-300 text-sm font-medium">
              ✅ NKDA — No Known Drug Allergies documented in this note
            </div>
          )}

          {/* Filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">Filter:</span>
            {['all', 'drug', 'food', 'environmental'].map((cat) => (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  categoryFilter === cat
                    ? 'bg-primary-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {cat === 'all' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1)}
              </button>
            ))}
          </div>

          {/* Results Table */}
          <div className="card overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Allergen</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Category</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Reactions</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Severity</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted w-28">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filteredAllergies.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-text-muted">
                      No allergies found matching the current filter.
                    </td>
                  </tr>
                ) : (
                  filteredAllergies.map((allergy: AllergyResult, idx: number) => {
                    const assertion = ASSERTION_LABELS[allergy.assertion_status] ?? ASSERTION_LABELS.present;
                    return (
                      <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-800/30">
                        <td className="px-4 py-3 font-medium text-text-primary">
                          {allergy.allergen}
                        </td>
                        <td className="px-4 py-3">
                          <CategoryBadge category={allergy.category} />
                        </td>
                        <td className="px-4 py-3 text-text-secondary text-xs">
                          {allergy.reactions?.length ? allergy.reactions.join(', ') : '—'}
                        </td>
                        <td className="px-4 py-3">
                          <SeverityBadge severity={allergy.severity ?? 'unknown'} />
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-medium ${assertion.color}`}>
                            {assertion.label}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <ConfidenceIndicator value={allergy.confidence} />
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Processing time */}
          {results.processing_time_ms != null && (
            <p className="text-xs text-text-muted text-right">
              Processed in {results.processing_time_ms.toFixed(1)}ms
            </p>
          )}
        </>
      )}
    </div>
  );
}
