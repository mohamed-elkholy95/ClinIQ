/**
 * ComorbidityCalculator page — Charlson Comorbidity Index (CCI) calculator.
 *
 * Quantifies disease burden from ICD-10-CM codes and/or free-text clinical
 * narratives using the Charlson–Deyo adaptation (Charlson 1987, Quan 2005).
 * Covers 17 disease categories with integer weights (1–6), hierarchical
 * exclusion rules, age adjustment, and 10-year mortality estimation.
 */

import { useState, useCallback } from 'react';
import type { ComorbidityResult, ComorbidityCategory } from '../types/clinical';

// ─── Sample Cases ────────────────────────────────────────────

const SAMPLE_CASES = [
  {
    label: 'Complex Patient',
    text: `PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus with peripheral neuropathy (ICD: E11.42)
2. Congestive Heart Failure, NYHA Class III (ICD: I50.23)
3. Chronic Kidney Disease Stage IV, GFR 22 mL/min (ICD: N18.4)
4. COPD with acute exacerbation (ICD: J44.1)
5. History of cerebrovascular accident (2020) with residual left hemiparesis
6. Peripheral vascular disease — s/p right fem-pop bypass 2019
7. Moderate hepatic fibrosis (Metavir F2) secondary to NASH
8. Atrial fibrillation on warfarin
9. Hypertension — well controlled on 3 agents
10. Peptic ulcer disease — no recent bleed`,
    age: 72,
    codes: ['E11.42', 'I50.23', 'N18.4', 'J44.1', 'I63.9', 'I73.9', 'K74.0', 'K27.9'],
  },
  {
    label: 'Healthy Adult',
    text: `PAST MEDICAL HISTORY:
1. Hypertension — well controlled on lisinopril 10mg
2. Seasonal allergic rhinitis
3. Appendectomy (2005)

No diabetes, no heart disease, no lung disease, no liver disease, no kidney disease.
No history of cancer. No history of stroke or TIA.`,
    age: 45,
    codes: ['I10'],
  },
  {
    label: 'Oncology Patient',
    text: `PAST MEDICAL HISTORY:
1. Metastatic colon adenocarcinoma — liver and lung metastases (stage IV)
2. Type 2 Diabetes Mellitus — on insulin
3. Moderate COPD — on inhalers
4. Peripheral neuropathy — chemotherapy-induced
5. History of deep vein thrombosis on anticoagulation
6. Mild chronic kidney disease, GFR 55
7. Dementia, mild — caregiver provides assistance with medications`,
    age: 78,
    codes: ['C18.9', 'C78.7', 'C78.0', 'E11.9', 'J44.1', 'G62.0', 'I82.90', 'N18.3', 'F03.90'],
  },
];

// ─── Risk Group Styling ──────────────────────────────────────

const RISK_STYLES: Record<string, { color: string; bg: string; label: string }> = {
  low: { color: 'text-green-700 dark:text-green-300', bg: 'bg-green-100 dark:bg-green-900/30', label: 'Low Risk' },
  mild: { color: 'text-blue-700 dark:text-blue-300', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Mild Risk' },
  moderate: { color: 'text-amber-700 dark:text-amber-300', bg: 'bg-amber-100 dark:bg-amber-900/30', label: 'Moderate Risk' },
  severe: { color: 'text-red-700 dark:text-red-300', bg: 'bg-red-100 dark:bg-red-900/30', label: 'Severe Risk' },
};

// ─── Category Weight Badge ───────────────────────────────────

function WeightBadge({ weight }: { weight: number }) {
  const color =
    weight >= 6
      ? 'bg-red-600 text-white'
      : weight >= 3
        ? 'bg-orange-500 text-white'
        : weight >= 2
          ? 'bg-amber-400 text-amber-900'
          : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300';

  return (
    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${color}`}>
      {weight}
    </span>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function ComorbidityCalculator() {
  const [text, setText] = useState('');
  const [codesInput, setCodesInput] = useState('');
  const [age, setAge] = useState<number | ''>('');
  const [results, setResults] = useState<ComorbidityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCalculate = useCallback(async () => {
    if (!text.trim() && !codesInput.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const codes = codesInput
        .split(/[,;\s]+/)
        .map((c) => c.trim())
        .filter(Boolean);

      const body: Record<string, unknown> = {};
      if (text.trim()) body.text = text;
      if (codes.length > 0) body.icd_codes = codes;
      if (age !== '') body.age = age;

      const response = await fetch('/api/v1/comorbidity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data: ComorbidityResult = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calculation failed');
    } finally {
      setLoading(false);
    }
  }, [text, codesInput, age]);

  const loadSample = (idx: number) => {
    const sample = SAMPLE_CASES[idx];
    setText(sample.text);
    setCodesInput(sample.codes.join(', '));
    setAge(sample.age);
    setResults(null);
    setError(null);
  };

  const riskStyle = RISK_STYLES[results?.risk_group ?? 'low'] ?? RISK_STYLES.low;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">
          Charlson Comorbidity Index
        </h1>
        <p className="mt-1 text-text-secondary">
          Calculate disease burden from ICD-10 codes and clinical text. 17 disease categories, age adjustment, and 10-year mortality estimation.
        </p>
      </div>

      {/* Input Section */}
      <div className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-primary">Patient Data</h2>
          <div className="flex gap-2">
            {SAMPLE_CASES.map((s, i) => (
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
          placeholder="Paste medical history or clinical note..."
          rows={6}
          className="w-full px-4 py-3 rounded-lg border border-border bg-surface text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 resize-y font-mono text-sm"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              ICD-10-CM Codes (comma-separated)
            </label>
            <input
              type="text"
              value={codesInput}
              onChange={(e) => setCodesInput(e.target.value)}
              placeholder="E11.42, I50.23, N18.4..."
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm font-mono"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              Patient Age (for age adjustment)
            </label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value ? parseInt(e.target.value) : '')}
              placeholder="e.g. 72"
              min={0}
              max={130}
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm"
            />
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleCalculate}
            disabled={(!text.trim() && !codesInput.trim()) || loading}
            className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-500 text-white hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Calculating...' : 'Calculate CCI'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card p-4 border-red-300 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {results && (
        <>
          {/* Score Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className={`card p-6 text-center border-2 ${riskStyle.bg}`}>
              <div className="text-5xl font-black text-text-primary">{results.total_score}</div>
              <div className="text-xs text-text-muted mt-2">CCI Score</div>
              <div className={`text-sm font-semibold mt-1 ${riskStyle.color}`}>
                {riskStyle.label}
              </div>
            </div>

            {results.age_adjusted_score != null && (
              <div className="card p-6 text-center">
                <div className="text-4xl font-bold text-text-primary">
                  {results.age_adjusted_score}
                </div>
                <div className="text-xs text-text-muted mt-2">Age-Adjusted Score</div>
                {age !== '' && (
                  <div className="text-xs text-text-secondary mt-1">
                    +{(results.age_adjusted_score ?? 0) - results.total_score} age points (age {age})
                  </div>
                )}
              </div>
            )}

            <div className="card p-6 text-center">
              <div className="text-4xl font-bold text-text-primary">
                {results.estimated_mortality != null
                  ? `${Math.round(results.estimated_mortality * 100)}%`
                  : '—'}
              </div>
              <div className="text-xs text-text-muted mt-2">10-Year Mortality</div>
              <div className="text-[10px] text-text-muted mt-1">
                Charlson exponential survival
              </div>
            </div>

            <div className="card p-6 text-center">
              <div className="text-4xl font-bold text-text-primary">
                {results.categories?.length ?? 0}
              </div>
              <div className="text-xs text-text-muted mt-2">Categories Identified</div>
              <div className="text-[10px] text-text-muted mt-1">of 17 possible</div>
            </div>
          </div>

          {/* Category Breakdown */}
          {results.categories && results.categories.length > 0 && (
            <div className="card p-5 space-y-4">
              <h3 className="text-sm font-semibold text-text-primary">
                Disease Categories Identified
              </h3>
              <div className="space-y-3">
                {results.categories
                  .sort((a: ComorbidityCategory, b: ComorbidityCategory) => b.weight - a.weight)
                  .map((cat: ComorbidityCategory, idx: number) => (
                    <div
                      key={idx}
                      className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800/30"
                    >
                      <WeightBadge weight={cat.weight} />
                      <div className="flex-1">
                        <div className="text-sm font-medium text-text-primary">
                          {cat.name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        {cat.description && (
                          <div className="text-xs text-text-muted">{cat.description}</div>
                        )}
                      </div>
                      {cat.matched_codes && cat.matched_codes.length > 0 && (
                        <div className="flex gap-1 flex-wrap">
                          {cat.matched_codes.map((code: string, ci: number) => (
                            <span
                              key={ci}
                              className="text-[10px] px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-text-muted font-mono"
                            >
                              {code}
                            </span>
                          ))}
                        </div>
                      )}
                      {cat.confidence != null && (
                        <span className="text-xs text-text-muted">
                          {Math.round(cat.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* CCI Score Guide */}
          <div className="card p-5">
            <h3 className="text-xs font-semibold text-text-muted mb-3">RISK GROUP REFERENCE</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(RISK_STYLES).map(([key, style]) => {
                const ranges: Record<string, string> = {
                  low: 'CCI 0',
                  mild: 'CCI 1–2',
                  moderate: 'CCI 3–4',
                  severe: 'CCI 5+',
                };
                return (
                  <div
                    key={key}
                    className={`p-3 rounded-lg text-center ${style.bg} ${
                      results.risk_group === key ? 'ring-2 ring-primary-500' : ''
                    }`}
                  >
                    <div className={`text-sm font-semibold ${style.color}`}>{style.label}</div>
                    <div className="text-xs text-text-muted mt-0.5">{ranges[key]}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {results.processing_time_ms != null && (
            <p className="text-xs text-text-muted text-right">
              Calculated in {results.processing_time_ms.toFixed(1)}ms
            </p>
          )}
        </>
      )}
    </div>
  );
}
