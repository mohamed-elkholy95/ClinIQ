/**
 * QualityAnalyzer page — Evaluate clinical note quality before NLP inference.
 *
 * Scores notes across 5 dimensions (completeness, readability, structure,
 * information density, consistency) and produces a letter grade (A–F) with
 * actionable findings and recommendations.  Helps clinicians and data teams
 * identify low-quality notes that may degrade NLP pipeline accuracy.
 */

import { useState, useCallback } from 'react';
import type { QualityReport, QualityDimensionScore, QualityFinding } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Good H&P Note',
    text: `CHIEF COMPLAINT: Chest pain, shortness of breath for 2 days.

HISTORY OF PRESENT ILLNESS:
Mr. Johnson is a 62-year-old male with past medical history significant for hypertension, hyperlipidemia, type 2 diabetes mellitus, and prior myocardial infarction (2019) who presents to the emergency department with 2 days of intermittent substernal chest pain. The pain is described as a squeezing sensation, 6/10 severity, radiating to the left arm, worse with exertion, improved with rest and sublingual nitroglycerin. Associated symptoms include dyspnea on exertion, diaphoresis, and mild nausea. Patient denies syncope, palpitations, or leg swelling. He reports compliance with home medications.

PAST MEDICAL HISTORY:
1. Hypertension — diagnosed 2010
2. Hyperlipidemia
3. Type 2 Diabetes Mellitus — A1c 7.2% (3 months ago)
4. Prior NSTEMI (2019) — s/p PCI with DES to LAD
5. GERD

MEDICATIONS:
Aspirin 81mg daily, Metoprolol succinate 50mg daily, Lisinopril 20mg daily, Atorvastatin 80mg daily, Metformin 1000mg BID, Omeprazole 20mg daily.

ALLERGIES: NKDA

ASSESSMENT AND PLAN:
1. Acute chest pain — Concerning for ACS given cardiac history. Serial troponins q6h. 12-lead ECG shows ST depression V4-V6. Cardiology consult for possible cardiac catheterization.
2. Dyspnea — Likely cardiac etiology. CXR pending. BNP ordered.
3. Diabetes — Hold metformin if cath planned. Sliding scale insulin.`,
  },
  {
    label: 'Poor Note',
    text: `pt c/o pain. hx of htn dm cad. gave meds. will f/u.

bp ok hr ok temp ok.

dx: chest pain
rx: see above`,
  },
  {
    label: 'Moderate Note',
    text: `PROGRESS NOTE:
Patient is a 45-year-old female admitted for pneumonia.

Patient reports feeling better today. Cough is improving. Still has mild dyspnea with ambulation. Denies fever, chills, or hemoptysis. Tolerating oral intake. Ambulating with assistance.

VITALS: BP 128/82, HR 88, Temp 99.1°F, RR 18, SpO2 95% on RA.

ASSESSMENT:
Community-acquired pneumonia — improving on IV antibiotics.

PLAN:
Continue levofloxacin. Transition to oral antibiotics tomorrow if afebrile. Pulmonary toilet. Encourage incentive spirometry.`,
  },
];

// ─── Grade Styling ───────────────────────────────────────────

const GRADE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  A: { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-300', border: 'border-green-400' },
  B: { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-300', border: 'border-blue-400' },
  C: { bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-300', border: 'border-yellow-400' },
  D: { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-300', border: 'border-orange-400' },
  F: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-300', border: 'border-red-400' },
};

const SEVERITY_ICONS: Record<string, string> = {
  critical: '🔴',
  warning: '🟡',
  info: '🔵',
};

const DIMENSION_ICONS: Record<string, string> = {
  completeness: '📋',
  readability: '📖',
  structure: '🏗️',
  information_density: '📊',
  consistency: '🔄',
};

// ─── Score Ring Component ────────────────────────────────────

function ScoreRing({ score, size = 80 }: { score: number; size?: number }) {
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color =
    score >= 90 ? '#22c55e' : score >= 80 ? '#3b82f6' : score >= 70 ? '#eab308' : score >= 60 ? '#f97316' : '#ef4444';

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeWidth="4"
        className="text-gray-200 dark:text-gray-700"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
      />
    </svg>
  );
}

// ─── Dimension Bar Component ─────────────────────────────────

function DimensionBar({ dimension }: { dimension: QualityDimensionScore }) {
  const score = Math.round(dimension.score * 100);
  const color =
    score >= 90 ? 'bg-green-500' : score >= 70 ? 'bg-blue-500' : score >= 50 ? 'bg-amber-500' : 'bg-red-500';
  const dimName = dimension.dimension ?? '';
  const icon = DIMENSION_ICONS[dimName] ?? '📋';
  const label = dimName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-sm text-text-primary">
          {icon} {label}
        </span>
        <span className="text-sm font-semibold text-text-primary">{score}%</span>
      </div>
      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${score}%` }} />
      </div>
      {dimension.findings && dimension.findings.length > 0 && (
        <div className="mt-1 space-y-0.5">
          {dimension.findings.map((f: QualityFinding, i: number) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-text-secondary">
              <span>{SEVERITY_ICONS[f.severity] ?? '🔵'}</span>
              <span>{f.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function QualityAnalyzer() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<QualityReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data: QualityReport = await response.json();
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
  };

  const gradeStyle = GRADE_COLORS[results?.grade ?? 'C'] ?? GRADE_COLORS.C;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Note Quality Analysis</h1>
        <p className="mt-1 text-text-secondary">
          Evaluate clinical note quality across 5 dimensions before running NLP inference.
        </p>
      </div>

      {/* Input */}
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
          placeholder="Paste clinical note to evaluate quality..."
          rows={8}
          className="w-full px-4 py-3 rounded-lg border border-border bg-surface text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 resize-y font-mono text-sm"
        />

        <div className="flex items-center justify-end gap-3">
          <span className="text-xs text-text-muted">
            {text.split(/\s+/).filter(Boolean).length} words
          </span>
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || loading}
            className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-500 text-white hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Analyzing...' : 'Analyze Quality'}
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
          {/* Grade + Overall Score */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`card p-6 text-center border-2 ${gradeStyle.border}`}>
              <div className={`text-6xl font-black ${gradeStyle.text}`}>{results.grade}</div>
              <div className="text-sm text-text-muted mt-2">Overall Grade</div>
            </div>
            <div className="card p-6 flex flex-col items-center justify-center">
              <div className="relative">
                <ScoreRing score={Math.round(results.overall_score)} size={100} />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xl font-bold text-text-primary">
                    {Math.round(results.overall_score)}
                  </span>
                </div>
              </div>
              <div className="text-sm text-text-muted mt-2">Overall Score</div>
            </div>
            <div className="card p-6 space-y-3">
              <h3 className="text-xs font-semibold text-text-muted">QUICK STATS</h3>
              <div className="space-y-2 text-sm text-text-secondary">
                <div className="flex justify-between">
                  <span>Word count</span>
                  <span className="font-medium text-text-primary">
                    {text.split(/\s+/).filter(Boolean).length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Dimensions scored</span>
                  <span className="font-medium text-text-primary">
                    {results.dimensions?.length ?? 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Findings</span>
                  <span className="font-medium text-text-primary">
                    {(results.dimensions ?? []).reduce(
                      (sum: number, d: QualityDimensionScore) => sum + (d.findings?.length ?? 0),
                      0
                    )}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Dimension Breakdown */}
          <div className="card p-5 space-y-5">
            <h3 className="text-sm font-semibold text-text-primary">Dimension Scores</h3>
            {(results.dimensions ?? []).map((dim: QualityDimensionScore, idx: number) => (
              <DimensionBar key={idx} dimension={dim} />
            ))}
          </div>

          {/* Recommendations */}
          {results.recommendations && results.recommendations.length > 0 && (
            <div className="card p-5 space-y-3">
              <h3 className="text-sm font-semibold text-text-primary">📝 Recommendations</h3>
              <ol className="space-y-2">
                {results.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-text-secondary">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 text-xs flex items-center justify-center font-medium">
                      {idx + 1}
                    </span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {/* Processing time */}
          {results.analysis_time_ms != null && (
            <p className="text-xs text-text-muted text-right">
              Analyzed in {results.analysis_time_ms.toFixed(1)}ms
            </p>
          )}
        </>
      )}
    </div>
  );
}
