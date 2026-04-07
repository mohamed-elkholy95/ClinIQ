/**
 * PipelineExplorer page — Interactive enhanced pipeline configuration and analysis.
 *
 * Lets users toggle individual NLP modules on/off, submit clinical text,
 * and view the results from all 14 modules in an organized, collapsible
 * layout.  This page is the frontend counterpart to the
 * ``POST /analyze/enhanced`` endpoint.
 */

import { useState, useCallback } from 'react';
import type {
  EnhancedAnalysisConfig,
  EnhancedAnalysisResponse,
} from '../types/clinical';

// ─── Module Definitions ──────────────────────────────────────

interface ModuleDef {
  key: keyof EnhancedAnalysisConfig;
  label: string;
  icon: string;
  description: string;
  defaultEnabled: boolean;
  phase: 'pre-processing' | 'extraction';
}

const MODULES: ModuleDef[] = [
  // Phase 1 — Pre-processing
  {
    key: 'enable_classification',
    label: 'Document Classification',
    icon: '📋',
    description: 'Identify document type (discharge summary, progress note, etc.)',
    defaultEnabled: true,
    phase: 'pre-processing',
  },
  {
    key: 'enable_sections',
    label: 'Section Parsing',
    icon: '📑',
    description: 'Segment document into clinical sections (HPI, Assessment, Plan, etc.)',
    defaultEnabled: true,
    phase: 'pre-processing',
  },
  {
    key: 'enable_quality',
    label: 'Quality Analysis',
    icon: '✅',
    description: 'Evaluate note completeness, readability, and structure',
    defaultEnabled: true,
    phase: 'pre-processing',
  },
  {
    key: 'enable_deidentification',
    label: 'De-identification',
    icon: '🔒',
    description: 'Detect and redact PHI per HIPAA Safe Harbor (off by default)',
    defaultEnabled: false,
    phase: 'pre-processing',
  },
  {
    key: 'enable_abbreviations',
    label: 'Abbreviation Expansion',
    icon: '🔤',
    description: 'Detect and expand clinical abbreviations (HTN → hypertension)',
    defaultEnabled: true,
    phase: 'pre-processing',
  },
  // Phase 2 — Extraction & Scoring
  {
    key: 'enable_medications',
    label: 'Medication Extraction',
    icon: '💊',
    description: 'Extract drugs, dosages, routes, frequencies, and status',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_allergies',
    label: 'Allergy Extraction',
    icon: '⚠️',
    description: 'Identify allergens, reactions, severity, and NKDA status',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_vitals',
    label: 'Vital Signs',
    icon: '💓',
    description: 'Extract BP, HR, temp, SpO2, weight, height, BMI, pain',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_temporal',
    label: 'Temporal Extraction',
    icon: '⏰',
    description: 'Extract dates, durations, frequencies, and temporal relations',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_assertions',
    label: 'Assertion Detection',
    icon: '🔍',
    description: 'Classify entity status (present, absent, possible, family, etc.)',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_normalization',
    label: 'Concept Normalization',
    icon: '🏷️',
    description: 'Map entities to UMLS CUI, SNOMED-CT, RxNorm, ICD-10-CM codes',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_sdoh',
    label: 'SDoH Extraction',
    icon: '🏠',
    description: 'Detect social determinants of health across 8 domains',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_relations',
    label: 'Relation Extraction',
    icon: '🔗',
    description: 'Identify semantic relations (treats, causes, side_effect_of, etc.)',
    defaultEnabled: true,
    phase: 'extraction',
  },
  {
    key: 'enable_comorbidity',
    label: 'Comorbidity Scoring',
    icon: '📊',
    description: 'Calculate Charlson Comorbidity Index and mortality estimate',
    defaultEnabled: true,
    phase: 'extraction',
  },
];

// ─── Sample Note ─────────────────────────────────────────────

const SAMPLE_NOTE = `DISCHARGE SUMMARY

PATIENT: [Name Redacted]
DATE: 01/15/2025

CHIEF COMPLAINT:
Chest pain and shortness of breath for 3 days.

HISTORY OF PRESENT ILLNESS:
65-year-old male with PMH of HTN, T2DM, hyperlipidemia, and COPD presents with substernal
chest pain radiating to left arm, onset 3 days ago. Associated SOB and diaphoresis.
Denies fever, cough, or leg swelling. Positive for orthopnea and PND.

PAST MEDICAL HISTORY:
1. Hypertension — diagnosed 2010
2. Type 2 Diabetes Mellitus — HbA1c 7.8%
3. Hyperlipidemia
4. COPD — moderate severity
5. Osteoarthritis bilateral knees
6. History of MI in 2018

ALLERGIES:
Penicillin — rash and hives
Sulfa drugs — anaphylaxis
NKFA to food or environmental allergens

MEDICATIONS:
1. Metformin 1000mg PO BID
2. Lisinopril 20mg PO daily
3. Atorvastatin 80mg PO at bedtime
4. Aspirin 81mg PO daily
5. Metoprolol succinate 50mg PO daily
6. Tiotropium 18mcg inhaled daily
7. Albuterol 2 puffs q4-6h PRN SOB
8. Acetaminophen 650mg PO q6h PRN pain

SOCIAL HISTORY:
Former smoker, quit 5 years ago (30 pack-year history). Denies alcohol or drug use.
Retired mechanic. Lives alone in apartment. No family support nearby.
Has difficulty affording medications. Uses public transportation.

VITAL SIGNS:
BP 158/92 mmHg, HR 88 bpm, Temp 98.4°F, RR 20/min, SpO2 94% on RA
Weight: 92 kg, Height: 175 cm

ASSESSMENT AND PLAN:
1. Acute coronary syndrome — troponin trending up, cardiology consulted for cath
2. COPD exacerbation — continue bronchodilators, add prednisone 40mg x 5 days
3. T2DM — hold metformin pending contrast study, insulin sliding scale
4. HTN — uptitrate metoprolol to 100mg daily
5. Hyperlipidemia — continue high-intensity statin
6. Pain management — continue acetaminophen PRN, avoid NSAIDs`;

// ─── Module Toggle ───────────────────────────────────────────

function ModuleToggle({
  module: mod,
  enabled,
  onToggle,
}: {
  module: ModuleDef;
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`flex items-center gap-3 rounded-lg border-2 p-3 text-left transition-all ${
        enabled
          ? 'border-indigo-500 bg-indigo-50'
          : 'border-gray-200 bg-white opacity-60 hover:opacity-80'
      }`}
    >
      <span className="text-xl">{mod.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-900 truncate">{mod.label}</span>
          <div
            className={`w-8 h-4 rounded-full transition-colors ${
              enabled ? 'bg-indigo-500' : 'bg-gray-300'
            } flex items-center ${enabled ? 'justify-end' : 'justify-start'} px-0.5`}
          >
            <div className="w-3 h-3 rounded-full bg-white shadow-sm" />
          </div>
        </div>
        <p className="text-xs text-gray-500 truncate">{mod.description}</p>
      </div>
    </button>
  );
}

// ─── Result Section Component ────────────────────────────────

function ResultSection({
  title,
  icon,
  children,
  errorMsg,
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
  errorMsg?: string;
}) {
  const [expanded, setExpanded] = useState(true);

  if (errorMsg) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-red-700">
          <span>{icon}</span> {title}
          <span className="text-xs font-normal text-red-500">— {errorMsg}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-white overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-700">
          <span>{icon}</span> {title}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && <div className="p-4">{children}</div>}
    </div>
  );
}

// ─── JSON Viewer ─────────────────────────────────────────────

function JsonPreview({ data }: { data: unknown }) {
  return (
    <pre className="text-xs font-mono bg-gray-50 rounded-lg p-3 overflow-x-auto max-h-64 overflow-y-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

// ─── Main Page ───────────────────────────────────────────────

export function PipelineExplorer() {
  const [text, setText] = useState('');
  const [config, setConfig] = useState<EnhancedAnalysisConfig>(() => {
    const initial: EnhancedAnalysisConfig = {};
    for (const mod of MODULES) {
      (initial as Record<string, boolean>)[mod.key] = mod.defaultEnabled;
    }
    return initial;
  });
  const [results, setResults] = useState<EnhancedAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleModule = (key: keyof EnhancedAnalysisConfig) => {
    setConfig((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const enabledCount = Object.values(config).filter(Boolean).length;

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/analyze/enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, config }),
      });
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data: EnhancedAnalysisResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [text, config]);

  const preProcessing = MODULES.filter((m) => m.phase === 'pre-processing');
  const extraction = MODULES.filter((m) => m.phase === 'extraction');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">🔬 Pipeline Explorer</h1>
        <p className="mt-1 text-sm text-gray-500">
          Configure and run ClinIQ's enhanced 14-module clinical NLP pipeline.
          Toggle modules on/off to customize analysis for your use case.
        </p>
      </div>

      {/* Module Toggles */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-700">
            Phase 1 — Pre-processing
          </h2>
          <span className="text-xs text-gray-500">{enabledCount}/14 modules enabled</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {preProcessing.map((mod) => (
            <ModuleToggle
              key={mod.key}
              module={mod}
              enabled={!!config[mod.key]}
              onToggle={() => toggleModule(mod.key)}
            />
          ))}
        </div>

        <h2 className="text-sm font-semibold text-gray-700 pt-2">
          Phase 2 — Extraction & Scoring
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {extraction.map((mod) => (
            <ModuleToggle
              key={mod.key}
              module={mod}
              enabled={!!config[mod.key]}
              onToggle={() => toggleModule(mod.key)}
            />
          ))}
        </div>

        {/* Quick Actions */}
        <div className="flex gap-2">
          <button
            onClick={() => {
              const all: EnhancedAnalysisConfig = {};
              for (const mod of MODULES) (all as Record<string, boolean>)[mod.key] = true;
              setConfig(all);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            Enable all
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={() => {
              const none: EnhancedAnalysisConfig = {};
              for (const mod of MODULES) (none as Record<string, boolean>)[mod.key] = false;
              setConfig(none);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            Disable all
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={() => {
              const defaults: EnhancedAnalysisConfig = {};
              for (const mod of MODULES)
                (defaults as Record<string, boolean>)[mod.key] = mod.defaultEnabled;
              setConfig(defaults);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            Reset defaults
          </button>
        </div>
      </div>

      {/* Input */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">Clinical Text</label>
          <button
            onClick={() => {
              setText(SAMPLE_NOTE);
              setResults(null);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            Load sample discharge summary
          </button>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a clinical document to analyze with the enhanced pipeline..."
          rows={12}
          className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">
            {text.length > 0
              ? `${text.length.toLocaleString()} chars · ${text.split(/\s+/).filter(Boolean).length} words`
              : ''}
          </span>
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || loading || enabledCount === 0}
            className="px-6 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing ({enabledCount} modules)…
              </span>
            ) : (
              `🔬 Analyze (${enabledCount} modules)`
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
          {/* Summary Bar */}
          <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-4 flex items-center justify-between">
            <div className="text-sm text-indigo-700">
              <strong>Analysis complete</strong> in{' '}
              {results.processing_time_ms.toFixed(1)}ms
              {Object.keys(results.component_errors).length > 0 && (
                <span className="text-red-600 ml-2">
                  ({Object.keys(results.component_errors).length} module errors)
                </span>
              )}
            </div>
          </div>

          {/* Module Results */}
          {results.classification && (
            <ResultSection title="Document Classification" icon="📋">
              <div className="flex items-center gap-3">
                <span className="text-lg font-semibold text-gray-900 capitalize">
                  {results.classification.predicted_type.replace(/_/g, ' ')}
                </span>
                <span className="text-sm text-gray-500">
                  ({Math.round(results.classification.confidence * 100)}% confidence)
                </span>
              </div>
              {results.classification.scores.length > 1 && (
                <div className="mt-3 space-y-1">
                  {results.classification.scores.slice(0, 5).map((s, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-32 truncate text-gray-600 capitalize">
                        {s.document_type.replace(/_/g, ' ')}
                      </span>
                      <div className="flex-1 h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-full bg-indigo-400 rounded-full"
                          style={{ width: `${Math.round(s.score * 100)}%` }}
                        />
                      </div>
                      <span className="text-gray-400 tabular-nums">
                        {Math.round(s.score * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </ResultSection>
          )}

          {results.quality && (
            <ResultSection title="Quality Analysis" icon="✅">
              <div className="flex items-center gap-4 mb-3">
                <div
                  className={`text-3xl font-bold ${
                    results.quality.overall_score >= 80
                      ? 'text-green-600'
                      : results.quality.overall_score >= 60
                        ? 'text-yellow-600'
                        : 'text-red-600'
                  }`}
                >
                  {results.quality.grade}
                </div>
                <div>
                  <p className="text-sm text-gray-700">
                    Overall Score: {results.quality.overall_score.toFixed(0)}/100
                  </p>
                  <p className="text-xs text-gray-500">
                    {results.quality.dimensions.length} dimensions evaluated
                  </p>
                </div>
              </div>
              {results.quality.recommendations.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-gray-600 mb-1">Recommendations:</p>
                  <ul className="list-disc list-inside text-xs text-gray-600 space-y-0.5">
                    {results.quality.recommendations.slice(0, 5).map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </ResultSection>
          )}

          {results.sections && (
            <ResultSection title={`Sections (${results.sections.count})`} icon="📑">
              <div className="space-y-2">
                {results.sections.sections.map((s, i) => (
                  <div key={i} className="text-sm border-l-4 border-indigo-300 pl-3">
                    <span className="font-medium text-gray-800 capitalize">
                      {s.category.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs text-gray-400 ml-2">
                      ({Math.round(s.confidence * 100)}%)
                    </span>
                    <p className="text-xs text-gray-500 line-clamp-2 mt-0.5">{s.body_text.slice(0, 150)}…</p>
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.medications && (
            <ResultSection title={`Medications (${results.medications.count})`} icon="💊">
              <div className="space-y-1">
                {results.medications.medications.map((m, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="font-medium text-gray-800">{m.drug_name}</span>
                    {m.dosage && <span className="text-gray-600">{m.dosage}</span>}
                    {m.route && <span className="text-gray-500">{m.route}</span>}
                    {m.frequency && <span className="text-gray-500">{m.frequency}</span>}
                    {m.prn && (
                      <span className="px-1.5 py-0.5 rounded text-xs bg-blue-50 text-blue-700">
                        PRN
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.allergies && (
            <ResultSection title={`Allergies (${results.allergies.count})`} icon="⚠️">
              {results.allergies.nkda_detected && (
                <p className="text-xs text-green-600 font-medium mb-2">
                  ✓ NKDA (No Known Drug Allergies) detected
                </p>
              )}
              <div className="space-y-1">
                {results.allergies.allergies.map((a, i) => (
                  <div key={i} className="text-sm">
                    <span className="font-medium text-gray-800">{a.canonical_name}</span>
                    <span className="text-xs text-gray-500 ml-1">({a.category})</span>
                    {a.reactions.length > 0 && (
                      <span className="text-xs text-red-600 ml-2">
                        → {a.reactions.join(', ')}
                      </span>
                    )}
                    {a.severity !== 'unknown' && (
                      <span
                        className={`ml-2 px-1.5 py-0.5 rounded text-xs ${
                          a.severity === 'life_threatening'
                            ? 'bg-red-100 text-red-800'
                            : a.severity === 'severe'
                              ? 'bg-orange-100 text-orange-800'
                              : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {a.severity}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.vitals && (
            <ResultSection title={`Vital Signs (${results.vitals.count})`} icon="💓">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {results.vitals.vitals.map((v, i) => (
                  <div key={i} className="rounded-lg bg-gray-50 p-3">
                    <p className="text-xs text-gray-500 capitalize">
                      {v.type.replace(/_/g, ' ')}
                    </p>
                    <p className="text-lg font-bold text-gray-900">
                      {v.value} <span className="text-sm font-normal text-gray-500">{v.unit}</span>
                    </p>
                    <span
                      className={`text-xs ${
                        v.interpretation === 'normal'
                          ? 'text-green-600'
                          : v.interpretation.includes('critical')
                            ? 'text-red-600'
                            : 'text-yellow-600'
                      }`}
                    >
                      {v.interpretation}
                    </span>
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.sdoh && (
            <ResultSection title={`SDoH Findings (${results.sdoh.findings.length})`} icon="🏠">
              <div className="flex gap-4 mb-3 text-xs">
                <span className="text-red-600">
                  {results.sdoh.adverse_count} adverse
                </span>
                <span className="text-green-600">
                  {results.sdoh.protective_count} protective
                </span>
                <span className="text-gray-500">
                  {results.sdoh.domains_detected.length} domains
                </span>
              </div>
              <div className="space-y-1">
                {results.sdoh.findings.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        f.sentiment === 'adverse'
                          ? 'bg-red-500'
                          : f.sentiment === 'protective'
                            ? 'bg-green-500'
                            : 'bg-gray-400'
                      }`}
                    />
                    <span className="text-gray-600 capitalize">{f.domain.replace(/_/g, ' ')}:</span>
                    <span className="text-gray-800">{f.trigger_text}</span>
                    {f.z_code && (
                      <span className="text-xs text-gray-400">[{f.z_code}]</span>
                    )}
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.comorbidity && (
            <ResultSection title="Comorbidity Index" icon="📊">
              <div className="flex items-center gap-6">
                <div>
                  <p className="text-xs text-gray-500">CCI Score</p>
                  <p className="text-3xl font-bold text-gray-900">{results.comorbidity.score}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Risk Group</p>
                  <p className="text-lg font-semibold capitalize text-gray-700">
                    {results.comorbidity.risk_group}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">10-Year Mortality</p>
                  <p className="text-lg font-semibold text-gray-700">
                    {(((results.comorbidity.ten_year_mortality ?? results.comorbidity.estimated_mortality) ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-1">
                {results.comorbidity.categories
                  .filter((c) => c.detected)
                  .map((c, i) => (
                    <span
                      key={i}
                      className="text-xs bg-orange-50 text-orange-700 px-2 py-1 rounded"
                    >
                      {c.name} (×{c.weight})
                    </span>
                  ))}
              </div>
            </ResultSection>
          )}

          {results.temporal && results.temporal.count > 0 && (
            <ResultSection title={`Temporal (${results.temporal.count})`} icon="⏰">
              <JsonPreview data={results.temporal.expressions} />
            </ResultSection>
          )}

          {results.relations && results.relations.count > 0 && (
            <ResultSection title={`Relations (${results.relations.count})`} icon="🔗">
              <div className="space-y-1">
                {results.relations.relations.map((r, i) => (
                  <div key={i} className="text-sm">
                    <span className="font-medium text-gray-800">{r.subject}</span>
                    <span className="text-indigo-600 mx-2">—{r.relation_type}→</span>
                    <span className="font-medium text-gray-800">{r.object}</span>
                    <span className="text-xs text-gray-400 ml-2">
                      ({Math.round(r.confidence * 100)}%)
                    </span>
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {results.abbreviations && results.abbreviations.count > 0 && (
            <ResultSection
              title={`Abbreviations (${results.abbreviations.count})`}
              icon="🔤"
            >
              <div className="space-y-1">
                {results.abbreviations.abbreviations.map((a, i) => (
                  <div key={i} className="text-sm">
                    <span className="font-mono font-bold text-gray-900">{a.abbreviation}</span>
                    <span className="text-gray-400 mx-1">→</span>
                    <span className="text-gray-700">{a.expansion}</span>
                    <span className="text-xs text-gray-400 ml-2">[{a.domain}]</span>
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {/* Component Errors */}
          {Object.keys(results.component_errors).length > 0 && (
            <ResultSection title="Module Errors" icon="⚠️">
              <div className="space-y-1">
                {Object.entries(results.component_errors).map(([key, msg]) => (
                  <div key={key} className="text-sm text-red-600">
                    <strong>{key}:</strong> {msg}
                  </div>
                ))}
              </div>
            </ResultSection>
          )}

          {/* Raw JSON */}
          <ResultSection title="Raw JSON Response" icon="{ }">
            <JsonPreview data={results} />
          </ResultSection>
        </div>
      )}
    </div>
  );
}

