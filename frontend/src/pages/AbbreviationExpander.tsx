/**
 * AbbreviationExpander page — Detect and expand clinical abbreviations
 * in free text with context-aware disambiguation.
 *
 * Uses the rule-based abbreviation expansion module (120+ medical
 * abbreviation entries across 12 clinical domains) to detect abbreviations,
 * expand them in-place, and flag ambiguous terms with alternative readings.
 * Includes domain filtering, confidence threshold, and dictionary stats.
 */

import { useState, useCallback } from 'react';
import type { AbbreviationExpansionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'ED Assessment',
    text: `CC: SOB, CP x 2 days
HPI: 65 yo M with PMH of HTN, DM, CAD s/p CABG presents with worsening
SOB and CP. Pt reports DOE and orthopnea. Denies PND. + bilateral LE edema.
VS: BP 180/100, HR 110, RR 24, SpO2 88% on RA.
PE: JVD present, bilateral crackles, S3 gallop. 2+ pitting edema BLE.
Labs: BNP 1200, Trop-I 0.02, Cr 1.8, BUN 35.
Impression: CHF exacerbation, r/o ACS. Plan: IV furosemide, O2 via NC,
repeat EKG, serial troponins, CXR, Cardiology c/s.`,
  },
  {
    label: 'Dental Progress Note',
    text: `Pt presents for SRP on UR and UL quadrants. PMH: DM2, HTN on ACEI.
BOP noted in #3, #14. CAL 5mm on #3 mesial. PD 6mm #14 distal.
Radiographs show horizontal bone loss. Tx plan: SRP today, re-eval 4-6 wks.
Post-op: RTC for LL and LR quadrants. OHI given — emphasized
interdental cleaning. Rx: CHX rinse 0.12% BID x 2 wks.
Assessment: Generalized chronic periodontitis, mod-severe. ADA code D4341.`,
  },
  {
    label: 'Discharge Summary',
    text: `DIAGNOSIS: NSTEMI, CHF, CKD Stage III, COPD exacerbation
HOSPITAL COURSE: Pt admitted via ED with ACS protocol. EKG showed ST
depression V4-V6. Heparin gtt and NTG initiated. Cardiology rec'd
cath — 90% LAD stenosis, DES placed. Post-PCI on DAPT (ASA + Plavix).
Pt developed ARF on CIN — Cr peaked 3.2, now improving.
COPD managed with nebs (albuterol/ipratropium), IV steroids → PO prednisone taper.
DISCHARGE MEDS: ASA 81mg, Clopidogrel 75mg, Atorvastatin 80mg, Metoprolol
XL 50mg, Lisinopril 10mg, Lasix 40mg PO BID, Albuterol MDI PRN.
F/U: PCP 1 wk, Cards 2 wks, Nephro 1 wk.`,
  },
];

// ─── Domain Colors ───────────────────────────────────────────

const DOMAIN_COLORS: Record<string, string> = {
  cardiology: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  pulmonology: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300',
  endocrine: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
  neurology: 'bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300',
  gastroenterology: 'bg-lime-100 text-lime-800 dark:bg-lime-900/30 dark:text-lime-300',
  renal: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
  infectious: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
  musculoskeletal: 'bg-stone-100 text-stone-800 dark:bg-stone-900/30 dark:text-stone-300',
  hematology: 'bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300',
  general: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  dental: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
  pharmacy: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
};

// ─── Helper Components ───────────────────────────────────────

function DomainBadge({ domain }: { domain: string }) {
  const color = DOMAIN_COLORS[domain.toLowerCase()] ?? DOMAIN_COLORS.general;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {domain}
    </span>
  );
}

function ConfidenceDot({ confidence }: { confidence: number }) {
  const color =
    confidence >= 0.9
      ? 'bg-green-500'
      : confidence >= 0.7
        ? 'bg-blue-500'
        : confidence >= 0.5
          ? 'bg-amber-500'
          : 'bg-red-500';
  return (
    <span className="inline-flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {(confidence * 100).toFixed(0)}%
    </span>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function AbbreviationExpander() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<AbbreviationExpansionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.6);
  const [showExpanded, setShowExpanded] = useState(true);

  const analyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/abbreviations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          min_confidence: minConfidence,
          expand_in_place: true,
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults({
        abbreviations: data.matches ?? [],
        expanded_text: data.expanded_text ?? null,
        count: data.total_found ?? 0,
        processing_time_ms: data.processing_time_ms ?? 0,
      });
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

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          🔤 Abbreviation Expander
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Detect and expand clinical abbreviations with context-aware disambiguation
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
        {/* Sample Note Buttons */}
        <div className="flex flex-wrap gap-2">
          {SAMPLE_NOTES.map((note, idx) => (
            <button
              key={note.label}
              onClick={() => loadSample(idx)}
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition-colors"
            >
              📋 {note.label}
            </button>
          ))}
        </div>

        {/* Text Input */}
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            placeholder="Paste clinical text containing abbreviations..."
            className="w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-sm focus:border-blue-500 focus:ring-blue-500"
            disabled={loading}
          />
          <div className="mt-1 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>{wordCount} words</span>
            <span>Max 50,000 characters</span>
          </div>
        </div>

        {/* Options */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Min Confidence:
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="w-24 h-2 bg-gray-200 rounded-lg appearance-none dark:bg-gray-700"
            />
            <span className="text-sm text-gray-600 dark:text-gray-400 w-10">
              {(minConfidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Analyze Button */}
        <button
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analyzing...
            </span>
          ) : (
            '🔍 Expand Abbreviations'
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
      {results && (
        <div className="space-y-4">
          {/* Stats Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{results.count}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Abbreviations Found</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">
                {results.abbreviations.filter((a) => a.is_ambiguous).length}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Ambiguous</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {new Set(results.abbreviations.map((a) => a.domain)).size}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Domains</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Processing Time</p>
            </div>
          </div>

          {/* Expanded Text Toggle */}
          {results.expanded_text && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  📝 Expanded Text
                </h2>
                <button
                  onClick={() => setShowExpanded(!showExpanded)}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {showExpanded ? 'Hide' : 'Show'}
                </button>
              </div>
              {showExpanded && (
                <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-4">
                  <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap font-mono leading-relaxed">
                    {results.expanded_text}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Abbreviation Table */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                🔤 Detected Abbreviations
              </h2>
            </div>
            {results.abbreviations.length === 0 ? (
              <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                No abbreviations detected above the confidence threshold.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-900/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Abbreviation</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Expansion</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Domain</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Confidence</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {results.abbreviations.map((abbr, idx) => (
                      <tr key={`${abbr.abbreviation}-${idx}`} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-3">
                          <code className="text-sm font-bold text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-1.5 py-0.5 rounded">
                            {abbr.abbreviation}
                          </code>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                          {abbr.expansion}
                        </td>
                        <td className="px-4 py-3">
                          <DomainBadge domain={abbr.domain} />
                        </td>
                        <td className="px-4 py-3">
                          <ConfidenceDot confidence={abbr.confidence} />
                        </td>
                        <td className="px-4 py-3">
                          {abbr.is_ambiguous ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">
                              ⚠️ Ambiguous
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                              ✅ Clear
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Domain Distribution */}
          {results.abbreviations.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                📊 Domain Distribution
              </h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(
                  results.abbreviations.reduce<Record<string, number>>((acc, a) => {
                    acc[a.domain] = (acc[a.domain] ?? 0) + 1;
                    return acc;
                  }, {})
                )
                  .sort(([, a], [, b]) => b - a)
                  .map(([domain, count]) => (
                    <div key={domain} className="flex items-center gap-2">
                      <DomainBadge domain={domain} />
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        ×{count}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
