/**
 * SDoHExtractor page — Extract Social Determinants of Health from clinical text.
 *
 * Identifies social and behavioural risk factors across 8 Healthy People 2030
 * domains: Housing, Employment, Education, Food Security, Transportation,
 * Social Support, Substance Use, and Financial.  Results include sentiment
 * classification (adverse/protective/neutral), ICD-10-CM Z-code mapping,
 * and negation-aware confidence scoring.
 */

import { useState, useCallback } from 'react';
import type { SDoHFinding, SDoHExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Social History',
    text: `SOCIAL HISTORY:
Patient is a 54-year-old male, recently homeless after eviction 3 months ago. Currently staying at a shelter. Previously employed as a construction worker — lost job due to back injury. Reports food insecurity, relying on food banks and soup kitchens. No reliable transportation to medical appointments; missed last 3 follow-ups.

Substance use: Current smoker, 1 pack per day x 30 years (30 pack-year history). Drinks 4-6 beers daily. Denies illicit drug use. Previously used methamphetamine, clean for 5 years.

Social support: Divorced, estranged from family. No emergency contact listed. Reports feeling isolated. Denies domestic violence.

Education: Did not complete high school. Limited health literacy noted — difficulty understanding medication instructions.

Financial: Uninsured. Unable to afford prescribed medications. Reports rationing insulin.`,
  },
  {
    label: 'Protective Factors',
    text: `SOCIAL HISTORY:
Patient lives with spouse in stable housing. Employed full-time as a teacher with good insurance benefits. Has strong family support system — wife and two adult children involved in care. Active in church community. College-educated, health literate, asks appropriate questions about treatment plan.

Substance use: Never smoked. Social drinker — 1-2 glasses of wine per week. No illicit drug use.
No food insecurity. Has reliable transportation. Financially stable with savings.

Exercises regularly — walks 30 minutes daily, yoga twice weekly.`,
  },
  {
    label: 'Mixed Factors',
    text: `SOCIAL HISTORY:
35-year-old female, single mother of 3 children (ages 4, 7, 11). Lives in subsidized housing — reports mold and cockroach issues. Employed part-time at retail store, no health benefits.

Enrolled in Medicaid and SNAP benefits. Uses food pantry weekly. Has car but reports difficulty affording gas for medical appointments.

Former smoker — quit 2 years ago. Occasional marijuana use for anxiety. Denies alcohol use. History of domestic violence — left abusive partner 1 year ago, currently in counseling. GED completed.

Support: Has one close friend who helps with childcare. Attends domestic violence support group. No family nearby.

Financial stress: Medical debt from ER visits. Skipping dental care due to cost.`,
  },
];

// ─── Domain Styling ──────────────────────────────────────────

const DOMAIN_CONFIG: Record<string, { icon: string; color: string; bg: string }> = {
  housing: { icon: '🏠', color: 'text-blue-700 dark:text-blue-300', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  employment: { icon: '💼', color: 'text-indigo-700 dark:text-indigo-300', bg: 'bg-indigo-100 dark:bg-indigo-900/30' },
  education: { icon: '📚', color: 'text-purple-700 dark:text-purple-300', bg: 'bg-purple-100 dark:bg-purple-900/30' },
  food_security: { icon: '🍎', color: 'text-orange-700 dark:text-orange-300', bg: 'bg-orange-100 dark:bg-orange-900/30' },
  transportation: { icon: '🚗', color: 'text-cyan-700 dark:text-cyan-300', bg: 'bg-cyan-100 dark:bg-cyan-900/30' },
  social_support: { icon: '🤝', color: 'text-pink-700 dark:text-pink-300', bg: 'bg-pink-100 dark:bg-pink-900/30' },
  substance_use: { icon: '🚬', color: 'text-red-700 dark:text-red-300', bg: 'bg-red-100 dark:bg-red-900/30' },
  financial: { icon: '💰', color: 'text-green-700 dark:text-green-300', bg: 'bg-green-100 dark:bg-green-900/30' },
};

const SENTIMENT_STYLES: Record<string, { label: string; color: string; icon: string }> = {
  adverse: { label: 'Risk Factor', color: 'text-red-600 dark:text-red-400', icon: '⚠️' },
  protective: { label: 'Protective', color: 'text-green-600 dark:text-green-400', icon: '🛡️' },
  neutral: { label: 'Neutral', color: 'text-gray-600 dark:text-gray-400', icon: '➖' },
};

// ─── Main Component ──────────────────────────────────────────

export function SDoHExtractor() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<SDoHExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [domainFilter, setDomainFilter] = useState<string>('all');

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/sdoh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data: SDoHExtractionResponse = await response.json();
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

  const findings: SDoHFinding[] = results?.findings ?? [];
  const filtered =
    domainFilter === 'all' ? findings : findings.filter((f) => f.domain === domainFilter);

  // Aggregate counts
  const sentimentCounts = findings.reduce(
    (acc: Record<string, number>, f: SDoHFinding) => {
      acc[f.sentiment] = (acc[f.sentiment] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const domainCounts = findings.reduce(
    (acc: Record<string, number>, f: SDoHFinding) => {
      acc[f.domain] = (acc[f.domain] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const activeDomains = Object.keys(domainCounts);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">
          Social Determinants of Health
        </h1>
        <p className="mt-1 text-text-secondary">
          Identify social risk and protective factors across 8 Healthy People 2030 domains with ICD-10-CM Z-code mapping.
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
          placeholder="Paste clinical note with social history..."
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
            {loading ? 'Analyzing...' : 'Extract SDoH'}
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
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-text-primary">{findings.length}</div>
              <div className="text-xs text-text-muted mt-1">Total Findings</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-red-600">{sentimentCounts.adverse ?? 0}</div>
              <div className="text-xs text-text-muted mt-1">⚠️ Risk Factors</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-green-600">{sentimentCounts.protective ?? 0}</div>
              <div className="text-xs text-text-muted mt-1">🛡️ Protective</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-2xl font-bold text-blue-600">{activeDomains.length}</div>
              <div className="text-xs text-text-muted mt-1">Domains Active</div>
            </div>
          </div>

          {/* Domain Filter Bar */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-text-muted">Domain:</span>
            <button
              onClick={() => setDomainFilter('all')}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                domainFilter === 'all'
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              All ({findings.length})
            </button>
            {Object.entries(DOMAIN_CONFIG).map(([domain, cfg]) => {
              const count = domainCounts[domain] ?? 0;
              if (count === 0) return null;
              return (
                <button
                  key={domain}
                  onClick={() => setDomainFilter(domain)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                    domainFilter === domain
                      ? 'bg-primary-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  {cfg.icon} {domain.replace('_', ' ')} ({count})
                </button>
              );
            })}
          </div>

          {/* Findings List */}
          <div className="space-y-3">
            {filtered.length === 0 ? (
              <div className="card p-8 text-center text-text-muted">
                No findings match the current filter.
              </div>
            ) : (
              filtered.map((finding: SDoHFinding, idx: number) => {
                const domainCfg = DOMAIN_CONFIG[finding.domain] ?? DOMAIN_CONFIG.housing;
                const sentimentCfg = SENTIMENT_STYLES[finding.sentiment] ?? SENTIMENT_STYLES.neutral;

                return (
                  <div key={idx} className="card p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${domainCfg.bg} ${domainCfg.color}`}>
                            {domainCfg.icon} {finding.domain.replace('_', ' ')}
                          </span>
                          <span className={`text-xs font-medium ${sentimentCfg.color}`}>
                            {sentimentCfg.icon} {sentimentCfg.label}
                          </span>
                          {finding.z_code && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-text-muted font-mono">
                              {finding.z_code}
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-text-primary">{finding.trigger_text}</p>
                        {finding.matched_text && finding.matched_text !== finding.trigger_text && (
                          <p className="text-xs text-text-muted italic">
                            Context: &ldquo;...{finding.matched_text}...&rdquo;
                          </p>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-text-muted">
                          {Math.round(finding.confidence * 100)}%
                        </div>
                        <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mt-1">
                          <div
                            className="h-full rounded-full bg-primary-500"
                            style={{ width: `${Math.round(finding.confidence * 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
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
