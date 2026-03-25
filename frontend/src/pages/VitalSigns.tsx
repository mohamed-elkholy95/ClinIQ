/**
 * VitalSigns page — Extract and interpret vital signs from clinical text.
 *
 * Uses the rule-based vital signs extraction module (9 vital types, 20+
 * regex patterns, qualitative descriptor detection) to identify BP, HR,
 * temperature, SpO2, respiratory rate, weight, height, BMI, and pain.
 * Results include physiological range validation and clinical interpretation
 * (normal/low/high/critical_low/critical_high) per AHA 2017 guidelines.
 */

import { useState, useCallback } from 'react';
import type { VitalSignResult, VitalExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Emergency Triage',
    text: `VITAL SIGNS (0830):
BP: 158/94 mmHg (hypertensive)
HR: 112 bpm (tachycardic)
Temp: 101.8°F (febrile)
RR: 22 breaths/min
SpO2: 93% on room air
Weight: 87 kg
Height: 5'10"
Pain: 7/10

Patient is afebrile on repeat check. Repeat vitals at 0945:
BP: 142/88, HR: 98, RR: 18, SpO2: 96% on 2L NC.`,
  },
  {
    label: 'Routine Physical',
    text: `PHYSICAL EXAMINATION:
Vitals: Blood pressure 118/76 mmHg, pulse 72 bpm, temperature 98.4°F,
respiratory rate 16/min, oxygen saturation 99% on RA.
Weight: 165 lbs, Height: 5'8", BMI calculated at 25.1 kg/m².
Patient reports pain 0/10 at rest.
General: Well-nourished, well-developed, no acute distress.
Cardiovascular: Regular rate and rhythm, no murmurs.`,
  },
  {
    label: 'ICU Progress Note',
    text: `ICU DAY 3 — 0600 ASSESSMENT:
Hemodynamics: BP 92/58 (hypotensive on vasopressors), MAP 69.
Heart rate 54 bpm (bradycardic), sinus rhythm.
Temperature 35.8°C (hypothermic). Warming blanket applied.
Respiratory: Rate 28/min (tachypneic), SpO2 88% on 6L high-flow.
FiO2 increased to 60%. ABG pending.
Weight: 92.3 kg (up 3kg from admission — fluid overload).
Pain score 4/10 on fentanyl drip.
Urine output 15 mL/hr — concerning for AKI.`,
  },
];

// ─── Interpretation Styling ──────────────────────────────────

const INTERPRETATION_STYLES: Record<string, { bg: string; text: string; icon: string }> = {
  normal: {
    bg: 'bg-green-100 dark:bg-green-900/30',
    text: 'text-green-800 dark:text-green-300',
    icon: '✅',
  },
  low: {
    bg: 'bg-blue-100 dark:bg-blue-900/30',
    text: 'text-blue-800 dark:text-blue-300',
    icon: '🔽',
  },
  high: {
    bg: 'bg-amber-100 dark:bg-amber-900/30',
    text: 'text-amber-800 dark:text-amber-300',
    icon: '🔼',
  },
  critical_low: {
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-800 dark:text-red-300',
    icon: '🔻',
  },
  critical_high: {
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-800 dark:text-red-300',
    icon: '🔺',
  },
};

const VITAL_ICONS: Record<string, string> = {
  blood_pressure: '🫀',
  heart_rate: '💓',
  temperature: '🌡️',
  respiratory_rate: '🌬️',
  oxygen_saturation: '🫁',
  weight: '⚖️',
  height: '📏',
  bmi: '📊',
  pain: '😣',
};

// ─── Vital Card Component ────────────────────────────────────

function VitalCard({ vital }: { vital: VitalSignResult }) {
  const interp = INTERPRETATION_STYLES[vital.interpretation ?? 'normal'] ?? INTERPRETATION_STYLES.normal;
  const icon = VITAL_ICONS[vital.type] ?? '📋';
  const label = vital.type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className={`card p-4 border-l-4 ${interp.bg} space-y-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <h3 className="text-sm font-semibold text-text-primary">{label}</h3>
        </div>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${interp.bg} ${interp.text}`}>
          {interp.icon} {(vital.interpretation ?? 'normal').replace('_', ' ')}
        </span>
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-text-primary">{vital.value}</span>
        <span className="text-sm text-text-muted">{vital.unit}</span>
      </div>

      {/* Diastolic for BP */}
      {vital.type === 'blood_pressure' && vital.diastolic != null && (
        <div className="text-xs text-text-secondary">
          Systolic/Diastolic: {vital.value}/{vital.diastolic} mmHg
          {vital.map != null && <span className="ml-2">• MAP: {vital.map}</span>}
        </div>
      )}

      {/* Trend */}
      {vital.trend && (
        <div className="text-xs text-text-muted">
          Trend: {vital.trend === 'improving' ? '📈' : vital.trend === 'worsening' ? '📉' : '➡️'}{' '}
          {vital.trend}
        </div>
      )}

      {/* Confidence */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full bg-primary-500"
            style={{ width: `${Math.round(vital.confidence * 100)}%` }}
          />
        </div>
        <span className="text-[10px] text-text-muted">{Math.round(vital.confidence * 100)}%</span>
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function VitalSigns() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<VitalExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/vitals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data: VitalExtractionResponse = await response.json();
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

  // Count interpretations for summary
  const interpretationCounts = (results?.vitals ?? []).reduce(
    (acc: Record<string, number>, v: VitalSignResult) => {
      const interp = v.interpretation ?? 'normal';
      acc[interp] = (acc[interp] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const criticalCount = (interpretationCounts.critical_low ?? 0) + (interpretationCounts.critical_high ?? 0);
  const abnormalCount = (interpretationCounts.low ?? 0) + (interpretationCounts.high ?? 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Vital Signs Extraction</h1>
        <p className="mt-1 text-text-secondary">
          Extract and interpret vital signs with clinical range validation and trend detection.
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
          placeholder="Paste clinical note with vital signs..."
          rows={7}
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
            {loading ? 'Extracting...' : 'Extract Vitals'}
          </button>
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
          {/* Summary Banner */}
          <div className="card p-4 flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-2xl font-bold text-text-primary">
                  {results.vitals?.length ?? 0}
                </span>
                <span className="text-xs text-text-muted ml-2">vitals detected</span>
              </div>
              {criticalCount > 0 && (
                <div className="px-3 py-1 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-sm font-medium">
                  🚨 {criticalCount} critical value{criticalCount > 1 ? 's' : ''}
                </div>
              )}
              {abnormalCount > 0 && (
                <div className="px-3 py-1 rounded-lg bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-sm font-medium">
                  ⚠️ {abnormalCount} abnormal
                </div>
              )}
              {criticalCount === 0 && abnormalCount === 0 && (results.vitals?.length ?? 0) > 0 && (
                <div className="px-3 py-1 rounded-lg bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-sm font-medium">
                  ✅ All within normal range
                </div>
              )}
            </div>
            {results.processing_time_ms != null && (
              <span className="text-xs text-text-muted">
                {results.processing_time_ms.toFixed(1)}ms
              </span>
            )}
          </div>

          {/* Vital Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {(results.vitals ?? []).map((vital: VitalSignResult, idx: number) => (
              <VitalCard key={idx} vital={vital} />
            ))}
          </div>

          {/* Interpretation Legend */}
          <div className="card p-4">
            <h3 className="text-xs font-semibold text-text-muted mb-3">INTERPRETATION LEGEND</h3>
            <div className="flex flex-wrap gap-3">
              {Object.entries(INTERPRETATION_STYLES).map(([key, style]) => (
                <div key={key} className="flex items-center gap-1.5">
                  <span className={`inline-block w-3 h-3 rounded-full ${style.bg}`} />
                  <span className="text-xs text-text-secondary">
                    {key.replace('_', ' ')}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
