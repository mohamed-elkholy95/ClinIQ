/**
 * StreamingAnalysis page — Real-time SSE pipeline analysis viewer.
 *
 * Submits clinical text to the POST /analyze/stream endpoint and displays
 * pipeline stage results as they arrive via Server-Sent Events.  Each stage
 * (NER, ICD, summary, risk) appears in its own card with a completion
 * indicator, giving users immediate feedback on long-running analyses.
 *
 * Design decisions:
 * - Uses the AbortController-based analyzeStream client function for
 *   clean cancellation support.
 * - Stage results are accumulated in state and rendered incrementally,
 *   so partial results are visible even if the stream is interrupted.
 * - A cancel button allows users to abort analysis mid-stream without
 *   waiting for all stages to complete.
 * - Sample clinical notes are provided for quick testing.
 */

import { useState, useRef, useCallback } from 'react';
import { Radio, Play, Square, CheckCircle2, Loader2, Clock, FileText, Sparkles } from 'lucide-react';
import { analyzeStream } from '../services/clinical';

/** Stage display configuration */
const STAGE_CONFIG: Record<string, { label: string; emoji: string; color: string }> = {
  ner: { label: 'Named Entity Recognition', emoji: '🏷️', color: 'border-blue-200 dark:border-blue-800' },
  icd: { label: 'ICD-10 Code Prediction', emoji: '📋', color: 'border-green-200 dark:border-green-800' },
  summary: { label: 'Clinical Summarization', emoji: '📝', color: 'border-purple-200 dark:border-purple-800' },
  risk: { label: 'Risk Scoring', emoji: '⚠️', color: 'border-orange-200 dark:border-orange-800' },
};

interface StageResult {
  stage: string;
  data: unknown;
  receivedAt: Date;
}

/** Sample clinical notes for testing */
const SAMPLE_NOTES = [
  {
    label: 'Emergency Visit',
    text: `Chief Complaint: Chest pain and shortness of breath for 2 hours.

History of Present Illness: 67-year-old male with PMH of hypertension, type 2 diabetes, and hyperlipidemia presents with acute onset substernal chest pain radiating to left arm. Pain is 8/10, crushing in nature. Associated with diaphoresis and dyspnea. Took aspirin 325mg at home. No prior MI.

Medications: Lisinopril 20mg daily, Metformin 1000mg BID, Atorvastatin 40mg daily.

Vitals: BP 165/95, HR 102, RR 22, SpO2 94% on RA, Temp 98.6°F.

Assessment: Acute coronary syndrome, likely STEMI. High risk for adverse cardiac event.
Plan: Stat ECG, troponin, CBC, BMP. Heparin drip. Cardiology consult for emergent cath.`,
  },
  {
    label: 'Discharge Summary',
    text: `Patient: 45-year-old female. Admission Date: 03/20/2026. Discharge Date: 03/24/2026.

Discharge Diagnosis: Community-acquired pneumonia, right lower lobe. Resolved acute kidney injury (Stage 1).

Hospital Course: Patient admitted with fever 102.4°F, productive cough, and hypoxia (SpO2 89%). CT chest confirmed RLL consolidation. Started on ceftriaxone 1g IV daily and azithromycin 500mg IV. Blood cultures negative. Creatinine peaked at 1.8 (baseline 0.9), resolved with IV fluids.

Discharge Medications: Amoxicillin-clavulanate 875/125mg PO BID x 5 days. Acetaminophen 650mg q6h PRN fever.

Follow-up: PCP in 1 week. Repeat chest X-ray in 6 weeks.`,
  },
];

export function StreamingAnalysis() {
  const [text, setText] = useState('');
  const [stages, setStages] = useState<StageResult[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [endTime, setEndTime] = useState<number | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleStart = useCallback(() => {
    if (!text.trim()) return;
    setStages([]);
    setError(null);
    setStreaming(true);
    setStartTime(Date.now());
    setEndTime(null);

    controllerRef.current = analyzeStream(
      text,
      (event) => {
        setStages((prev) => [
          ...prev,
          { stage: event.stage, data: event.data, receivedAt: new Date() },
        ]);
        // Check if this was the last expected stage
        if (event.stage === 'done' || event.stage === 'error') {
          setStreaming(false);
          setEndTime(Date.now());
        }
      },
      () => {
        setStreaming(false);
        setEndTime(Date.now());
      }
    );
  }, [text]);

  const handleCancel = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    setStreaming(false);
    setEndTime(Date.now());
  }, []);

  const loadSample = (index: number) => {
    setText(SAMPLE_NOTES[index].text);
  };

  const completedStages = stages.filter((s) => s.stage !== 'done' && s.stage !== 'error');
  const elapsed = startTime
    ? ((endTime || Date.now()) - startTime) / 1000
    : 0;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
          <Radio className="w-7 h-7 text-primary-500" />
          Streaming Analysis
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          Real-time Server-Sent Events pipeline — watch NER, ICD-10, summarization,
          and risk scoring results arrive as each stage completes.
        </p>
      </div>

      {/* Input area */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
        <div className="flex items-center justify-between mb-1">
          <label className="text-sm font-medium text-text-primary">Clinical Note</label>
          <span className="text-xs text-text-muted">{wordCount} words</span>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a clinical note to analyse in real time..."
          rows={8}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface-dim text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm font-mono resize-y"
          disabled={streaming}
          aria-label="Clinical text input"
        />

        {/* Sample notes */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-text-muted">Sample notes:</span>
          {SAMPLE_NOTES.map((sample, i) => (
            <button
              key={i}
              onClick={() => loadSample(i)}
              disabled={streaming}
              className="px-2.5 py-1 rounded-md border border-border text-xs text-text-secondary hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50 transition-colors"
            >
              {sample.label}
            </button>
          ))}
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2">
          {!streaming ? (
            <button
              onClick={handleStart}
              disabled={!text.trim()}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary-500 text-white font-medium hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Play className="w-4 h-4" />
              Start Analysis
            </button>
          ) : (
            <button
              onClick={handleCancel}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-red-500 text-white font-medium hover:bg-red-600 transition-colors"
            >
              <Square className="w-4 h-4" />
              Cancel
            </button>
          )}
          {streaming && (
            <span className="flex items-center gap-1.5 text-sm text-text-muted">
              <Loader2 className="w-4 h-4 animate-spin" />
              Streaming... {elapsed.toFixed(1)}s
            </span>
          )}
          {!streaming && endTime && startTime && (
            <span className="flex items-center gap-1 text-xs text-text-muted">
              <Clock className="w-3 h-3" />
              Completed in {((endTime - startTime) / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm" role="alert">
          {error}
        </div>
      )}

      {/* Stage progress indicator */}
      {(streaming || completedStages.length > 0) && (
        <div className="flex items-center gap-2 px-1">
          {Object.keys(STAGE_CONFIG).map((stageKey) => {
            const completed = completedStages.some((s) => s.stage === stageKey);
            return (
              <div key={stageKey} className="flex items-center gap-1.5">
                {completed ? (
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                ) : streaming ? (
                  <Loader2 className="w-4 h-4 text-text-muted animate-spin" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-gray-300 dark:border-gray-600" />
                )}
                <span className={`text-xs ${completed ? 'text-text-primary font-medium' : 'text-text-muted'}`}>
                  {STAGE_CONFIG[stageKey].emoji} {stageKey.toUpperCase()}
                </span>
                {stageKey !== 'risk' && <span className="text-text-muted mx-1">→</span>}
              </div>
            );
          })}
        </div>
      )}

      {/* Stage result cards */}
      {completedStages.length > 0 && (
        <div className="space-y-3">
          {completedStages.map((result, index) => {
            const config = STAGE_CONFIG[result.stage] || {
              label: result.stage,
              emoji: '📦',
              color: 'border-gray-200 dark:border-gray-700',
            };
            return (
              <div
                key={`${result.stage}-${index}`}
                className={`bg-surface rounded-lg border-2 ${config.color} p-4`}
                data-testid="stage-result"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{config.emoji}</span>
                    <h3 className="font-semibold text-text-primary">{config.label}</h3>
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  </div>
                  <span className="text-xs text-text-muted">
                    {result.receivedAt.toLocaleTimeString()}
                  </span>
                </div>
                <div className="bg-surface-dim rounded-lg p-3 overflow-x-auto">
                  <pre className="text-xs text-text-secondary font-mono whitespace-pre-wrap">
                    {JSON.stringify(result.data, null, 2)}
                  </pre>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Idle empty state */}
      {!streaming && completedStages.length === 0 && !error && (
        <div className="text-center py-12 text-text-muted">
          <Sparkles className="w-10 h-10 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Submit a clinical note to see real-time analysis results.</p>
        </div>
      )}
    </div>
  );
}
