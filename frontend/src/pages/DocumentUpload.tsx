import { useState, useCallback, useRef } from 'react';
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  X,
  Sparkles,
} from 'lucide-react';
import { clsx } from 'clsx';
import { EntityTag } from '../components/EntityTag';
import { ConfidenceBar } from '../components/ConfidenceBar';
import { RiskGauge } from '../components/RiskGauge';
import type { AnalysisResult, Entity, RiskLevel } from '../types';

// Demo result data
const demoResult: AnalysisResult = {
  id: 'demo-1',
  text: 'Patient presents with type 2 diabetes mellitus and hypertension. Currently taking metformin 1000mg twice daily and lisinopril 10mg daily. Recent HbA1c was 7.8%. Patient reports persistent fatigue and occasional chest pain. Physical examination reveals elevated blood pressure at 150/95 mmHg. Recommend echocardiogram and adjustment of antihypertensive therapy. Consider adding empagliflozin for cardiovascular benefit.',
  entities: [
    { id: 'e1', text: 'type 2 diabetes mellitus', type: 'disease', start: 22, end: 47, confidence: 0.97 },
    { id: 'e2', text: 'hypertension', type: 'disease', start: 52, end: 64, confidence: 0.95 },
    { id: 'e3', text: 'metformin', type: 'medication', start: 85, end: 94, confidence: 0.99 },
    { id: 'e4', text: 'lisinopril', type: 'medication', start: 118, end: 128, confidence: 0.98 },
    { id: 'e5', text: 'HbA1c', type: 'lab_value', start: 147, end: 152, confidence: 0.96 },
    { id: 'e6', text: 'fatigue', type: 'symptom', start: 184, end: 191, confidence: 0.89 },
    { id: 'e7', text: 'chest pain', type: 'symptom', start: 207, end: 217, confidence: 0.92 },
    { id: 'e8', text: 'blood pressure', type: 'lab_value', start: 261, end: 275, confidence: 0.94 },
    { id: 'e9', text: 'echocardiogram', type: 'procedure', start: 303, end: 317, confidence: 0.91 },
    { id: 'e10', text: 'empagliflozin', type: 'medication', start: 379, end: 392, confidence: 0.93 },
  ],
  icd_predictions: [
    { code: 'E11.65', description: 'Type 2 diabetes mellitus with hyperglycemia', confidence: 0.94, chapter: 'Endocrine', category: 'Diabetes mellitus', evidence: ['type 2 diabetes mellitus', 'HbA1c 7.8%'] },
    { code: 'I10', description: 'Essential (primary) hypertension', confidence: 0.92, chapter: 'Circulatory', category: 'Hypertensive diseases', evidence: ['hypertension', 'blood pressure 150/95'] },
    { code: 'R53.83', description: 'Other fatigue', confidence: 0.78, chapter: 'Symptoms', category: 'General symptoms', evidence: ['persistent fatigue'] },
    { code: 'R07.9', description: 'Chest pain, unspecified', confidence: 0.75, chapter: 'Symptoms', category: 'Chest symptoms', evidence: ['occasional chest pain'] },
  ],
  summary: {
    summary: 'Patient with uncontrolled type 2 diabetes (HbA1c 7.8%) and hypertension (150/95 mmHg) on metformin and lisinopril. Presenting with fatigue and chest pain. Cardiac workup recommended with potential therapy adjustment including empagliflozin.',
    key_findings: [
      'Uncontrolled type 2 diabetes with HbA1c 7.8%',
      'Elevated blood pressure at 150/95 mmHg despite lisinopril',
      'New symptoms of fatigue and chest pain require cardiac evaluation',
      'Consideration for SGLT2 inhibitor (empagliflozin) for cardioprotection',
    ],
    detail_level: 'standard',
    word_count: 42,
    generated_at: '2026-03-23T14:30:00Z',
  },
  risk_assessment: {
    overall_score: 68,
    level: 'high' as RiskLevel,
    factors: [
      { name: 'Uncontrolled diabetes', score: 75, category: 'Metabolic', description: 'HbA1c above target range' },
      { name: 'Hypertension', score: 70, category: 'Cardiovascular', description: 'Blood pressure significantly elevated' },
      { name: 'Chest pain', score: 65, category: 'Cardiovascular', description: 'New cardiac symptom requiring evaluation' },
      { name: 'Polypharmacy risk', score: 45, category: 'Medication', description: 'Multiple medications with potential interactions' },
    ],
    recommendations: [
      'Urgent echocardiogram to evaluate chest pain',
      'Consider intensifying antihypertensive therapy',
      'Monitor renal function with SGLT2 inhibitor addition',
      'Follow-up HbA1c in 3 months',
    ],
    category_scores: {
      Cardiovascular: 72,
      Metabolic: 75,
      Medication: 45,
    },
  },
  processing_time_ms: 342,
  created_at: '2026-03-23T14:30:00Z',
};

function HighlightedText({ text, entities }: { text: string; entities: Entity[] }) {
  if (!entities.length) return <span>{text}</span>;

  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const segments: React.ReactNode[] = [];
  let lastEnd = 0;

  const colors: Record<string, string> = {
    disease: 'bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-200 border-b-2 border-red-400',
    medication: 'bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-200 border-b-2 border-blue-400',
    procedure: 'bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-200 border-b-2 border-green-400',
    anatomy: 'bg-purple-100 dark:bg-purple-900/40 text-purple-800 dark:text-purple-200 border-b-2 border-purple-400',
    symptom: 'bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-200 border-b-2 border-amber-400',
    lab_value: 'bg-cyan-100 dark:bg-cyan-900/40 text-cyan-800 dark:text-cyan-200 border-b-2 border-cyan-400',
  };

  sorted.forEach((entity, i) => {
    if (entity.start > lastEnd) {
      segments.push(
        <span key={`text-${i}`}>{text.slice(lastEnd, entity.start)}</span>
      );
    }
    segments.push(
      <mark
        key={entity.id}
        className={clsx('rounded px-0.5 cursor-help', colors[entity.type])}
        title={`${entity.type}: ${entity.text} (${Math.round(entity.confidence * 100)}%)`}
      >
        {text.slice(entity.start, entity.end)}
      </mark>
    );
    lastEnd = entity.end;
  });

  if (lastEnd < text.length) {
    segments.push(<span key="text-end">{text.slice(lastEnd)}</span>);
  }

  return <>{segments}</>;
}

export function DocumentUpload() {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = useCallback(() => {
    if (!inputText.trim()) return;
    setIsAnalyzing(true);
    // Simulate analysis with demo data
    setTimeout(() => {
      setResult({
        ...demoResult,
        text: inputText,
      });
      setIsAnalyzing(false);
    }, 1500);
  }, [inputText]);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'text/plain') {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setInputText(ev.target?.result as string);
      };
      reader.readAsText(file);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setInputText(ev.target?.result as string);
      };
      reader.readAsText(file);
    }
  }, []);

  const handleClear = useCallback(() => {
    setInputText('');
    setResult(null);
  }, []);

  const loadSample = useCallback(() => {
    setInputText(demoResult.text);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Document Upload</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Upload or paste clinical text for NLP analysis.
        </p>
      </div>

      {/* Input section */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-text-primary">
            Clinical Text Input
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={loadSample}
              className="text-xs font-medium text-primary-500 hover:text-primary-600 transition-colors"
            >
              Load sample
            </button>
            {inputText && (
              <button
                onClick={handleClear}
                className="p-1 rounded text-text-muted hover:text-text-primary transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Text area */}
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste clinical text here (discharge summaries, progress notes, lab reports...)"
          className={clsx(
            'w-full h-40 p-4 rounded-lg border text-sm resize-y',
            'bg-surface-dim border-border text-text-primary placeholder:text-text-muted',
            'focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500',
            'transition-colors'
          )}
        />

        {/* File upload zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleFileDrop}
          onClick={() => fileRef.current?.click()}
          className={clsx(
            'drop-zone mt-4 rounded-lg p-6 text-center cursor-pointer',
            dragActive && 'active'
          )}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.doc,.docx"
            onChange={handleFileSelect}
            className="hidden"
          />
          <Upload className="w-8 h-8 mx-auto text-text-muted mb-2" />
          <p className="text-sm text-text-secondary">
            <span className="font-medium text-primary-500">Click to upload</span>{' '}
            or drag and drop
          </p>
          <p className="text-xs text-text-muted mt-1">
            TXT, DOC, DOCX up to 10MB
          </p>
        </div>

        {/* Analyze button */}
        <div className="mt-4 flex items-center justify-between">
          <p className="text-xs text-text-muted">
            {inputText.length > 0 && `${inputText.split(/\s+/).filter(Boolean).length} words`}
          </p>
          <button
            onClick={handleAnalyze}
            disabled={!inputText.trim() || isAnalyzing}
            className={clsx(
              'inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold',
              'bg-primary-500 text-white shadow-sm',
              'hover:bg-primary-600 transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Processing info */}
          <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
            <CheckCircle2 className="w-4 h-4" />
            Analysis completed in {result.processing_time_ms}ms
          </div>

          {/* Highlighted entities in text */}
          <div className="rounded-xl border border-border bg-surface p-6">
            <h3 className="text-base font-semibold text-text-primary mb-4">
              Annotated Text
            </h3>
            <div className="text-sm leading-relaxed text-text-primary">
              <HighlightedText text={result.text} entities={result.entities} />
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {result.entities.map((e) => (
                <EntityTag
                  key={e.id}
                  type={e.type}
                  text={e.text}
                  confidence={e.confidence}
                  size="sm"
                />
              ))}
            </div>
          </div>

          {/* Results grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ICD Predictions */}
            <div className="rounded-xl border border-border bg-surface p-6">
              <h3 className="text-base font-semibold text-text-primary mb-4">
                <FileText className="w-4 h-4 inline-block mr-2 -mt-0.5" />
                ICD-10 Predictions
              </h3>
              <div className="space-y-3">
                {result.icd_predictions.map((pred) => (
                  <div key={pred.code} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-mono font-semibold text-primary-500">
                        {pred.code}
                      </span>
                      <span className="text-xs text-text-muted">{pred.chapter}</span>
                    </div>
                    <p className="text-sm text-text-secondary">{pred.description}</p>
                    <ConfidenceBar value={pred.confidence} size="sm" />
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Score */}
            <div className="rounded-xl border border-border bg-surface p-6">
              <h3 className="text-base font-semibold text-text-primary mb-4">
                <AlertCircle className="w-4 h-4 inline-block mr-2 -mt-0.5" />
                Risk Assessment
              </h3>
              <div className="flex justify-center mb-4">
                <RiskGauge
                  score={result.risk_assessment.overall_score}
                  level={result.risk_assessment.level}
                  size={140}
                />
              </div>
              <div className="space-y-2">
                {result.risk_assessment.factors.slice(0, 3).map((f) => (
                  <div key={f.name} className="flex items-center justify-between text-sm">
                    <span className="text-text-secondary">{f.name}</span>
                    <span className="font-medium text-text-primary">{f.score}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Summary */}
            <div className="lg:col-span-2 rounded-xl border border-border bg-surface p-6">
              <h3 className="text-base font-semibold text-text-primary mb-3">
                Clinical Summary
              </h3>
              <p className="text-sm text-text-secondary leading-relaxed">
                {result.summary.summary}
              </p>
              <div className="mt-4">
                <h4 className="text-sm font-semibold text-text-primary mb-2">
                  Key Findings
                </h4>
                <ul className="space-y-1.5">
                  {result.summary.key_findings.map((finding, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                      <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      {finding}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
