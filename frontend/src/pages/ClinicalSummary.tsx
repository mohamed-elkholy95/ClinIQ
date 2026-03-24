import { useState } from 'react';
import { FileText, CheckCircle2, Clock, AlignLeft } from 'lucide-react';
import { clsx } from 'clsx';
import type { ClinicalSummary as ClinicalSummaryType, SummaryDetail } from '../types';

// Demo summaries at different detail levels
const demoSummaries: Record<SummaryDetail, ClinicalSummaryType> = {
  brief: {
    summary: 'Patient with uncontrolled T2DM (HbA1c 7.8%) and hypertension (150/95) on metformin/lisinopril. New chest pain and fatigue. Cardiac workup recommended.',
    key_findings: [
      'Uncontrolled type 2 diabetes mellitus',
      'Elevated blood pressure despite medication',
      'New cardiac symptoms requiring evaluation',
    ],
    detail_level: 'brief',
    word_count: 28,
    generated_at: '2026-03-23T14:30:00Z',
  },
  standard: {
    summary: 'Patient presents with uncontrolled type 2 diabetes (HbA1c 7.8%) and hypertension (150/95 mmHg) currently managed with metformin 1000mg BID and lisinopril 10mg daily. New symptoms of persistent fatigue and occasional chest pain warrant cardiac evaluation. Physical examination confirms elevated blood pressure. Echocardiogram recommended with consideration for empagliflozin addition for dual glycemic and cardiovascular benefit.',
    key_findings: [
      'Uncontrolled type 2 diabetes with HbA1c 7.8%, above recommended target',
      'Blood pressure 150/95 mmHg despite lisinopril 10mg daily therapy',
      'New-onset fatigue and chest pain require urgent cardiac workup',
      'Potential for SGLT2 inhibitor therapy to address both metabolic and cardiac risk',
      'Current medication regimen may need intensification',
    ],
    detail_level: 'standard',
    word_count: 62,
    generated_at: '2026-03-23T14:30:00Z',
  },
  detailed: {
    summary: 'This 62-year-old patient presents with a complex cardiometabolic profile characterized by inadequately controlled type 2 diabetes mellitus (most recent HbA1c 7.8%, target <7.0%) and resistant hypertension (measured at 150/95 mmHg, well above the ACC/AHA target of <130/80 for diabetic patients). Current pharmacological management includes metformin 1000mg twice daily (appropriate first-line therapy at maximum dose) and lisinopril 10mg daily (suboptimal dosing with room for uptitration). The patient reports new-onset persistent fatigue and episodic chest pain, both of which raise concern for potential cardiovascular complications including but not limited to ischemic heart disease, diabetic cardiomyopathy, or decompensated heart failure. The presence of these symptoms in a patient with multiple cardiovascular risk factors necessitates urgent diagnostic evaluation including transthoracic echocardiography and potentially stress testing or coronary angiography. Pharmacological optimization should include consideration of empagliflozin or another SGLT2 inhibitor, which has demonstrated cardiovascular mortality benefit in the EMPA-REG OUTCOME trial, alongside intensification of antihypertensive therapy through either lisinopril dose increase or addition of amlodipine.',
    key_findings: [
      'HbA1c 7.8% indicates inadequate glycemic control despite maximal metformin dosing',
      'Blood pressure 150/95 mmHg represents uncontrolled hypertension with suboptimal ACE inhibitor dosing (lisinopril 10mg, max dose 40mg)',
      'New-onset persistent fatigue and episodic chest pain are concerning for underlying cardiovascular pathology',
      'Multiple overlapping cardiovascular risk factors (diabetes, hypertension, age) elevate baseline cardiac risk significantly',
      'SGLT2 inhibitor (empagliflozin) recommended based on EMPA-REG OUTCOME trial evidence for cardiovascular risk reduction',
      'Urgent echocardiography indicated to evaluate cardiac structure and function',
      'Renal function monitoring essential (creatinine 1.2) before and after SGLT2 inhibitor initiation',
      'Consider lipid panel assessment and statin optimization given cardiovascular risk profile',
    ],
    detail_level: 'detailed',
    word_count: 198,
    generated_at: '2026-03-23T14:30:00Z',
  },
};

const detailLevels: { value: SummaryDetail; label: string; description: string }[] = [
  { value: 'brief', label: 'Brief', description: 'Key points only' },
  { value: 'standard', label: 'Standard', description: 'Balanced detail' },
  { value: 'detailed', label: 'Detailed', description: 'Full clinical context' },
];

export function ClinicalSummary() {
  const [detailLevel, setDetailLevel] = useState<SummaryDetail>('standard');
  const summary = demoSummaries[detailLevel];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Clinical Summary</h1>
        <p className="mt-1 text-sm text-text-secondary">
          AI-generated clinical summaries with adjustable detail levels.
        </p>
      </div>

      {/* Detail level selector */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">
          Detail Level
        </h3>
        <div className="grid grid-cols-3 gap-3">
          {detailLevels.map((level) => (
            <button
              key={level.value}
              onClick={() => setDetailLevel(level.value)}
              className={clsx(
                'p-3 rounded-lg border text-left transition-all',
                detailLevel === level.value
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 ring-1 ring-primary-500/30'
                  : 'border-border hover:border-primary-300 hover:bg-surface-dim'
              )}
            >
              <p
                className={clsx(
                  'text-sm font-semibold',
                  detailLevel === level.value
                    ? 'text-primary-600 dark:text-primary-400'
                    : 'text-text-primary'
                )}
              >
                {level.label}
              </p>
              <p className="text-xs text-text-muted mt-0.5">{level.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Summary content */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-text-primary flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Summary
          </h3>
          <div className="flex items-center gap-4 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <AlignLeft className="w-3.5 h-3.5" />
              {summary.word_count} words
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              {new Date(summary.generated_at).toLocaleTimeString()}
            </span>
          </div>
        </div>

        <div className="prose prose-sm max-w-none">
          <p className="text-sm leading-relaxed text-text-primary whitespace-pre-wrap">
            {summary.summary}
          </p>
        </div>
      </div>

      {/* Key findings */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-base font-semibold text-text-primary mb-4">
          Key Findings
        </h3>
        <div className="space-y-3">
          {summary.key_findings.map((finding, i) => (
            <div
              key={i}
              className="flex items-start gap-3 p-3 rounded-lg bg-surface-dim"
            >
              <CheckCircle2 className="w-4 h-4 text-accent-500 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-text-primary leading-relaxed">{finding}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Metadata */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-3">
          Summary Metadata
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-text-muted">Detail Level</p>
            <p className="text-sm font-medium text-text-primary capitalize mt-0.5">
              {summary.detail_level}
            </p>
          </div>
          <div>
            <p className="text-xs text-text-muted">Word Count</p>
            <p className="text-sm font-medium text-text-primary mt-0.5">
              {summary.word_count}
            </p>
          </div>
          <div>
            <p className="text-xs text-text-muted">Key Findings</p>
            <p className="text-sm font-medium text-text-primary mt-0.5">
              {summary.key_findings.length}
            </p>
          </div>
          <div>
            <p className="text-xs text-text-muted">Generated</p>
            <p className="text-sm font-medium text-text-primary mt-0.5">
              {new Date(summary.generated_at).toLocaleString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
