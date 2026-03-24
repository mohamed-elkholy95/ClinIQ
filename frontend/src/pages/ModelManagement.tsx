import { clsx } from 'clsx';
import {
  Cpu,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Activity,
  TrendingUp,
} from 'lucide-react';
import { ConfidenceBar } from '../components/ConfidenceBar';
import type { ModelInfo, ModelStatus } from '../types';

const demoModels: ModelInfo[] = [
  {
    id: 'm1',
    name: 'ClinIQ-NER',
    version: '2.3.1',
    type: 'Named Entity Recognition',
    status: 'active',
    accuracy: 0.94,
    f1_score: 0.92,
    precision: 0.93,
    recall: 0.91,
    last_trained: '2026-03-15T10:00:00Z',
    deployed_at: '2026-03-16T08:00:00Z',
    description: 'Clinical named entity recognition model for diseases, medications, procedures, anatomy, symptoms, and lab values. Based on BioBERT with clinical domain fine-tuning.',
  },
  {
    id: 'm2',
    name: 'ClinIQ-ICD',
    version: '1.8.0',
    type: 'ICD-10 Code Prediction',
    status: 'active',
    accuracy: 0.89,
    f1_score: 0.87,
    precision: 0.90,
    recall: 0.84,
    last_trained: '2026-03-10T14:00:00Z',
    deployed_at: '2026-03-11T09:00:00Z',
    description: 'Multi-label ICD-10 code prediction model using hierarchical attention networks. Supports full ICD-10-CM code set with chapter-level and code-level predictions.',
  },
  {
    id: 'm3',
    name: 'ClinIQ-Summary',
    version: '1.5.2',
    type: 'Clinical Summarization',
    status: 'active',
    accuracy: 0.91,
    f1_score: 0.88,
    precision: 0.89,
    recall: 0.87,
    last_trained: '2026-03-08T16:00:00Z',
    deployed_at: '2026-03-09T10:00:00Z',
    description: 'Abstractive clinical text summarization model with controllable detail levels. Generates structured summaries with key findings extraction.',
  },
  {
    id: 'm4',
    name: 'ClinIQ-Risk',
    version: '1.2.0',
    type: 'Risk Assessment',
    status: 'active',
    accuracy: 0.86,
    f1_score: 0.83,
    precision: 0.85,
    recall: 0.81,
    last_trained: '2026-03-01T12:00:00Z',
    deployed_at: '2026-03-02T08:00:00Z',
    description: 'Multi-factor clinical risk assessment model. Evaluates cardiovascular, metabolic, medication, and general risk categories with evidence-based scoring.',
  },
  {
    id: 'm5',
    name: 'ClinIQ-Temporal',
    version: '0.9.0',
    type: 'Temporal Extraction',
    status: 'training',
    accuracy: 0.78,
    f1_score: 0.74,
    precision: 0.76,
    recall: 0.72,
    last_trained: '2026-03-20T18:00:00Z',
    description: 'Temporal expression and event ordering model for clinical timeline construction. Currently in training for v1.0 release.',
  },
  {
    id: 'm6',
    name: 'ClinIQ-Relation',
    version: '0.6.0',
    type: 'Relation Extraction',
    status: 'inactive',
    accuracy: 0.71,
    f1_score: 0.68,
    precision: 0.70,
    recall: 0.66,
    last_trained: '2026-02-15T10:00:00Z',
    description: 'Clinical relation extraction model for identifying relationships between entities (treatment-of, causes, contraindicates). Pending retraining with expanded dataset.',
  },
];

const statusConfig: Record<ModelStatus, { icon: typeof CheckCircle2; color: string; bg: string; label: string }> = {
  active: {
    icon: CheckCircle2,
    color: 'text-green-600 dark:text-green-400',
    bg: 'bg-green-50 dark:bg-green-950/30',
    label: 'Active',
  },
  inactive: {
    icon: XCircle,
    color: 'text-gray-500 dark:text-gray-400',
    bg: 'bg-gray-50 dark:bg-gray-800',
    label: 'Inactive',
  },
  training: {
    icon: Loader2,
    color: 'text-blue-600 dark:text-blue-400',
    bg: 'bg-blue-50 dark:bg-blue-950/30',
    label: 'Training',
  },
  failed: {
    icon: XCircle,
    color: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-50 dark:bg-red-950/30',
    label: 'Failed',
  },
};

function MetricCard({ label, value }: { label: string; value: number | undefined }) {
  if (value === undefined) return null;
  return (
    <div>
      <p className="text-xs text-text-muted">{label}</p>
      <div className="mt-1">
        <ConfidenceBar value={value} size="sm" />
      </div>
    </div>
  );
}

export function ModelManagement() {
  const activeCount = demoModels.filter((m) => m.status === 'active').length;
  const trainingCount = demoModels.filter((m) => m.status === 'training').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Model Management</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Monitor and manage deployed NLP models, their performance, and training status.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
          <div className="p-2.5 rounded-lg bg-primary-50 dark:bg-primary-900/30">
            <Cpu className="w-5 h-5 text-primary-500" />
          </div>
          <div>
            <p className="text-2xl font-bold text-text-primary">{demoModels.length}</p>
            <p className="text-xs text-text-muted">Total Models</p>
          </div>
        </div>
        <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
          <div className="p-2.5 rounded-lg bg-green-50 dark:bg-green-950/30">
            <Activity className="w-5 h-5 text-green-500" />
          </div>
          <div>
            <p className="text-2xl font-bold text-text-primary">{activeCount}</p>
            <p className="text-xs text-text-muted">Active / Deployed</p>
          </div>
        </div>
        <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
          <div className="p-2.5 rounded-lg bg-blue-50 dark:bg-blue-950/30">
            <TrendingUp className="w-5 h-5 text-blue-500" />
          </div>
          <div>
            <p className="text-2xl font-bold text-text-primary">{trainingCount}</p>
            <p className="text-xs text-text-muted">Currently Training</p>
          </div>
        </div>
      </div>

      {/* Model cards */}
      <div className="space-y-4">
        {demoModels.map((model) => {
          const status = statusConfig[model.status];
          const StatusIcon = status.icon;

          return (
            <div
              key={model.id}
              className="rounded-xl border border-border bg-surface overflow-hidden card-hover"
            >
              <div className="p-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-1">
                      <h3 className="text-lg font-semibold text-text-primary">
                        {model.name}
                      </h3>
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-dim text-text-muted border border-border">
                        v{model.version}
                      </span>
                      <span
                        className={clsx(
                          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium',
                          status.bg,
                          status.color
                        )}
                      >
                        <StatusIcon
                          className={clsx(
                            'w-3 h-3',
                            model.status === 'training' && 'animate-spin'
                          )}
                        />
                        {status.label}
                      </span>
                    </div>
                    <p className="text-sm text-text-muted mb-1">{model.type}</p>
                    <p className="text-sm text-text-secondary leading-relaxed">
                      {model.description}
                    </p>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4 pt-4 border-t border-border">
                  <MetricCard label="Accuracy" value={model.accuracy} />
                  <MetricCard label="F1 Score" value={model.f1_score} />
                  <MetricCard label="Precision" value={model.precision} />
                  <MetricCard label="Recall" value={model.recall} />
                </div>

                {/* Timestamps */}
                <div className="flex flex-wrap items-center gap-4 mt-4 pt-3 border-t border-border text-xs text-text-muted">
                  {model.last_trained && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Trained:{' '}
                      {new Date(model.last_trained).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })}
                    </span>
                  )}
                  {model.deployed_at && (
                    <span className="flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 text-green-500" />
                      Deployed:{' '}
                      {new Date(model.deployed_at).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })}
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
