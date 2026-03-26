import { useState } from 'react';
import {
  BarChart3,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Users,
  Tags,
  FileText,
  Code2,
  TrendingUp,
  Info,
} from 'lucide-react';
import {
  evaluateClassification,
  evaluateAgreement,
  evaluateNER,
  evaluateROUGE,
  evaluateICD,
  evaluateAUPRC,
} from '../services/clinical';
import type {
  ClassificationEvalResponse,
  KappaResponse,
  NEREvalResponse,
  ROUGEResponse,
  ICDEvalResponse,
  AUPRCResponse,
} from '../types/clinical';

// ---------------------------------------------------------------------------
// Metric tabs
// ---------------------------------------------------------------------------

type MetricTab =
  | 'classification'
  | 'agreement'
  | 'ner'
  | 'rouge'
  | 'icd'
  | 'auprc';

interface TabConfig {
  id: MetricTab;
  label: string;
  icon: React.ElementType;
  description: string;
}

const TABS: TabConfig[] = [
  {
    id: 'classification',
    label: 'Classification',
    icon: BarChart3,
    description: 'MCC, confusion matrix, and calibration metrics',
  },
  {
    id: 'agreement',
    label: 'Agreement',
    icon: Users,
    description: "Cohen's Kappa inter-annotator agreement",
  },
  {
    id: 'ner',
    label: 'NER',
    icon: Tags,
    description: 'Exact and partial entity span matching',
  },
  {
    id: 'rouge',
    label: 'ROUGE',
    icon: FileText,
    description: 'ROUGE-1/2/L with precision, recall, and F1',
  },
  {
    id: 'icd',
    label: 'ICD-10',
    icon: Code2,
    description: 'Hierarchical ICD-10 accuracy evaluation',
  },
  {
    id: 'auprc',
    label: 'AUPRC',
    icon: TrendingUp,
    description: 'Area Under Precision-Recall Curve',
  },
];

// ---------------------------------------------------------------------------
// Sample data for each metric
// ---------------------------------------------------------------------------

const SAMPLE_CLASSIFICATION = {
  y_true: [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
  y_pred: [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
  y_prob: [0.92, 0.15, 0.88, 0.45, 0.10, 0.72, 0.95, 0.08, 0.80, 0.91, 0.20, 0.12, 0.35, 0.85, 0.18, 0.78, 0.65, 0.90, 0.25, 0.11],
};

const SAMPLE_AGREEMENT = {
  rater_a: ['positive', 'negative', 'positive', 'neutral', 'negative', 'positive', 'positive', 'negative', 'neutral', 'positive'],
  rater_b: ['positive', 'negative', 'neutral', 'neutral', 'negative', 'positive', 'positive', 'positive', 'neutral', 'positive'],
};

const SAMPLE_NER = {
  gold_entities: [
    { entity_type: 'DISEASE', start: 0, end: 12 },
    { entity_type: 'MEDICATION', start: 25, end: 35 },
    { entity_type: 'DOSAGE', start: 36, end: 41 },
    { entity_type: 'DISEASE', start: 55, end: 67 },
    { entity_type: 'PROCEDURE', start: 80, end: 95 },
  ],
  pred_entities: [
    { entity_type: 'DISEASE', start: 0, end: 12 },
    { entity_type: 'MEDICATION', start: 24, end: 36 },
    { entity_type: 'DISEASE', start: 56, end: 65 },
    { entity_type: 'PROCEDURE', start: 80, end: 95 },
    { entity_type: 'LAB_VALUE', start: 100, end: 110 },
  ],
};

const SAMPLE_ROUGE = {
  reference:
    'Patient presented with acute chest pain radiating to left arm. ECG showed ST-elevation in leads V1-V4 consistent with anterior STEMI. Emergent cardiac catheterization revealed 95% occlusion of the LAD. Successful PCI with drug-eluting stent placement. Post-procedure troponin trending down. Started on dual antiplatelet therapy with aspirin and clopidogrel.',
  hypothesis:
    'Patient had chest pain. ECG revealed ST-elevation suggesting anterior STEMI. Cardiac catheterization showed LAD occlusion. PCI performed with stent placement. Troponin improving. Started on aspirin and clopidogrel.',
};

const SAMPLE_ICD = {
  gold_codes: ['E11.65', 'I10', 'J44.1', 'N18.3', 'E78.5', 'I25.10', 'G47.33', 'E66.01'],
  pred_codes: ['E11.9', 'I10', 'J44.0', 'N18.9', 'E78.5', 'I25.10', 'G47.30', 'E66.09'],
};

const SAMPLE_AUPRC = {
  y_true: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
  y_scores: [0.05, 0.12, 0.08, 0.03, 0.15, 0.02, 0.10, 0.07, 0.04, 0.20, 0.11, 0.06, 0.09, 0.14, 0.88, 0.75, 0.30, 0.18, 0.92, 0.22],
  label: 'rare_diagnosis',
};

// ---------------------------------------------------------------------------
// Helper: score colour
// ---------------------------------------------------------------------------

function scoreColor(value: number, thresholds: [number, number, number] = [0.8, 0.6, 0.4]): string {
  if (value >= thresholds[0]) return 'text-green-600 dark:text-green-400';
  if (value >= thresholds[1]) return 'text-blue-600 dark:text-blue-400';
  if (value >= thresholds[2]) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-red-600 dark:text-red-400';
}

function scoreBg(value: number, thresholds: [number, number, number] = [0.8, 0.6, 0.4]): string {
  if (value >= thresholds[0]) return 'bg-green-100 dark:bg-green-900/30 border-green-300 dark:border-green-700';
  if (value >= thresholds[1]) return 'bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700';
  if (value >= thresholds[2]) return 'bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700';
  return 'bg-red-100 dark:bg-red-900/30 border-red-300 dark:border-red-700';
}

// MCC ranges from -1 to +1, so we normalise to 0-1 for colour thresholds
function mccColor(mcc: number): string {
  return scoreColor((mcc + 1) / 2, [0.8, 0.6, 0.4]);
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 text-center">
      <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Result panels for each metric
// ---------------------------------------------------------------------------

function ClassificationResultPanel({ result }: { result: ClassificationEvalResponse }) {
  const total = result.tp + result.fp + result.fn + result.tn;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="MCC" value={result.mcc.toFixed(3)} sub="(-1 to +1)" />
        <StatCard label="True Positive" value={String(result.tp)} />
        <StatCard label="False Positive" value={String(result.fp)} />
        <StatCard label="Accuracy" value={((result.tp + result.tn) / total * 100).toFixed(1) + '%'} />
      </div>

      {/* Confusion matrix */}
      <div>
        <h4 className="text-sm font-semibold mb-2">Confusion Matrix</h4>
        <div className="grid grid-cols-3 gap-1 max-w-xs text-sm text-center">
          <div />
          <div className="font-medium text-gray-500">Pred 0</div>
          <div className="font-medium text-gray-500">Pred 1</div>
          <div className="font-medium text-gray-500">True 0</div>
          <div className="bg-green-100 dark:bg-green-900/30 rounded p-2 font-mono">{result.tn}</div>
          <div className="bg-red-100 dark:bg-red-900/30 rounded p-2 font-mono">{result.fp}</div>
          <div className="font-medium text-gray-500">True 1</div>
          <div className="bg-red-100 dark:bg-red-900/30 rounded p-2 font-mono">{result.fn}</div>
          <div className="bg-green-100 dark:bg-green-900/30 rounded p-2 font-mono">{result.tp}</div>
        </div>
      </div>

      {/* Calibration */}
      {result.calibration && (
        <div>
          <h4 className="text-sm font-semibold mb-2">Calibration</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <StatCard
              label="ECE"
              value={result.calibration.expected_calibration_error.toFixed(4)}
              sub="lower is better"
            />
            <StatCard
              label="Brier Score"
              value={result.calibration.brier_score.toFixed(4)}
              sub="0 = perfect"
            />
            <StatCard label="Bins" value={String(result.calibration.n_bins)} />
          </div>
        </div>
      )}

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

function AgreementResultPanel({ result }: { result: KappaResponse }) {
  const kappaLabel =
    result.kappa >= 0.81
      ? 'Almost Perfect'
      : result.kappa >= 0.61
        ? 'Substantial'
        : result.kappa >= 0.41
          ? 'Moderate'
          : result.kappa >= 0.21
            ? 'Fair'
            : result.kappa >= 0.01
              ? 'Slight'
              : 'Poor';

  return (
    <div className="space-y-4">
      <div className={`rounded-lg border p-6 text-center ${scoreBg(result.kappa)}`}>
        <p className="text-xs uppercase tracking-wide text-gray-500">Cohen's Kappa</p>
        <p className={`text-4xl font-bold mt-1 ${scoreColor(result.kappa)}`}>
          {result.kappa.toFixed(3)}
        </p>
        <p className="text-sm mt-1 font-medium">{kappaLabel} Agreement</p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Observed Agreement" value={(result.observed_agreement * 100).toFixed(1) + '%'} />
        <StatCard label="Expected Agreement" value={(result.expected_agreement * 100).toFixed(1) + '%'} />
        <StatCard label="Items" value={String(result.n_items)} />
      </div>

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

function NERResultPanel({ result }: { result: NEREvalResponse }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className={`rounded-lg border p-4 text-center ${scoreBg(result.exact_f1)}`}>
          <p className="text-xs text-gray-500 uppercase">Exact F1</p>
          <p className={`text-2xl font-bold ${scoreColor(result.exact_f1)}`}>
            {(result.exact_f1 * 100).toFixed(1)}%
          </p>
        </div>
        <div className={`rounded-lg border p-4 text-center ${scoreBg(result.partial_f1)}`}>
          <p className="text-xs text-gray-500 uppercase">Partial F1</p>
          <p className={`text-2xl font-bold ${scoreColor(result.partial_f1)}`}>
            {(result.partial_f1 * 100).toFixed(1)}%
          </p>
        </div>
        <div className={`rounded-lg border p-4 text-center ${scoreBg(result.type_weighted_f1)}`}>
          <p className="text-xs text-gray-500 uppercase">Type-Weighted F1</p>
          <p className={`text-2xl font-bold ${scoreColor(result.type_weighted_f1)}`}>
            {(result.type_weighted_f1 * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Gold Entities" value={String(result.n_gold)} />
        <StatCard label="Pred Entities" value={String(result.n_pred)} />
        <StatCard label="Exact Matches" value={String(result.n_exact_matches)} />
        <StatCard label="Partial Matches" value={String(result.n_partial_matches)} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Mean Overlap" value={(result.mean_overlap * 100).toFixed(1) + '%'} />
        <StatCard label="Unmatched Pred" value={String(result.n_unmatched_pred)} />
      </div>

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

function ROUGEResultPanel({ result }: { result: ROUGEResponse }) {
  const variants = [
    { name: 'ROUGE-1', data: result.rouge1 },
    { name: 'ROUGE-2', data: result.rouge2 },
    { name: 'ROUGE-L', data: result.rougeL },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        {variants.map((v) => (
          <div
            key={v.name}
            className={`rounded-lg border p-4 text-center ${scoreBg(v.data.f1)}`}
          >
            <p className="text-xs text-gray-500 uppercase">{v.name} F1</p>
            <p className={`text-2xl font-bold ${scoreColor(v.data.f1)}`}>
              {(v.data.f1 * 100).toFixed(1)}%
            </p>
            <div className="mt-2 text-xs text-gray-500 space-y-0.5">
              <p>P: {(v.data.precision * 100).toFixed(1)}%</p>
              <p>R: {(v.data.recall * 100).toFixed(1)}%</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Reference Length" value={String(result.reference_length)} sub="words" />
        <StatCard label="Hypothesis Length" value={String(result.hypothesis_length)} sub="words" />
        <StatCard label="Compression" value={result.length_ratio.toFixed(2)} sub="length ratio" />
      </div>

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

function ICDResultPanel({ result }: { result: ICDEvalResponse }) {
  const levels = [
    { label: 'Chapter', accuracy: result.chapter_accuracy, matches: result.chapter_matches },
    { label: '3-Char Block', accuracy: result.block_accuracy, matches: result.block_matches },
    { label: 'Full Code', accuracy: result.full_code_accuracy, matches: result.full_code_matches },
  ];

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        {levels.map((l) => (
          <div key={l.label} className="flex items-center gap-3">
            <span className="w-28 text-sm font-medium text-gray-600 dark:text-gray-300">
              {l.label}
            </span>
            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  l.accuracy >= 0.8
                    ? 'bg-green-500'
                    : l.accuracy >= 0.5
                      ? 'bg-blue-500'
                      : 'bg-yellow-500'
                }`}
                style={{ width: `${l.accuracy * 100}%` }}
              />
            </div>
            <span className="w-20 text-sm text-right font-mono">
              {(l.accuracy * 100).toFixed(1)}%
            </span>
            <span className="w-16 text-xs text-gray-400 text-right">
              {l.matches}/{result.n_samples}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg text-sm">
        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
        <span className="text-blue-700 dark:text-blue-300">
          Chapter accuracy reveals organ-system-level correctness. Block accuracy captures
          disease category. Full code measures exact ICD-10-CM precision including specificity digits.
        </span>
      </div>

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

function AUPRCResultPanel({ result }: { result: AUPRCResponse }) {
  const prevalence = result.n_positive / result.n_total;

  return (
    <div className="space-y-4">
      <div className={`rounded-lg border p-6 text-center ${scoreBg(result.auprc)}`}>
        <p className="text-xs uppercase tracking-wide text-gray-500">AUPRC</p>
        <p className={`text-4xl font-bold mt-1 ${scoreColor(result.auprc)}`}>
          {result.auprc.toFixed(4)}
        </p>
        <p className="text-sm mt-1 text-gray-500">
          Label: <span className="font-mono">{result.label}</span>
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Positives" value={String(result.n_positive)} />
        <StatCard label="Total" value={String(result.n_total)} />
        <StatCard
          label="Prevalence"
          value={(prevalence * 100).toFixed(1) + '%'}
          sub="baseline"
        />
      </div>

      <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg text-sm">
        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
        <span className="text-blue-700 dark:text-blue-300">
          AUPRC is more informative than AUROC for imbalanced clinical datasets.
          A random classifier achieves AUPRC ≈ prevalence ({(prevalence * 100).toFixed(1)}%).
        </span>
      </div>

      <p className="text-xs text-gray-400">{result.processing_time_ms.toFixed(1)} ms</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type AnyResult =
  | ClassificationEvalResponse
  | KappaResponse
  | NEREvalResponse
  | ROUGEResponse
  | ICDEvalResponse
  | AUPRCResponse;

export function EvaluationDashboard() {
  const [activeTab, setActiveTab] = useState<MetricTab>('classification');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnyResult | null>(null);

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res: AnyResult;
      switch (activeTab) {
        case 'classification':
          res = await evaluateClassification(SAMPLE_CLASSIFICATION);
          break;
        case 'agreement':
          res = await evaluateAgreement(SAMPLE_AGREEMENT);
          break;
        case 'ner':
          res = await evaluateNER(SAMPLE_NER);
          break;
        case 'rouge':
          res = await evaluateROUGE(SAMPLE_ROUGE);
          break;
        case 'icd':
          res = await evaluateICD(SAMPLE_ICD);
          break;
        case 'auprc':
          res = await evaluateAUPRC(SAMPLE_AUPRC);
          break;
      }
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (tab: MetricTab) => {
    setActiveTab(tab);
    setResult(null);
    setError(null);
  };

  const renderResult = () => {
    if (!result) return null;

    switch (activeTab) {
      case 'classification':
        return <ClassificationResultPanel result={result as ClassificationEvalResponse} />;
      case 'agreement':
        return <AgreementResultPanel result={result as KappaResponse} />;
      case 'ner':
        return <NERResultPanel result={result as NEREvalResponse} />;
      case 'rouge':
        return <ROUGEResultPanel result={result as ROUGEResponse} />;
      case 'icd':
        return <ICDResultPanel result={result as ICDEvalResponse} />;
      case 'auprc':
        return <AUPRCResultPanel result={result as AUPRCResponse} />;
    }
  };

  const activeConfig = TABS.find((t) => t.id === activeTab)!;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Evaluation Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Compute clinical NLP evaluation metrics — classification, NER, summarisation,
          ICD-10, and annotation agreement.
        </p>
      </div>

      {/* Metric tabs */}
      <div className="flex flex-wrap gap-2">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Active metric description */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-start gap-3">
          <activeConfig.icon className="h-5 w-5 text-indigo-500 mt-0.5" />
          <div>
            <h3 className="font-semibold">{activeConfig.label}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">{activeConfig.description}</p>
          </div>
        </div>

        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="mt-4 px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Evaluating…
            </>
          ) : (
            <>
              <BarChart3 className="h-4 w-4" />
              Run with Sample Data
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 p-4 rounded-lg">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
            <h3 className="font-semibold">Results</h3>
          </div>
          {renderResult()}
        </div>
      )}
    </div>
  );
}
