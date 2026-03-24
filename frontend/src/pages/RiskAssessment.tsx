import { clsx } from 'clsx';
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { RiskGauge, riskColors } from '../components/RiskGauge';
import { ConfidenceBar } from '../components/ConfidenceBar';
import type { RiskAssessment as RiskAssessmentType, RiskLevel } from '../types';

const demoRisk: RiskAssessmentType = {
  overall_score: 68,
  level: 'high' as RiskLevel,
  factors: [
    { name: 'Uncontrolled Diabetes', score: 75, category: 'Metabolic', description: 'HbA1c 7.8% exceeds target range of <7.0%. Long-term microvascular and macrovascular complications risk elevated.' },
    { name: 'Resistant Hypertension', score: 70, category: 'Cardiovascular', description: 'Blood pressure 150/95 mmHg despite ACE inhibitor therapy. Target is <130/80 for diabetic patients.' },
    { name: 'Chest Pain Symptom', score: 65, category: 'Cardiovascular', description: 'New-onset chest pain in patient with multiple cardiovascular risk factors requires urgent evaluation.' },
    { name: 'Chronic Fatigue', score: 50, category: 'General', description: 'Persistent fatigue may indicate decompensation, anemia, or thyroid dysfunction.' },
    { name: 'Polypharmacy Risk', score: 45, category: 'Medication', description: 'Multiple concurrent medications increase drug interaction potential and adherence challenges.' },
    { name: 'Renal Function', score: 40, category: 'Metabolic', description: 'Creatinine 1.2 mg/dL suggests early-stage chronic kidney disease risk in diabetic patient.' },
  ],
  recommendations: [
    'Order urgent transthoracic echocardiogram to evaluate cardiac structure and function',
    'Consider cardiac stress test or coronary angiography given chest pain with risk factors',
    'Uptitrate lisinopril from 10mg to 20-40mg daily or add amlodipine for blood pressure control',
    'Initiate empagliflozin 10mg for combined glycemic and cardiovascular benefit',
    'Order comprehensive metabolic panel to assess renal function baseline',
    'Schedule follow-up HbA1c measurement in 3 months to assess therapy response',
    'Consider thyroid function tests to evaluate fatigue etiology',
    'Provide patient education on warning signs requiring emergency evaluation',
  ],
  category_scores: {
    Cardiovascular: 72,
    Metabolic: 68,
    Medication: 45,
    General: 50,
  },
};

function getScoreColor(score: number): string {
  if (score >= 70) return '#ef4444';
  if (score >= 50) return '#f97316';
  if (score >= 30) return '#eab308';
  return '#22c55e';
}

export function RiskAssessment() {
  const risk = demoRisk;

  const categoryData = Object.entries(risk.category_scores).map(
    ([category, score]) => ({
      category,
      score,
    })
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Risk Assessment</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Comprehensive patient risk evaluation with factor analysis and recommendations.
        </p>
      </div>

      {/* Top section: gauge + category scores */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk gauge */}
        <div className="rounded-xl border border-border bg-surface p-6 flex flex-col items-center justify-center">
          <h3 className="text-base font-semibold text-text-primary mb-4">
            Overall Risk Score
          </h3>
          <RiskGauge score={risk.overall_score} level={risk.level} size={180} />
        </div>

        {/* Category scores chart */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-surface p-6">
          <h3 className="text-base font-semibold text-text-primary mb-4">
            <TrendingUp className="w-4 h-4 inline-block mr-2 -mt-0.5" />
            Category Risk Scores
          </h3>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoryData}
                layout="vertical"
                margin={{ top: 5, right: 20, bottom: 5, left: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tick={{ fontSize: 11 }}
                  className="text-text-muted"
                />
                <YAxis
                  dataKey="category"
                  type="category"
                  tick={{ fontSize: 12 }}
                  className="text-text-muted"
                  width={75}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '8px',
                    fontSize: '13px',
                  }}
                  formatter={(value: number) => [`${value}/100`, 'Score']}
                />
                <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
                  {categoryData.map((entry, idx) => (
                    <Cell key={idx} fill={getScoreColor(entry.score)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Risk factors */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-base font-semibold text-text-primary mb-4">
          <ShieldAlert className="w-4 h-4 inline-block mr-2 -mt-0.5" />
          Risk Factor Breakdown
        </h3>
        <div className="space-y-4">
          {risk.factors.map((factor) => (
            <div
              key={factor.name}
              className="p-4 rounded-lg border border-border bg-surface-dim"
            >
              <div className="flex items-start justify-between gap-4 mb-2">
                <div>
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-semibold text-text-primary">
                      {factor.name}
                    </h4>
                    <span className="text-xs px-2 py-0.5 rounded bg-surface text-text-muted border border-border">
                      {factor.category}
                    </span>
                  </div>
                  <p className="text-sm text-text-secondary mt-1">
                    {factor.description}
                  </p>
                </div>
                <span
                  className={clsx(
                    'text-lg font-bold tabular-nums flex-shrink-0',
                    factor.score >= 70
                      ? 'text-red-500'
                      : factor.score >= 50
                        ? 'text-orange-500'
                        : 'text-green-500'
                  )}
                >
                  {factor.score}
                </span>
              </div>
              <ConfidenceBar value={factor.score / 100} size="sm" showPercentage={false} />
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-base font-semibold text-text-primary mb-4">
          <AlertTriangle className="w-4 h-4 inline-block mr-2 -mt-0.5" />
          Clinical Recommendations
        </h3>
        <div className="space-y-3">
          {risk.recommendations.map((rec, i) => (
            <div
              key={i}
              className="flex items-start gap-3 p-3 rounded-lg hover:bg-surface-dim transition-colors"
            >
              <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900/30 flex-shrink-0 mt-0.5">
                <span className="text-xs font-bold text-primary-600 dark:text-primary-400">
                  {i + 1}
                </span>
              </div>
              <p className="text-sm text-text-primary leading-relaxed">{rec}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Risk level legend */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-3">
          Risk Level Reference
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {(Object.entries(riskColors) as [RiskLevel, typeof riskColors['low']][]).map(
            ([level, colors]) => (
              <div
                key={level}
                className={clsx('flex items-center gap-2 p-3 rounded-lg', colors.bg)}
              >
                <CheckCircle2
                  className="w-4 h-4 flex-shrink-0"
                  style={{ color: colors.stroke }}
                />
                <div>
                  <p className={clsx('text-sm font-semibold', colors.text)}>
                    {colors.label}
                  </p>
                  <p className="text-xs text-text-muted">
                    {level === 'low' && '0-25'}
                    {level === 'moderate' && '26-50'}
                    {level === 'high' && '51-75'}
                    {level === 'critical' && '76-100'}
                  </p>
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
