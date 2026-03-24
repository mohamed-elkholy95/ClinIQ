import { useState } from 'react';
import {
  FileText,
  Tags,
  ShieldAlert,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
} from 'lucide-react';
import { clsx } from 'clsx';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { DashboardStats, EntityType } from '../types';

// Demo data for initial rendering (replaced by API in production)
const demoStats: DashboardStats = {
  total_documents: 1247,
  total_entities: 18432,
  avg_risk_score: 42,
  documents_today: 23,
  processing_volume: [
    { date: '2026-03-10', count: 45 },
    { date: '2026-03-11', count: 52 },
    { date: '2026-03-12', count: 38 },
    { date: '2026-03-13', count: 67 },
    { date: '2026-03-14', count: 55 },
    { date: '2026-03-15', count: 44 },
    { date: '2026-03-16', count: 71 },
    { date: '2026-03-17', count: 63 },
    { date: '2026-03-18', count: 58 },
    { date: '2026-03-19', count: 82 },
    { date: '2026-03-20', count: 76 },
    { date: '2026-03-21', count: 91 },
    { date: '2026-03-22', count: 85 },
    { date: '2026-03-23', count: 79 },
  ],
  entity_distribution: [
    { type: 'disease' as EntityType, count: 4521 },
    { type: 'medication' as EntityType, count: 5234 },
    { type: 'procedure' as EntityType, count: 2187 },
    { type: 'anatomy' as EntityType, count: 3102 },
    { type: 'symptom' as EntityType, count: 2456 },
    { type: 'lab_value' as EntityType, count: 932 },
  ],
  recent_activity: [
    { id: '1', action: 'Document analyzed', document_title: 'Discharge Summary - Patient 4821', timestamp: '2026-03-23T14:32:00Z' },
    { id: '2', action: 'Batch completed', document_title: 'Radiology Reports Batch #42', timestamp: '2026-03-23T13:15:00Z' },
    { id: '3', action: 'Risk alert', document_title: 'Admission Note - Patient 1293', timestamp: '2026-03-23T12:48:00Z' },
    { id: '4', action: 'Document analyzed', document_title: 'Lab Results - Patient 7102', timestamp: '2026-03-23T11:22:00Z' },
    { id: '5', action: 'Model updated', document_title: 'NER Model v2.3.1', timestamp: '2026-03-23T10:05:00Z' },
  ],
};

const statCards = [
  {
    label: 'Documents Processed',
    value: demoStats.total_documents.toLocaleString(),
    change: '+12%',
    trend: 'up' as const,
    icon: FileText,
    color: 'text-blue-600 dark:text-blue-400',
    bg: 'bg-blue-50 dark:bg-blue-950/40',
  },
  {
    label: 'Entities Found',
    value: demoStats.total_entities.toLocaleString(),
    change: '+8%',
    trend: 'up' as const,
    icon: Tags,
    color: 'text-purple-600 dark:text-purple-400',
    bg: 'bg-purple-50 dark:bg-purple-950/40',
  },
  {
    label: 'Avg Risk Score',
    value: demoStats.avg_risk_score.toString(),
    change: '-3%',
    trend: 'down' as const,
    icon: ShieldAlert,
    color: 'text-amber-600 dark:text-amber-400',
    bg: 'bg-amber-50 dark:bg-amber-950/40',
  },
  {
    label: 'Today\'s Documents',
    value: demoStats.documents_today.toString(),
    change: '+18%',
    trend: 'up' as const,
    icon: TrendingUp,
    color: 'text-green-600 dark:text-green-400',
    bg: 'bg-green-50 dark:bg-green-950/40',
  },
];

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString();
}

export function Dashboard() {
  const [_stats] = useState(demoStats);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Dashboard</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Overview of clinical document processing and analysis metrics.
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {statCards.map((card) => (
          <div
            key={card.label}
            className="card-hover rounded-xl border border-border bg-surface p-5"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-text-secondary">
                {card.label}
              </span>
              <div className={clsx('p-2 rounded-lg', card.bg)}>
                <card.icon className={clsx('w-4 h-4', card.color)} />
              </div>
            </div>
            <div className="mt-3">
              <span className="text-2xl font-bold text-text-primary">
                {card.value}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1">
              {card.trend === 'up' ? (
                <ArrowUpRight className="w-3.5 h-3.5 text-green-500" />
              ) : (
                <ArrowDownRight className="w-3.5 h-3.5 text-red-500" />
              )}
              <span
                className={clsx(
                  'text-xs font-medium',
                  card.trend === 'up' ? 'text-green-600' : 'text-red-600'
                )}
              >
                {card.change}
              </span>
              <span className="text-xs text-text-muted ml-1">vs last week</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts and activity row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Processing volume chart */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-surface p-6">
          <h3 className="text-base font-semibold text-text-primary mb-4">
            Processing Volume
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={demoStats.processing_volume}
                margin={{ top: 5, right: 5, bottom: 5, left: 0 }}
              >
                <defs>
                  <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0f4c81" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#0f4c81" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  className="text-text-muted"
                  tickFormatter={(val: string) => {
                    const d = new Date(val);
                    return `${d.getMonth() + 1}/${d.getDate()}`;
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  className="text-text-muted"
                  width={35}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '8px',
                    fontSize: '13px',
                  }}
                  labelFormatter={(val: string) =>
                    new Date(val).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })
                  }
                />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="#0f4c81"
                  strokeWidth={2}
                  fill="url(#volumeGradient)"
                  name="Documents"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent activity */}
        <div className="rounded-xl border border-border bg-surface p-6">
          <h3 className="text-base font-semibold text-text-primary mb-4">
            Recent Activity
          </h3>
          <div className="space-y-4">
            {demoStats.recent_activity.map((item) => (
              <div key={item.id} className="flex items-start gap-3">
                <div className="mt-0.5 p-1.5 rounded-lg bg-primary-50 dark:bg-primary-900/30">
                  <Clock className="w-3.5 h-3.5 text-primary-500" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-text-primary truncate">
                    {item.action}
                  </p>
                  <p className="text-xs text-text-muted truncate">
                    {item.document_title}
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">
                    {formatTime(item.timestamp)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Entity distribution */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-base font-semibold text-text-primary mb-4">
          Entity Distribution
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {demoStats.entity_distribution.map((item) => {
            const colors: Record<string, string> = {
              disease: 'bg-red-500',
              medication: 'bg-blue-500',
              procedure: 'bg-green-500',
              anatomy: 'bg-purple-500',
              symptom: 'bg-amber-500',
              lab_value: 'bg-cyan-500',
            };
            return (
              <div
                key={item.type}
                className="text-center p-4 rounded-lg bg-surface-dim"
              >
                <div
                  className={clsx(
                    'mx-auto w-3 h-3 rounded-full mb-2',
                    colors[item.type]
                  )}
                />
                <p className="text-lg font-bold text-text-primary">
                  {item.count.toLocaleString()}
                </p>
                <p className="text-xs text-text-muted capitalize mt-0.5">
                  {item.type.replace('_', ' ')}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
