import { useState, useMemo } from 'react';
import { Search, Filter, X } from 'lucide-react';
import { clsx } from 'clsx';
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
import { EntityTag, EntityTypeBadge, entityLabels } from '../components/EntityTag';
import { ConfidenceBar } from '../components/ConfidenceBar';
import type { Entity, EntityType } from '../types';

// Demo entities
const demoEntities: Entity[] = [
  { id: '1', text: 'type 2 diabetes mellitus', type: 'disease', start: 0, end: 24, confidence: 0.97, cui: 'C0011860' },
  { id: '2', text: 'hypertension', type: 'disease', start: 29, end: 41, confidence: 0.95, cui: 'C0020538' },
  { id: '3', text: 'coronary artery disease', type: 'disease', start: 0, end: 23, confidence: 0.93 },
  { id: '4', text: 'heart failure', type: 'disease', start: 0, end: 13, confidence: 0.91 },
  { id: '5', text: 'metformin', type: 'medication', start: 0, end: 9, confidence: 0.99, cui: 'C0025598' },
  { id: '6', text: 'lisinopril', type: 'medication', start: 0, end: 10, confidence: 0.98 },
  { id: '7', text: 'empagliflozin', type: 'medication', start: 0, end: 13, confidence: 0.96 },
  { id: '8', text: 'aspirin', type: 'medication', start: 0, end: 7, confidence: 0.97 },
  { id: '9', text: 'atorvastatin', type: 'medication', start: 0, end: 12, confidence: 0.95 },
  { id: '10', text: 'echocardiogram', type: 'procedure', start: 0, end: 14, confidence: 0.91 },
  { id: '11', text: 'cardiac catheterization', type: 'procedure', start: 0, end: 23, confidence: 0.88 },
  { id: '12', text: 'coronary artery', type: 'anatomy', start: 0, end: 15, confidence: 0.94 },
  { id: '13', text: 'left ventricle', type: 'anatomy', start: 0, end: 14, confidence: 0.92 },
  { id: '14', text: 'myocardium', type: 'anatomy', start: 0, end: 10, confidence: 0.89 },
  { id: '15', text: 'fatigue', type: 'symptom', start: 0, end: 7, confidence: 0.89 },
  { id: '16', text: 'chest pain', type: 'symptom', start: 0, end: 10, confidence: 0.92 },
  { id: '17', text: 'dyspnea', type: 'symptom', start: 0, end: 7, confidence: 0.87 },
  { id: '18', text: 'edema', type: 'symptom', start: 0, end: 5, confidence: 0.84 },
  { id: '19', text: 'HbA1c 7.8%', type: 'lab_value', start: 0, end: 10, confidence: 0.96 },
  { id: '20', text: 'blood pressure 150/95', type: 'lab_value', start: 0, end: 21, confidence: 0.94 },
  { id: '21', text: 'creatinine 1.2', type: 'lab_value', start: 0, end: 14, confidence: 0.91 },
  { id: '22', text: 'BNP 450', type: 'lab_value', start: 0, end: 7, confidence: 0.88 },
];

const entityTypes: EntityType[] = ['disease', 'medication', 'procedure', 'anatomy', 'symptom', 'lab_value'];

const barColors: Record<EntityType, string> = {
  disease: '#ef4444',
  medication: '#3b82f6',
  procedure: '#22c55e',
  anatomy: '#a855f7',
  symptom: '#f59e0b',
  lab_value: '#06b6d4',
};

export function EntityViewer() {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<EntityType | 'all'>('all');
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);

  const filtered = useMemo(() => {
    return demoEntities.filter((e) => {
      const matchesType = typeFilter === 'all' || e.type === typeFilter;
      const matchesSearch = !search || e.text.toLowerCase().includes(search.toLowerCase());
      return matchesType && matchesSearch;
    });
  }, [search, typeFilter]);

  const frequencyData = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const e of demoEntities) {
      counts[e.type] = (counts[e.type] || 0) + 1;
    }
    return entityTypes.map((type) => ({
      type,
      label: entityLabels[type],
      count: counts[type] || 0,
    }));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Entity Viewer</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Browse and filter extracted clinical entities across all documents.
        </p>
      </div>

      {/* Frequency chart */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-base font-semibold text-text-primary mb-4">
          Entity Frequency by Type
        </h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={frequencyData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} className="text-text-muted" />
              <YAxis tick={{ fontSize: 11 }} className="text-text-muted" width={30} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--color-surface)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '8px',
                  fontSize: '13px',
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Count">
                {frequencyData.map((entry) => (
                  <Cell key={entry.type} fill={barColors[entry.type as EntityType]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search entities..."
            className={clsx(
              'w-full pl-9 pr-8 py-2.5 rounded-lg border text-sm',
              'bg-surface border-border text-text-primary placeholder:text-text-muted',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500'
            )}
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-text-muted hover:text-text-primary"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-text-muted" />
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as EntityType | 'all')}
            className={clsx(
              'px-3 py-2.5 rounded-lg border text-sm',
              'bg-surface border-border text-text-primary',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500'
            )}
          >
            <option value="all">All Types</option>
            {entityTypes.map((type) => (
              <option key={type} value={type}>
                {entityLabels[type]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results count */}
      <p className="text-sm text-text-muted">
        Showing {filtered.length} of {demoEntities.length} entities
      </p>

      {/* Main content: entity list + detail panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Entity list */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-surface overflow-hidden">
          <div className="divide-y divide-border">
            {filtered.map((entity) => (
              <button
                key={entity.id}
                onClick={() => setSelectedEntity(entity)}
                className={clsx(
                  'w-full flex items-center gap-4 p-4 text-left transition-colors',
                  'hover:bg-surface-dim',
                  selectedEntity?.id === entity.id && 'bg-primary-50 dark:bg-primary-900/20'
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-text-primary truncate">
                      {entity.text}
                    </span>
                    <EntityTypeBadge type={entity.type} />
                  </div>
                  <ConfidenceBar value={entity.confidence} size="sm" />
                </div>
                {entity.cui && (
                  <span className="text-xs font-mono text-text-muted">
                    {entity.cui}
                  </span>
                )}
              </button>
            ))}
            {filtered.length === 0 && (
              <div className="p-8 text-center text-sm text-text-muted">
                No entities match your search criteria.
              </div>
            )}
          </div>
        </div>

        {/* Detail panel */}
        <div className="rounded-xl border border-border bg-surface p-6">
          <h3 className="text-base font-semibold text-text-primary mb-4">
            Entity Details
          </h3>
          {selectedEntity ? (
            <div className="space-y-4">
              <div>
                <EntityTag
                  type={selectedEntity.type}
                  text={selectedEntity.text}
                  confidence={selectedEntity.confidence}
                />
              </div>

              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-text-muted uppercase tracking-wider">
                    Type
                  </p>
                  <p className="text-sm text-text-primary mt-0.5 capitalize">
                    {selectedEntity.type.replace('_', ' ')}
                  </p>
                </div>

                <div>
                  <p className="text-xs font-medium text-text-muted uppercase tracking-wider">
                    Confidence
                  </p>
                  <div className="mt-1">
                    <ConfidenceBar value={selectedEntity.confidence} size="md" />
                  </div>
                </div>

                {selectedEntity.cui && (
                  <div>
                    <p className="text-xs font-medium text-text-muted uppercase tracking-wider">
                      UMLS CUI
                    </p>
                    <p className="text-sm font-mono text-primary-500 mt-0.5">
                      {selectedEntity.cui}
                    </p>
                  </div>
                )}

                <div>
                  <p className="text-xs font-medium text-text-muted uppercase tracking-wider">
                    Text Span
                  </p>
                  <p className="text-sm text-text-secondary mt-0.5">
                    Characters {selectedEntity.start}-{selectedEntity.end}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-text-muted">
              Select an entity from the list to view details.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
