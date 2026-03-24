import { useState } from 'react';
import { Calendar, ChevronDown, ChevronUp } from 'lucide-react';
import { clsx } from 'clsx';
import { EntityTag, entityLabels } from '../components/EntityTag';
import type { TimelineEvent, EntityType } from '../types';

const demoTimeline: TimelineEvent[] = [
  {
    id: 't1',
    date: '2025-06-15',
    type: 'disease',
    description: 'Initial diagnosis of type 2 diabetes mellitus. Fasting glucose 186 mg/dL, HbA1c 8.2%. Started on metformin 500mg BID.',
    entities: [
      { id: 'e1', text: 'type 2 diabetes mellitus', type: 'disease', start: 22, end: 47, confidence: 0.97 },
      { id: 'e2', text: 'metformin', type: 'medication', start: 0, end: 9, confidence: 0.99 },
    ],
    source_text: 'Discharge Summary',
  },
  {
    id: 't2',
    date: '2025-09-10',
    type: 'medication',
    description: 'Metformin uptitrated to 1000mg BID. HbA1c improved to 7.5%. Blood pressure noted elevated at 142/88 mmHg.',
    entities: [
      { id: 'e3', text: 'metformin', type: 'medication', start: 0, end: 9, confidence: 0.99 },
      { id: 'e4', text: 'HbA1c', type: 'lab_value', start: 0, end: 5, confidence: 0.96 },
    ],
    source_text: 'Follow-up Note',
  },
  {
    id: 't3',
    date: '2025-11-22',
    type: 'disease',
    description: 'Diagnosis of essential hypertension. Blood pressure 148/92 mmHg on repeat measurement. Started on lisinopril 10mg daily.',
    entities: [
      { id: 'e5', text: 'essential hypertension', type: 'disease', start: 13, end: 35, confidence: 0.95 },
      { id: 'e6', text: 'lisinopril', type: 'medication', start: 0, end: 10, confidence: 0.98 },
    ],
    source_text: 'Clinic Visit Note',
  },
  {
    id: 't4',
    date: '2026-01-14',
    type: 'lab_value',
    description: 'Routine labs: HbA1c 7.8% (worsening trend), creatinine 1.1, BUN 18, eGFR 72. Lipid panel shows LDL 142.',
    entities: [
      { id: 'e7', text: 'HbA1c 7.8%', type: 'lab_value', start: 0, end: 10, confidence: 0.96 },
      { id: 'e8', text: 'creatinine 1.1', type: 'lab_value', start: 0, end: 14, confidence: 0.93 },
      { id: 'e9', text: 'eGFR 72', type: 'lab_value', start: 0, end: 7, confidence: 0.91 },
    ],
    source_text: 'Lab Report',
  },
  {
    id: 't5',
    date: '2026-02-28',
    type: 'symptom',
    description: 'Patient reports new persistent fatigue for past 3 weeks and occasional chest discomfort with exertion. Blood pressure 150/95 mmHg.',
    entities: [
      { id: 'e10', text: 'fatigue', type: 'symptom', start: 0, end: 7, confidence: 0.89 },
      { id: 'e11', text: 'chest discomfort', type: 'symptom', start: 0, end: 16, confidence: 0.87 },
    ],
    source_text: 'Clinic Visit Note',
  },
  {
    id: 't6',
    date: '2026-03-15',
    type: 'procedure',
    description: 'Echocardiogram performed: mild left ventricular hypertrophy, EF 55%, no significant valvular disease. Cardiac stress test pending.',
    entities: [
      { id: 'e12', text: 'echocardiogram', type: 'procedure', start: 0, end: 14, confidence: 0.95 },
      { id: 'e13', text: 'left ventricular hypertrophy', type: 'anatomy', start: 0, end: 28, confidence: 0.91 },
    ],
    source_text: 'Radiology Report',
  },
  {
    id: 't7',
    date: '2026-03-23',
    type: 'medication',
    description: 'Comprehensive review. Adding empagliflozin 10mg for cardiovascular protection. Plan to uptitrate lisinopril to 20mg. Referral to cardiology.',
    entities: [
      { id: 'e14', text: 'empagliflozin', type: 'medication', start: 0, end: 13, confidence: 0.96 },
      { id: 'e15', text: 'lisinopril', type: 'medication', start: 0, end: 10, confidence: 0.98 },
    ],
    source_text: 'Progress Note',
  },
];

const typeColors: Record<EntityType, string> = {
  disease: 'border-red-400 bg-red-50 dark:bg-red-950/30',
  medication: 'border-blue-400 bg-blue-50 dark:bg-blue-950/30',
  procedure: 'border-green-400 bg-green-50 dark:bg-green-950/30',
  anatomy: 'border-purple-400 bg-purple-50 dark:bg-purple-950/30',
  symptom: 'border-amber-400 bg-amber-50 dark:bg-amber-950/30',
  lab_value: 'border-cyan-400 bg-cyan-50 dark:bg-cyan-950/30',
};

const dotColors: Record<EntityType, string> = {
  disease: 'bg-red-500',
  medication: 'bg-blue-500',
  procedure: 'bg-green-500',
  anatomy: 'bg-purple-500',
  symptom: 'bg-amber-500',
  lab_value: 'bg-cyan-500',
};

export function Timeline() {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [typeFilter, setTypeFilter] = useState<EntityType | 'all'>('all');

  const filtered = typeFilter === 'all'
    ? demoTimeline
    : demoTimeline.filter((event) => event.type === typeFilter);

  const toggleItem = (id: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Patient Timeline</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Chronological view of clinical events extracted from patient documents.
        </p>
      </div>

      {/* Filter */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-text-secondary mr-1">Filter by type:</span>
        <button
          onClick={() => setTypeFilter('all')}
          className={clsx(
            'px-3 py-1.5 rounded-full text-xs font-medium border transition-colors',
            typeFilter === 'all'
              ? 'bg-primary-500 text-white border-primary-500'
              : 'bg-surface border-border text-text-secondary hover:border-primary-300'
          )}
        >
          All
        </button>
        {(Object.keys(entityLabels) as EntityType[]).map((type) => (
          <button
            key={type}
            onClick={() => setTypeFilter(type)}
            className={clsx(
              'px-3 py-1.5 rounded-full text-xs font-medium border transition-colors',
              typeFilter === type
                ? 'bg-primary-500 text-white border-primary-500'
                : 'bg-surface border-border text-text-secondary hover:border-primary-300'
            )}
          >
            {entityLabels[type]}
          </button>
        ))}
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-[23px] top-0 bottom-0 w-0.5 bg-border" />

        <div className="space-y-4">
          {filtered.map((event) => {
            const isExpanded = expandedItems.has(event.id);

            return (
              <div key={event.id} className="relative flex gap-4">
                {/* Dot */}
                <div className="relative z-10 flex-shrink-0 mt-5">
                  <div
                    className={clsx(
                      'w-[12px] h-[12px] rounded-full border-2 border-surface',
                      dotColors[event.type]
                    )}
                    style={{ marginLeft: '12px' }}
                  />
                </div>

                {/* Card */}
                <div
                  className={clsx(
                    'flex-1 rounded-xl border-l-4 bg-surface border border-border overflow-hidden',
                    typeColors[event.type]
                  )}
                >
                  <button
                    onClick={() => toggleItem(event.id)}
                    className="w-full p-4 text-left hover:bg-surface-dim/50 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Calendar className="w-3.5 h-3.5 text-text-muted" />
                          <span className="text-xs font-semibold text-text-muted">
                            {new Date(event.date).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'long',
                              day: 'numeric',
                            })}
                          </span>
                          <span className="text-xs text-text-muted">
                            &middot; {event.source_text}
                          </span>
                        </div>
                        <p className="text-sm text-text-primary leading-relaxed">
                          {event.description}
                        </p>
                      </div>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0 mt-1" />
                      )}
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="px-4 pb-4 border-t border-border pt-3">
                      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
                        Extracted Entities
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {event.entities.map((entity) => (
                          <EntityTag
                            key={entity.id}
                            type={entity.type}
                            text={entity.text}
                            confidence={entity.confidence}
                            size="sm"
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-12 text-sm text-text-muted">
            No timeline events match the selected filter.
          </div>
        )}
      </div>
    </div>
  );
}
