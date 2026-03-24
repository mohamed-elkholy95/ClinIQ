import { useState, useMemo } from 'react';
import { Search, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { clsx } from 'clsx';
import { ConfidenceBar } from '../components/ConfidenceBar';
import type { ICDPrediction } from '../types';

// Demo ICD predictions grouped by chapter
const demoPredictions: ICDPrediction[] = [
  { code: 'E11.65', description: 'Type 2 diabetes mellitus with hyperglycemia', confidence: 0.94, chapter: 'Endocrine, nutritional and metabolic diseases', category: 'Diabetes mellitus', evidence: ['type 2 diabetes mellitus', 'HbA1c 7.8%', 'metformin therapy'] },
  { code: 'E11.22', description: 'Type 2 diabetes mellitus with diabetic chronic kidney disease', confidence: 0.72, chapter: 'Endocrine, nutritional and metabolic diseases', category: 'Diabetes mellitus', evidence: ['creatinine 1.2', 'diabetes'] },
  { code: 'I10', description: 'Essential (primary) hypertension', confidence: 0.92, chapter: 'Diseases of the circulatory system', category: 'Hypertensive diseases', evidence: ['hypertension', 'blood pressure 150/95', 'lisinopril'] },
  { code: 'I25.10', description: 'Atherosclerotic heart disease of native coronary artery without angina pectoris', confidence: 0.85, chapter: 'Diseases of the circulatory system', category: 'Ischemic heart diseases', evidence: ['coronary artery disease', 'atorvastatin'] },
  { code: 'I50.9', description: 'Heart failure, unspecified', confidence: 0.78, chapter: 'Diseases of the circulatory system', category: 'Heart failure', evidence: ['heart failure', 'BNP 450', 'dyspnea', 'edema'] },
  { code: 'R53.83', description: 'Other fatigue', confidence: 0.78, chapter: 'Symptoms, signs and abnormal findings', category: 'General symptoms and signs', evidence: ['persistent fatigue'] },
  { code: 'R07.9', description: 'Chest pain, unspecified', confidence: 0.75, chapter: 'Symptoms, signs and abnormal findings', category: 'Symptoms involving the circulatory and respiratory systems', evidence: ['occasional chest pain'] },
  { code: 'R06.00', description: 'Dyspnea, unspecified', confidence: 0.71, chapter: 'Symptoms, signs and abnormal findings', category: 'Symptoms involving the circulatory and respiratory systems', evidence: ['dyspnea'] },
  { code: 'R60.0', description: 'Localized edema', confidence: 0.65, chapter: 'Symptoms, signs and abnormal findings', category: 'General symptoms and signs', evidence: ['edema'] },
  { code: 'Z79.84', description: 'Long term (current) use of oral hypoglycemic drugs', confidence: 0.88, chapter: 'Factors influencing health status', category: 'Status codes', evidence: ['metformin 1000mg', 'empagliflozin'] },
  { code: 'Z87.39', description: 'Other diseases of the musculoskeletal system and connective tissue', confidence: 0.45, chapter: 'Factors influencing health status', category: 'Personal history', evidence: [] },
];

export function ICDResults() {
  const [search, setSearch] = useState('');
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set());
  const [minConfidence, setMinConfidence] = useState(0);

  const filtered = useMemo(() => {
    return demoPredictions.filter((p) => {
      const matchesSearch =
        !search ||
        p.code.toLowerCase().includes(search.toLowerCase()) ||
        p.description.toLowerCase().includes(search.toLowerCase());
      const matchesConfidence = p.confidence >= minConfidence;
      return matchesSearch && matchesConfidence;
    });
  }, [search, minConfidence]);

  const grouped = useMemo(() => {
    const groups = new Map<string, ICDPrediction[]>();
    for (const pred of filtered) {
      const existing = groups.get(pred.chapter) || [];
      existing.push(pred);
      groups.set(pred.chapter, existing);
    }
    return groups;
  }, [filtered]);

  const toggleChapter = (chapter: string) => {
    setExpandedChapters((prev) => {
      const next = new Set(prev);
      if (next.has(chapter)) {
        next.delete(chapter);
      } else {
        next.add(chapter);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpandedChapters(new Set(grouped.keys()));
  };

  const collapseAll = () => {
    setExpandedChapters(new Set());
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">ICD-10 Predictions</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Predicted ICD-10 codes with confidence scores grouped by chapter.
        </p>
      </div>

      {/* Filters */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by code or description..."
              className={clsx(
                'w-full pl-9 pr-4 py-2 rounded-lg border text-sm',
                'bg-surface-dim border-border text-text-primary placeholder:text-text-muted',
                'focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500'
              )}
            />
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-text-secondary whitespace-nowrap">
              Min confidence:
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono text-text-primary w-10 text-right">
              {Math.round(minConfidence * 100)}%
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3 mt-3">
          <span className="text-xs text-text-muted">
            {filtered.length} predictions in {grouped.size} chapters
          </span>
          <div className="flex-1" />
          <button
            onClick={expandAll}
            className="text-xs font-medium text-primary-500 hover:text-primary-600"
          >
            Expand all
          </button>
          <button
            onClick={collapseAll}
            className="text-xs font-medium text-primary-500 hover:text-primary-600"
          >
            Collapse all
          </button>
        </div>
      </div>

      {/* Grouped results */}
      <div className="space-y-3">
        {Array.from(grouped.entries()).map(([chapter, predictions]) => {
          const isExpanded = expandedChapters.has(chapter);
          const avgConfidence =
            predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;

          return (
            <div
              key={chapter}
              className="rounded-xl border border-border bg-surface overflow-hidden"
            >
              {/* Chapter header */}
              <button
                onClick={() => toggleChapter(chapter)}
                className="w-full flex items-center gap-3 p-4 hover:bg-surface-dim transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0" />
                )}
                <div className="flex-1 text-left">
                  <h3 className="text-sm font-semibold text-text-primary">
                    {chapter}
                  </h3>
                  <p className="text-xs text-text-muted mt-0.5">
                    {predictions.length} code{predictions.length !== 1 ? 's' : ''} &middot;
                    Avg confidence: {Math.round(avgConfidence * 100)}%
                  </p>
                </div>
                <div className="w-20">
                  <ConfidenceBar value={avgConfidence} size="sm" showPercentage={false} />
                </div>
              </button>

              {/* Predictions */}
              {isExpanded && (
                <div className="border-t border-border divide-y divide-border">
                  {predictions.map((pred) => (
                    <div key={pred.code} className="p-4 pl-11">
                      <div className="flex items-start gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-mono font-bold text-primary-500">
                              {pred.code}
                            </span>
                            <a
                              href={`https://icd.who.int/browse10/2019/en#/${pred.code}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-text-muted hover:text-primary-500"
                            >
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          </div>
                          <p className="text-sm text-text-primary">
                            {pred.description}
                          </p>
                          <p className="text-xs text-text-muted mt-1">
                            Category: {pred.category}
                          </p>
                          {pred.evidence.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {pred.evidence.map((ev, i) => (
                                <span
                                  key={i}
                                  className="inline-block px-2 py-0.5 text-xs rounded bg-surface-dim text-text-secondary"
                                >
                                  {ev}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        <div className="w-32 flex-shrink-0">
                          <ConfidenceBar value={pred.confidence} size="sm" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
