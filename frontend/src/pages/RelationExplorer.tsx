/**
 * RelationExplorer page — Extract clinical entity relations from free text.
 *
 * Detects 12 relation types (treats, causes, diagnoses, contraindicates,
 * administered_for, dosage_of, location_of, result_of, worsens, prevents,
 * monitors, side_effect_of) between clinical entities.  Results are shown
 * as a relation table with directional arrows and confidence bars.
 */

import { useState, useCallback } from 'react';
import type { RelationResult, RelationExtractionResponse } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Treatment Plan',
    text: `ASSESSMENT AND PLAN:

1. Hypertension — treated with lisinopril 20 mg daily.  BP goal <130/80.
   Metoprolol 50 mg added for rate control.
   Amlodipine 5 mg causes ankle edema — switch to losartan if persistent.

2. Type 2 Diabetes — controlled on metformin 1000 mg BID.
   HbA1c monitors glycemic control, last value 7.2%.
   Metformin causes GI upset — take with food.
   If HbA1c > 8%, will start glipizide.

3. Hyperlipidemia — atorvastatin 40 mg prevents cardiovascular events.
   LDL monitors treatment response, goal < 70.
   Statins contraindicated with gemfibrozil.

4. Chronic kidney disease stage 3 — creatinine result of 1.8 mg/dL.
   ACE inhibitors prevent renal progression.
   NSAIDs worsen kidney function — discontinue ibuprofen.`,
  },
  {
    label: 'Admission Note',
    text: `ADMISSION NOTE:
Diagnosis: Community-acquired pneumonia, right lower lobe.
Pneumonia located in the right lower lobe on chest X-ray.

Treatment initiated:
- Ceftriaxone 1g IV treats bacterial pneumonia.
- Azithromycin 500 mg administered for atypical coverage.
- Albuterol nebuliser treats bronchospasm.
- Dexamethasone prevents inflammatory damage.

Lab results:
- WBC 15,000 — result of infection.
- Procalcitonin monitors bacterial infection severity.
- CRP result of systemic inflammation.
- Blood cultures pending to diagnose bacteremia.

Complications:
- Ceftriaxone causes Clostridioides difficile — monitor stool output.
- Azithromycin causes QT prolongation — obtain baseline ECG.
- Supplemental oxygen prevents hypoxic organ damage.`,
  },
  {
    label: 'Dental Note',
    text: `DENTAL TREATMENT NOTE:
Diagnosis: Irreversible pulpitis tooth #19 with periapical abscess.
Abscess located at the apex of tooth #19.

Treatment:
- Amoxicillin 500 mg TID treats periapical abscess.
- Ibuprofen 600 mg administered for pain management.
- Root canal therapy treats irreversible pulpitis.
- Chlorhexidine rinse prevents post-operative infection.

Monitoring:
- Periapical radiograph monitors healing after root canal.
- Percussion test diagnoses continued inflammation.
- Vitality test result of pulp necrosis.

Cautions:
- Amoxicillin causes allergic reactions in penicillin-sensitive patients.
- Opioids worsen constipation — prescribe stool softener if needed.
- Clindamycin dosage of 300 mg QID for penicillin-allergic patients.`,
  },
];

// ─── Relation Type Styling ───────────────────────────────────

const RELATION_STYLES: Record<string, { color: string; arrow: string; label: string }> = {
  treats: {
    color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    arrow: '→ treats →',
    label: 'Treats',
  },
  causes: {
    color: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
    arrow: '→ causes →',
    label: 'Causes',
  },
  diagnoses: {
    color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    arrow: '→ diagnoses →',
    label: 'Diagnoses',
  },
  contraindicates: {
    color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
    arrow: '→ contraindicates →',
    label: 'Contraindicates',
  },
  administered_for: {
    color: 'bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300',
    arrow: '→ administered for →',
    label: 'Administered For',
  },
  dosage_of: {
    color: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300',
    arrow: '→ dosage of →',
    label: 'Dosage Of',
  },
  location_of: {
    color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300',
    arrow: '→ located in →',
    label: 'Location Of',
  },
  result_of: {
    color: 'bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300',
    arrow: '→ result of →',
    label: 'Result Of',
  },
  worsens: {
    color: 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300',
    arrow: '→ worsens →',
    label: 'Worsens',
  },
  prevents: {
    color: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300',
    arrow: '→ prevents →',
    label: 'Prevents',
  },
  monitors: {
    color: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    arrow: '→ monitors →',
    label: 'Monitors',
  },
  side_effect_of: {
    color: 'bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300',
    arrow: '→ side effect of →',
    label: 'Side Effect Of',
  },
};

// ─── Component ───────────────────────────────────────────────

export function RelationExplorer() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<RelationExtractionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { extractRelations } = await import('../services/clinical');
      const data = await extractRelations(text);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [text]);

  const loadSample = (idx: number) => {
    setText(SAMPLE_NOTES[idx].text);
    setResults(null);
    setError(null);
    setActiveFilter(null);
  };

  // Compute relation type counts
  const typeCounts: Record<string, number> = {};
  if (results) {
    for (const rel of results.relations) {
      typeCounts[rel.relation_type] = (typeCounts[rel.relation_type] || 0) + 1;
    }
  }

  const activeTypes = Object.keys(typeCounts);

  const filteredRelations = results
    ? activeFilter
      ? results.relations.filter((r) => r.relation_type === activeFilter)
      : results.relations
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Relation Explorer
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Extract clinical entity relations — treatments, causes, diagnoses,
          contraindications, and more from clinical notes.
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-3">
          <label
            htmlFor="relation-input"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Clinical Text
          </label>
          <span className="text-xs text-gray-400">{wordCount} words</span>
        </div>

        <textarea
          id="relation-input"
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a clinical note to extract entity relations..."
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-3 text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />

        {/* Sample Notes */}
        <div className="mt-3 flex flex-wrap gap-2">
          {SAMPLE_NOTES.map((sample, idx) => (
            <button
              key={sample.label}
              onClick={() => loadSample(idx)}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {sample.label}
            </button>
          ))}
        </div>

        {/* Analyze Button */}
        <div className="mt-4">
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Extracting…' : 'Extract Relations'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-blue-600">{results.count}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Total Relations</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-purple-600">{activeTypes.length}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Relation Types</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-green-600">{typeCounts['treats'] || 0}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Treatments</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-gray-600 dark:text-gray-300">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Processing Time</p>
            </div>
          </div>

          {/* Type Filter Bar */}
          {activeTypes.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setActiveFilter(null)}
                className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                  activeFilter === null
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200'
                }`}
              >
                All ({results.count})
              </button>
              {activeTypes.map((type) => {
                const style = RELATION_STYLES[type];
                return (
                  <button
                    key={type}
                    onClick={() => setActiveFilter(activeFilter === type ? null : type)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                      activeFilter === type
                        ? 'bg-blue-600 text-white'
                        : style?.color || 'bg-gray-100 text-gray-700'
                    }`}
                  >
                    {style?.label || type} ({typeCounts[type]})
                  </button>
                );
              })}
            </div>
          )}

          {/* Relation Cards */}
          {filteredRelations.length > 0 ? (
            <div className="space-y-3">
              {filteredRelations.map((rel, idx) => {
                const style = RELATION_STYLES[rel.relation_type] || {
                  color: 'bg-gray-100 text-gray-800',
                  arrow: '→',
                  label: rel.relation_type,
                };
                return (
                  <div
                    key={`${rel.subject}-${rel.object}-${rel.relation_type}-${idx}`}
                    className="bg-white dark:bg-gray-800 rounded-lg shadow p-4"
                  >
                    <div className="flex items-center gap-3 flex-wrap">
                      {/* Subject */}
                      <span className="px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 rounded-lg text-sm font-medium">
                        {rel.subject}
                      </span>

                      {/* Relation Arrow */}
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${style.color}`}
                      >
                        {style.arrow}
                      </span>

                      {/* Object */}
                      <span className="px-3 py-1.5 bg-purple-50 dark:bg-purple-900/20 text-purple-800 dark:text-purple-300 rounded-lg text-sm font-medium">
                        {rel.object}
                      </span>

                      {/* Confidence */}
                      <div className="ml-auto flex items-center gap-2">
                        <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${rel.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {(rel.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {/* Evidence */}
                    {rel.evidence && (
                      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400 italic">
                        &ldquo;{rel.evidence}&rdquo;
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
              <p className="text-gray-500 dark:text-gray-400">
                No relations found matching the current filter.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
