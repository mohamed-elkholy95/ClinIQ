/**
 * SectionParser page — Parse clinical documents into structured sections
 * with category detection and position mapping.
 *
 * Uses the rule-based section parser to detect clinical document sections
 * (History, Assessment, Plan, Medications, etc.), map their positions,
 * and classify them into clinical categories.  Sections are displayed as
 * colour-coded cards with expandable body text and confidence scores.
 */

import { useState, useCallback } from 'react';
import type { SectionParseResponse, SectionResult } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'H&P Note',
    text: `CHIEF COMPLAINT:
Chest pain and shortness of breath for 3 days.

HISTORY OF PRESENT ILLNESS:
Mr. Johnson is a 68-year-old male with a history of hypertension, type 2 diabetes,
and coronary artery disease who presents with progressive chest pain radiating to
the left arm and worsening dyspnea on exertion. He reports the pain began 3 days
ago during light activity and has been intermittent. He denies nausea, vomiting,
diaphoresis, or syncope. He has been compliant with his medications.

PAST MEDICAL HISTORY:
1. Hypertension — diagnosed 2010
2. Type 2 Diabetes — A1c 7.2% (last checked 3 months ago)
3. Coronary artery disease — s/p stent to LAD 2019
4. Hyperlipidemia
5. GERD

MEDICATIONS:
1. Metoprolol succinate 50mg daily
2. Lisinopril 20mg daily
3. Metformin 1000mg BID
4. Atorvastatin 40mg at bedtime
5. Aspirin 81mg daily
6. Omeprazole 20mg daily

ALLERGIES:
Penicillin — rash

PHYSICAL EXAMINATION:
Vital Signs: BP 158/92, HR 88, RR 20, SpO2 96% on RA, T 98.6°F
General: Alert, oriented, in mild distress
HEENT: PERRLA, EOMI, no JVD
Cardiac: Regular rate and rhythm, no murmurs, rubs or gallops
Pulmonary: Bilateral crackles at bases, no wheezes
Abdomen: Soft, non-tender, non-distended
Extremities: 1+ bilateral pedal edema

ASSESSMENT AND PLAN:
1. Acute coronary syndrome — rule out NSTEMI
   - Serial troponins q6h, EKG now and in 6 hours
   - Heparin drip per ACS protocol
   - Continue aspirin, add clopidogrel 75mg
   - Cardiology consultation for possible catheterization

2. CHF exacerbation — likely secondary to ACS
   - IV furosemide 40mg now
   - Strict I&O, daily weights
   - Fluid restriction 1.5L/day
   - Echocardiogram in AM`,
  },
  {
    label: 'Discharge Summary',
    text: `DISCHARGE SUMMARY

DATE OF ADMISSION: 03/20/2026
DATE OF DISCHARGE: 03/25/2026

ADMITTING DIAGNOSIS: Non-ST elevation myocardial infarction

DISCHARGE DIAGNOSES:
1. NSTEMI — successfully treated with PCI to LAD
2. Congestive heart failure, NYHA Class III
3. Chronic kidney disease, Stage III (GFR 42)
4. Type 2 Diabetes, controlled

HOSPITAL COURSE:
Patient was admitted with chest pain and dyspnea. Initial troponin was 0.8 ng/mL,
peaked at 2.4. EKG showed ST depression in V4-V6. Cardiac catheterization on HD2
revealed 85% stenosis of mid-LAD; drug-eluting stent placed successfully. Post-PCI
course complicated by mild contrast nephropathy (Cr peaked at 2.1, baseline 1.6).
CHF managed with IV diuresis, transitioned to oral therapy on HD3.

DISCHARGE MEDICATIONS:
1. Aspirin 81mg daily — indefinitely
2. Clopidogrel 75mg daily — minimum 12 months post-stent
3. Metoprolol succinate 100mg daily (increased from 50mg)
4. Atorvastatin 80mg daily (increased from 40mg)
5. Lisinopril 10mg daily (decreased due to CKD)
6. Furosemide 40mg BID (new)
7. Metformin 500mg BID (decreased due to CKD)

FOLLOW-UP:
1. Cardiology — Dr. Chen in 1 week
2. Nephrology — Dr. Patel in 2 weeks
3. PCP — Dr. Williams in 1 week
4. Cardiac rehab — referral placed

PATIENT EDUCATION:
Low-sodium diet discussed. Heart failure warning signs reviewed.
Medication compliance emphasized. Activity restrictions for 2 weeks post-PCI.`,
  },
  {
    label: 'Dental Treatment Note',
    text: `PATIENT: Maria Garcia (DOB: 05/12/1985)
DATE: 03/26/2026

CHIEF COMPLAINT:
"My lower right back tooth has been hurting for two weeks."

DENTAL HISTORY:
Last dental visit was 18 months ago. Reports irregular brushing (once daily).
No flossing. History of multiple restorations. No prior endodontic treatment.

CLINICAL EXAMINATION:
Extraoral: No swelling, no lymphadenopathy, TMJ WNL, no clicking
Intraoral: Caries noted on #30 (DO), #19 (MO). Heavy calculus LL.
#30: Positive to percussion, negative to cold, periapical radiolucency noted.
Periodontal screening: PSR scores 2-2 / 3-2 / 2-3
BOP: #30, #31, #18, #19

RADIOGRAPHIC FINDINGS:
PA #30: Periapical radiolucency approximately 3mm diameter. Caries extending to pulp.
BWX: Interproximal caries #19 MO, into dentin but not pulp.
Pano: No other significant findings. Third molars previously extracted.

DIAGNOSIS:
1. Pulpal necrosis #30 with periapical abscess
2. Caries #19 MO — moderate
3. Generalized mild to moderate gingivitis with localized periodontitis

TREATMENT PLAN:
Phase 1: Emergency
- RCT #30 (start today, complete next visit)
- Amoxicillin 500mg TID x 7 days

Phase 2: Restorative
- Core buildup and crown #30
- Composite restoration #19 MO

Phase 3: Periodontal
- Adult prophylaxis with subgingival scaling LL
- OHI reinforcement
- Re-evaluate perio in 4-6 weeks`,
  },
];

// ─── Category Colors and Icons ───────────────────────────────

const CATEGORY_STYLES: Record<string, { color: string; icon: string }> = {
  chief_complaint: { color: 'border-red-400 bg-red-50 dark:bg-red-900/20', icon: '🎯' },
  history: { color: 'border-blue-400 bg-blue-50 dark:bg-blue-900/20', icon: '📋' },
  past_medical_history: { color: 'border-indigo-400 bg-indigo-50 dark:bg-indigo-900/20', icon: '🏥' },
  medications: { color: 'border-purple-400 bg-purple-50 dark:bg-purple-900/20', icon: '💊' },
  allergies: { color: 'border-orange-400 bg-orange-50 dark:bg-orange-900/20', icon: '⚠️' },
  physical_exam: { color: 'border-green-400 bg-green-50 dark:bg-green-900/20', icon: '🩺' },
  assessment: { color: 'border-teal-400 bg-teal-50 dark:bg-teal-900/20', icon: '📊' },
  plan: { color: 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20', icon: '📝' },
  assessment_and_plan: { color: 'border-teal-400 bg-teal-50 dark:bg-teal-900/20', icon: '📊' },
  review_of_systems: { color: 'border-sky-400 bg-sky-50 dark:bg-sky-900/20', icon: '🔍' },
  vital_signs: { color: 'border-rose-400 bg-rose-50 dark:bg-rose-900/20', icon: '❤️' },
  labs: { color: 'border-amber-400 bg-amber-50 dark:bg-amber-900/20', icon: '🧪' },
  imaging: { color: 'border-cyan-400 bg-cyan-50 dark:bg-cyan-900/20', icon: '📷' },
  procedures: { color: 'border-fuchsia-400 bg-fuchsia-50 dark:bg-fuchsia-900/20', icon: '🔧' },
  discharge: { color: 'border-lime-400 bg-lime-50 dark:bg-lime-900/20', icon: '🏠' },
  follow_up: { color: 'border-violet-400 bg-violet-50 dark:bg-violet-900/20', icon: '📅' },
  dental_exam: { color: 'border-teal-400 bg-teal-50 dark:bg-teal-900/20', icon: '🦷' },
  diagnosis: { color: 'border-red-400 bg-red-50 dark:bg-red-900/20', icon: '🔬' },
  treatment_plan: { color: 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20', icon: '📋' },
  patient_education: { color: 'border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20', icon: '📚' },
  hospital_course: { color: 'border-blue-400 bg-blue-50 dark:bg-blue-900/20', icon: '🏨' },
  radiographic_findings: { color: 'border-cyan-400 bg-cyan-50 dark:bg-cyan-900/20', icon: '📷' },
  dental_history: { color: 'border-teal-400 bg-teal-50 dark:bg-teal-900/20', icon: '🦷' },
};

const DEFAULT_STYLE = { color: 'border-gray-400 bg-gray-50 dark:bg-gray-900/20', icon: '📄' };

function getCategoryStyle(category: string) {
  const key = category.toLowerCase().replace(/\s+/g, '_');
  return CATEGORY_STYLES[key] ?? DEFAULT_STYLE;
}

// ─── Section Card Component ──────────────────────────────────

function SectionCard({ section, index }: { section: SectionResult; index: number }) {
  const [expanded, setExpanded] = useState(index < 5); // Auto-expand first 5
  const style = getCategoryStyle(section.category);

  return (
    <div className={`border-l-4 rounded-lg shadow-sm overflow-hidden ${style.color}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:opacity-80 transition-opacity"
      >
        <div className="flex items-center gap-3">
          <span className="text-lg">{style.icon}</span>
          <div>
            <span className="font-semibold text-gray-900 dark:text-white text-sm">
              {section.header_text || section.category}
            </span>
            <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
              ({section.category})
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {(section.confidence * 100).toFixed(0)}%
          </span>
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>
      {expanded && section.body_text && (
        <div className="px-4 pb-4">
          <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-sans leading-relaxed bg-white/50 dark:bg-gray-800/50 rounded p-3">
            {section.body_text}
          </pre>
          <div className="mt-2 flex gap-4 text-xs text-gray-400">
            <span>Start: char {section.header_start}</span>
            <span>End: char {section.body_end}</span>
            <span>Length: {section.body_text.length} chars</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────

export function SectionParser() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<SectionParseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/sections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
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
  };

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          📑 Section Parser
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Parse clinical documents into structured sections with category detection
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
        {/* Sample Note Buttons */}
        <div className="flex flex-wrap gap-2">
          {SAMPLE_NOTES.map((note, idx) => (
            <button
              key={note.label}
              onClick={() => loadSample(idx)}
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-900/30 dark:text-indigo-300 dark:hover:bg-indigo-900/50 transition-colors"
            >
              📋 {note.label}
            </button>
          ))}
        </div>

        {/* Text Input */}
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            placeholder="Paste a clinical document to parse into sections..."
            className="w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            disabled={loading}
          />
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {wordCount} words
          </div>
        </div>

        {/* Analyze Button */}
        <button
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Parsing...
            </span>
          ) : (
            '📑 Parse Sections'
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-700 dark:text-red-300">❌ {error}</p>
          </div>
        )}
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Stats Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{results.count}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Sections Found</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {new Set(results.sections.map((s) => s.category)).size}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Unique Categories</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
              <p className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                {results.processing_time_ms.toFixed(1)}ms
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Processing Time</p>
            </div>
          </div>

          {/* Section Cards */}
          <div className="space-y-3">
            {results.sections.map((section, idx) => (
              <SectionCard key={`${section.category}-${idx}`} section={section} index={idx} />
            ))}
          </div>

          {/* Category Summary */}
          {results.sections.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                📊 Category Summary
              </h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(
                  results.sections.reduce<Record<string, number>>((acc, s) => {
                    acc[s.category] = (acc[s.category] ?? 0) + 1;
                    return acc;
                  }, {})
                )
                  .sort(([, a], [, b]) => b - a)
                  .map(([cat, count]) => {
                    const style = getCategoryStyle(cat);
                    return (
                      <span
                        key={cat}
                        className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium border ${style.color}`}
                      >
                        {style.icon} {cat} <span className="font-bold">×{count}</span>
                      </span>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
