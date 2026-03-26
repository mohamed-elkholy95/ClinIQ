/**
 * DocumentClassifier page — Classify clinical documents into 14 types.
 *
 * Uses the rule-based document classification engine to identify document
 * type (discharge summary, progress note, H&P, operative note, etc.)
 * with per-type confidence scores, evidence keywords, and a visual
 * score distribution chart.
 */

import { useState, useCallback } from 'react';
import type { ClassificationResponse, ClassificationScore } from '../types/clinical';

// ─── Sample Clinical Notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'Discharge Summary',
    text: `DISCHARGE SUMMARY

Patient Name: [REDACTED]
Date of Admission: 03/10/2025
Date of Discharge: 03/17/2025

DISCHARGE DIAGNOSIS:
1. Community-acquired pneumonia
2. Type 2 diabetes mellitus
3. Hypertension

HOSPITAL COURSE:
Patient was admitted through the ED with fever, productive cough, and dyspnea.
Chest X-ray confirmed right lower lobe consolidation.  Blood cultures were drawn
and patient was started on ceftriaxone and azithromycin.  Sputum culture grew
Streptococcus pneumoniae.  Patient improved on day 3 and was transitioned to
oral amoxicillin-clavulanate.

Diabetes was managed with sliding scale insulin during admission.  Metformin was
held due to acute illness.  HbA1c was 8.1%, suggesting suboptimal control.

DISCHARGE MEDICATIONS:
- Amoxicillin-clavulanate 875 mg BID x 5 days
- Metformin 1000 mg BID (resume)
- Lisinopril 20 mg daily
- Atorvastatin 40 mg at bedtime

DISCHARGE INSTRUCTIONS:
- Follow up with PCP in 1 week
- Repeat chest X-ray in 6 weeks
- Return to ED if fever > 101°F, worsening cough, or difficulty breathing

CONDITION AT DISCHARGE: Improved, stable.`,
  },
  {
    label: 'Operative Note',
    text: `OPERATIVE NOTE

Date of Surgery: 03/15/2025
Surgeon: [REDACTED]
Assistant: [REDACTED]
Anesthesia: General endotracheal

PRE-OPERATIVE DIAGNOSIS: Acute appendicitis
POST-OPERATIVE DIAGNOSIS: Acute appendicitis with perforation

PROCEDURE PERFORMED: Laparoscopic appendectomy

INDICATION:
22-year-old male presenting with 24 hours of RLQ pain, nausea, and fever.
CT abdomen showed dilated appendix with surrounding fat stranding.
WBC 16,000.  Decision made for surgical intervention.

DESCRIPTION OF PROCEDURE:
Patient was placed supine.  General anesthesia induced.  Abdomen prepped and
draped in sterile fashion.  Pneumoperitoneum established via Veress needle.
12mm trocar placed at umbilicus.  Two 5mm ports placed in LLQ and suprapubic
regions.  Appendix identified — inflamed with perforation at the tip and
localised purulent fluid collection.

Mesoappendix divided with LigaSure.  Appendix base ligated with Endoloops
and divided.  Specimen placed in endobag and removed.  Irrigation performed
with 500mL warm saline.  Hemostasis confirmed.  Ports removed under vision.
Fascia closed at 12mm site.  Skin closed with subcuticular 4-0 Monocryl.

ESTIMATED BLOOD LOSS: 25 mL
SPECIMENS: Appendix to pathology
DRAINS: None
COMPLICATIONS: None

DISPOSITION: To recovery, stable.`,
  },
  {
    label: 'Radiology Report',
    text: `RADIOLOGY REPORT

Exam: CT Chest with IV Contrast
Date: 03/12/2025
Clinical History: 65-year-old female with persistent cough, weight loss, 
and hemoptysis.  Evaluate for pulmonary mass.

TECHNIQUE:
Helical CT of the chest performed with 100 mL Omnipaque IV contrast.
Axial images obtained from thoracic inlet to adrenal glands.  Coronal
and sagittal reformats generated.

COMPARISON: Chest X-ray dated 03/01/2025.

FINDINGS:
LUNGS: 3.2 x 2.8 cm spiculated mass in the right upper lobe, abutting
the mediastinal pleura.  No cavitation.  Multiple sub-centimeter nodules
in both lower lobes (largest 6mm in RLL) — likely metastatic.  No
consolidation or ground-glass opacity.

MEDIASTINUM: Enlarged subcarinal lymph node measuring 2.1 cm (short axis).
Right paratracheal lymph node 1.5 cm.  No pericardial effusion.
Heart size normal.

PLEURA: Small right-sided pleural effusion.  No pneumothorax.

BONES: Sclerotic lesion in T8 vertebral body — suspicious for metastasis.

IMPRESSION:
1. Right upper lobe spiculated mass highly suspicious for primary
   bronchogenic carcinoma.
2. Bilateral pulmonary nodules and mediastinal lymphadenopathy suggesting
   stage III-IV disease.
3. Suspicious T8 osseous lesion — recommend bone scan.
4. Small right pleural effusion — consider thoracentesis for cytology.`,
  },
];

// ─── Document Type Icons ─────────────────────────────────────

const TYPE_ICONS: Record<string, string> = {
  discharge_summary: '🏥',
  progress_note: '📋',
  history_physical: '📝',
  operative_note: '🔪',
  consultation_note: '🤝',
  radiology_report: '📡',
  pathology_report: '🔬',
  laboratory_report: '🧪',
  nursing_note: '👩‍⚕️',
  emergency_note: '🚨',
  dental_note: '🦷',
  prescription: '💊',
  referral: '📩',
  unknown: '❓',
};

// ─── Score Bar Colour ────────────────────────────────────────

function scoreColor(score: number): string {
  if (score >= 0.8) return 'bg-green-500';
  if (score >= 0.5) return 'bg-blue-500';
  if (score >= 0.3) return 'bg-yellow-500';
  return 'bg-gray-400';
}

function scoreTextColor(score: number): string {
  if (score >= 0.8) return 'text-green-600 dark:text-green-400';
  if (score >= 0.5) return 'text-blue-600 dark:text-blue-400';
  if (score >= 0.3) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-gray-500 dark:text-gray-400';
}

// ─── Component ───────────────────────────────────────────────

export function DocumentClassifier() {
  const [text, setText] = useState('');
  const [results, setResults] = useState<ClassificationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { classifyDocument } = await import('../services/clinical');
      const data = await classifyDocument(text);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setLoading(false);
    }
  }, [text]);

  const loadSample = (idx: number) => {
    setText(SAMPLE_NOTES[idx].text);
    setResults(null);
    setError(null);
  };

  // Sort scores descending
  const sortedScores: ClassificationScore[] = results
    ? [...results.scores].sort((a, b) => b.score - a.score)
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Document Classifier
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Classify clinical documents into 14 types — discharge summaries,
          operative notes, radiology reports, and more.
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-3">
          <label
            htmlFor="classify-input"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Clinical Document
          </label>
          <span className="text-xs text-gray-400">{wordCount} words</span>
        </div>

        <textarea
          id="classify-input"
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a clinical document to classify its type..."
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
            {loading ? 'Classifying…' : 'Classify Document'}
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
          {/* Predicted Type Card */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
            <p className="text-5xl mb-3">
              {TYPE_ICONS[results.predicted_type] || '📄'}
            </p>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white capitalize">
              {results.predicted_type.replace(/_/g, ' ')}
            </h2>
            <p className={`mt-1 text-lg font-semibold ${scoreTextColor(results.confidence)}`}>
              {(results.confidence * 100).toFixed(1)}% confidence
            </p>
            <p className="mt-1 text-xs text-gray-400">
              Processed in {results.processing_time_ms.toFixed(1)}ms
            </p>
          </div>

          {/* Score Distribution */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
              Classification Scores
            </h3>
            <div className="space-y-3">
              {sortedScores.map((item) => (
                <div key={item.document_type} className="flex items-center gap-3">
                  <span className="text-lg w-8 text-center">
                    {TYPE_ICONS[item.document_type] || '📄'}
                  </span>
                  <span className="w-40 text-sm text-gray-700 dark:text-gray-300 capitalize truncate">
                    {item.document_type.replace(/_/g, ' ')}
                  </span>
                  <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${scoreColor(item.score)} ${
                        item.document_type === results.predicted_type ? 'ring-2 ring-blue-400' : ''
                      }`}
                      style={{ width: `${Math.max(item.score * 100, 1)}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-xs text-gray-500 dark:text-gray-400">
                    {(item.score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Evidence for Top Predictions */}
          {sortedScores
            .filter((s) => s.evidence && s.evidence.length > 0 && s.score >= 0.1)
            .slice(0, 5)
            .map((item) => (
              <div
                key={`evidence-${item.document_type}`}
                className="bg-white dark:bg-gray-800 rounded-lg shadow p-4"
              >
                <div className="flex items-center gap-2 mb-2">
                  <span>{TYPE_ICONS[item.document_type] || '📄'}</span>
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                    {item.document_type.replace(/_/g, ' ')} — Evidence Keywords
                  </h4>
                  <span className="ml-auto text-xs text-gray-400">
                    {(item.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {item.evidence.map((kw, idx) => (
                    <span
                      key={`${kw}-${idx}`}
                      className="px-2 py-0.5 text-xs rounded bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
