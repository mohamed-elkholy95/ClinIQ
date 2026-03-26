# Clinical Error Analysis

Systematic analysis of error patterns across ClinIQ's ML modules, informed by
clinical domain expertise. This document examines failure modes, their clinical
significance, root causes, and mitigation strategies.

---

## Table of Contents

1. [Methodology](#methodology)
2. [NER — Entity Recognition Errors](#ner--entity-recognition-errors)
3. [ICD-10 — Code Prediction Errors](#icd-10--code-prediction-errors)
4. [Risk Scoring Errors](#risk-scoring-errors)
5. [Clinical Summarisation Errors](#clinical-summarisation-errors)
6. [Dental NLP Errors](#dental-nlp-errors)
7. [Medication Extraction Errors](#medication-extraction-errors)
8. [De-identification Errors](#de-identification-errors)
9. [Cross-Module Error Propagation](#cross-module-error-propagation)
10. [Clinical Safety Taxonomy](#clinical-safety-taxonomy)
11. [Recommendations](#recommendations)

---

## Methodology

Errors are categorised by:

- **Clinical severity**: Critical (could affect patient care), Moderate
  (incorrect but unlikely to harm), Low (cosmetic or informational).
- **Error type**: False positive (hallucinated), false negative (missed),
  misclassification (wrong label), boundary error (partial span).
- **Frequency**: Common (>5% of cases), Occasional (1–5%), Rare (<1%).

Where possible, examples use synthetic clinical text representative of real
documentation patterns observed in practice.

---

## NER — Entity Recognition Errors

### Error Pattern 1: Anatomical Ambiguity

**Severity**: Moderate | **Frequency**: Common

Clinical text frequently uses terms that are valid in multiple contexts:

| Input | Predicted | Correct | Issue |
|-------|-----------|---------|-------|
| "Patient has a broken **crown**" | ANATOMY | PROCEDURE | In dental context, "crown" is a prosthetic |
| "**Discharge** from wound" | — | SYMPTOM | Confused with document type "discharge" |
| "Left **bank** of lesion" | — | ANATOMY | Rare anatomical reference missed |

**Root cause**: Context-free token matching. The rule-based layer matches
surface forms without sentence-level disambiguation.

**Mitigation**: The composite NER pipeline uses majority voting across
rule-based, scispaCy, and transformer models. Transformer models capture
context better, but rule-based patterns dominate for known entities. Adding
a dental domain context flag improves crown/bridge disambiguation.

### Error Pattern 2: Negated Entities

**Severity**: Critical | **Frequency**: Occasional

```
"Patient denies chest pain, shortness of breath, or nausea."
```

Without negation detection, NER extracts "chest pain", "shortness of breath",
and "nausea" as present conditions — the opposite of clinical reality.

**Root cause**: NER operates at the entity level; negation scope requires
syntactic understanding.

**Mitigation**: ClinIQ's assertion detection module (NegEx-inspired) runs as a
post-processing stage. The enhanced pipeline chains NER → assertion to mark
entities as `NEGATED`, `POSSIBLE`, `HYPOTHETICAL`, etc. However, complex
negation scope (double negatives, long-distance negation) remains challenging:

```
"There is no evidence that the patient does not have diabetes."
→ Assertion module may mark "diabetes" as NEGATED (incorrect — double negative)
```

### Error Pattern 3: Abbreviation Collisions

**Severity**: Moderate | **Frequency**: Common

Clinical abbreviations are notoriously ambiguous:

| Abbreviation | Context A | Context B |
|-------------|-----------|-----------|
| `MS` | Multiple Sclerosis | Morphine Sulfate |
| `BS` | Blood Sugar | Bowel Sounds |
| `PT` | Physical Therapy | Prothrombin Time |
| `CA` | Cancer | Calcium |
| `RA` | Rheumatoid Arthritis | Right Atrium |

**Root cause**: The abbreviation expansion module maintains a dictionary of
120+ entries across 12 clinical domains, but many abbreviations have
legitimate multiple expansions.

**Mitigation**: Context-aware expansion uses surrounding terms and detected
document section. Emergency department notes mentioning "MS" near "pain
management" favour "Morphine Sulfate"; neurology notes favour "Multiple
Sclerosis". Ambiguous cases are flagged rather than silently expanded.

### Error Pattern 4: Span Boundary Errors

**Severity**: Low | **Frequency**: Common

```
Input:  "acute exacerbation of chronic obstructive pulmonary disease"
Pred:   [chronic obstructive pulmonary disease]  (missed "acute exacerbation of")
Gold:   [acute exacerbation of chronic obstructive pulmonary disease]
```

Partial spans capture the disease but lose clinical acuity. The partial NER
evaluation module (Jaccard overlap) quantifies this: typical overlap is 70–85%
for multi-word entities.

**Mitigation**: The evaluation framework reports both exact and partial F1,
making boundary quality visible. Transformer models with BIO tagging handle
multi-word entities better than rule-based patterns.

---

## ICD-10 — Code Prediction Errors

### Error Pattern 1: Specificity Mismatch

**Severity**: Moderate | **Frequency**: Common

```
Text:  "Type 2 diabetes with diabetic retinopathy"
Pred:  E11.9  (Type 2 DM without complications)
Gold:  E11.319 (Type 2 DM with unspecified diabetic retinopathy)
```

The model predicts the correct disease family but wrong specificity level.
ClinIQ's hierarchical ICD-10 evaluation reveals this pattern: chapter accuracy
is typically 15–20% higher than full-code accuracy.

**Clinical impact**: Revenue cycle implications (less specific codes = lower
reimbursement) but limited direct patient safety risk.

**Root cause**: The ~400 high-frequency code dictionary captures common codes
well, but long-tail specific codes (e.g., laterality, episode of care) require
deeper clinical reasoning.

**Mitigation**: The hierarchical prediction approach — first predict chapter,
then narrow within chapter — reduces catastrophic errors (wrong organ system).
The confidence threshold mechanism flags low-confidence predictions for human
review rather than committing to a specific code.

### Error Pattern 2: Comorbidity Splitting

**Severity**: Moderate | **Frequency**: Occasional

```
Text: "Hypertensive chronic kidney disease, stage 3"
Pred: [I10 (HTN), N18.3 (CKD stage 3)]  — two separate codes
Gold: [I12.9 (Hypertensive CKD)]          — one combination code
```

ICD-10 has specific combination codes for conditions that frequently co-occur.
The multi-label classifier may predict component codes independently rather
than recognising the combination.

**Root cause**: Multi-label sigmoid outputs treat each code independently;
there is no explicit modelling of ICD-10 combination code rules.

**Mitigation**: A post-processing rule layer checks predicted code sets against
known combination code patterns and suggests the combination code when
component codes co-occur. This is documented as a planned enhancement.

### Error Pattern 3: Present-on-Admission Confusion

**Severity**: Critical | **Frequency**: Occasional

```
Text: "Patient developed hospital-acquired pneumonia on day 5."
Pred: J18.9 (Pneumonia, unspecified)  — no POA distinction
```

The model predicts the diagnosis but cannot distinguish conditions present at
admission from those acquired during the encounter. This distinction is
required for billing and quality reporting.

**Root cause**: POA status requires temporal reasoning about when conditions
developed relative to the admission date. The ICD module operates on text
content, not document timeline.

**Mitigation**: The temporal extraction module identifies time expressions
("on day 5", "post-admission") that can contextualise ICD predictions. Full
POA classification is flagged as a future enhancement requiring encounter-level
context beyond single-document analysis.

---

## Risk Scoring Errors

### Error Pattern 1: Missing Upstream Entities

**Severity**: Critical | **Frequency**: Occasional

The risk scorer depends on NER-extracted entities. If NER misses a critical
condition (e.g., "metastatic cancer" in an abbreviated note), the risk score
is artificially low.

```
Text: "Pt w/ mets to liver, ESRD on HD, recent MI"
NER:  Extracts "liver", "ESRD", "HD" but misses "mets" (abbreviation)
Risk: Moderate (missing cancer → missing the highest-weight factor)
Gold: Critical
```

**Root cause**: Error propagation from upstream NER. The risk scorer is a
downstream consumer of NER output; it has no independent text analysis.

**Mitigation**: The enhanced pipeline runs abbreviation expansion before NER,
improving entity extraction from abbreviated text. The risk module also has
keyword-based fallback rules that scan raw text for high-severity terms
("metastatic", "terminal", "code blue") independently of NER.

### Error Pattern 2: Additive Model Limitations

**Severity**: Moderate | **Frequency**: Occasional

The rule-based risk scorer uses additive factor weights. This means two
moderate-risk factors sum to a higher score than one high-risk factor, which
may not reflect clinical reality.

```
Factor A: Hypertension       → weight 0.15
Factor B: Mild obesity       → weight 0.10
Sum: 0.25

Factor C: Active malignancy  → weight 0.30
→ Two mild factors outscore one serious factor
```

**Root cause**: Additive models cannot capture non-linear risk interactions
(e.g., diabetes + CKD is more dangerous than the sum of parts).

**Mitigation**: This trade-off is by design — the additive model is
transparent and explainable, which is essential for clinical decision support
where clinicians must understand *why* a score was assigned. The model
complements (not replaces) validated clinical calculators like the Charlson
Comorbidity Index, which the comorbidity module implements separately.

---

## Clinical Summarisation Errors

### Error Pattern 1: Hallucinated Details

**Severity**: Critical | **Frequency**: Rare (rule-based) / Occasional (generative)

Rule-based extractive summarisation cannot hallucinate because it selects
existing sentences. However, if future versions use generative models:

```
Input:  "Patient prescribed lisinopril 10mg daily."
Output: "Patient prescribed lisinopril 20mg twice daily."  (hallucinated dose)
```

**Mitigation**: ClinIQ uses extractive summarisation (sentence selection and
ranking) specifically to avoid hallucination. The architecture documentation
explicitly records this design decision: extractive over generative for clinical
safety. ROUGE evaluation with length ratio monitoring detects compression
anomalies.

### Error Pattern 2: Section Weighting Bias

**Severity**: Moderate | **Frequency**: Occasional

The summariser weights "Assessment and Plan" sections higher than "Social
History" or "Review of Systems". This is clinically appropriate for most use
cases but may suppress important social determinants:

```
A patient with housing instability affecting medication adherence may have
this detail buried in social history and excluded from the summary.
```

**Mitigation**: The SDoH extraction module runs independently and surfaces
social determinants regardless of summarisation. The enhanced pipeline
includes SDoH as a dedicated output alongside the summary.

---

## Dental NLP Errors

### Error Pattern 1: Tooth Numbering System Confusion

**Severity**: Moderate | **Frequency**: Occasional

Three numbering systems coexist in dental documentation:

| System | Example | Meaning |
|--------|---------|---------|
| Universal | #30 | Lower right first molar |
| Palmer | LR6 | Lower right first molar |
| FDI | 46 | Lower right first molar |

Misidentifying the system converts one tooth to a completely different tooth.
"#14" in Universal is the upper left first premolar; in FDI it's the upper
right canine.

**Root cause**: Many dental notes don't explicitly state which numbering
system they use. The module infers it from context (US practices typically
use Universal; international practices use FDI).

**Mitigation**: The dental NER module uses heuristics: numbers 1–32 suggest
Universal, two-digit numbers starting with 1–4 suggest FDI, alphanumeric
patterns suggest Palmer. When ambiguous, both interpretations are returned
with confidence scores.

### Error Pattern 2: Periodontal Measurement Parsing

**Severity**: Low | **Frequency**: Occasional

```
Input:  "Probing depths: 3,4,3,5,4,3 on #3 buccal"
```

Six measurements per tooth (mesiobuccal, buccal, distobuccal, mesiolingual,
lingual, distolingual) must be correctly associated with tooth and surface.
OCR artifacts, inconsistent formatting, and missing delimiters cause parsing
errors.

**Mitigation**: Regex patterns handle common formats. Validation rules reject
biologically impossible values (probing depths >15mm, negative values).

---

## Medication Extraction Errors

### Error Pattern 1: Discontinued vs Active Medications

**Severity**: Critical | **Frequency**: Occasional

```
"Metformin 500mg BID — DISCONTINUED due to GI intolerance.
Started glipizide 5mg daily."
```

Both medications are extracted, but status classification must correctly
identify metformin as discontinued and glipizide as active. Errors in status
classification could lead to a "current medications" list containing stopped
drugs.

**Root cause**: Status keywords ("discontinued", "held", "stopped") may appear
far from the medication name, outside the extraction window.

**Mitigation**: The medication extractor uses a multi-pass approach: first
extract medication names, then scan the surrounding context (±3 sentences)
for status modifiers. The assertion detection module provides additional
signal about whether a medication mention is in a "negated" context.

### Error Pattern 2: Dosage Unit Ambiguity

**Severity**: Moderate | **Frequency**: Rare

```
"Amoxicillin 500 TID" — Is that 500mg or 500mcg?
```

When units are omitted, the module must infer based on the drug and typical
dosing. Amoxicillin 500mcg is not a real dose, but for less common drugs the
inference is harder.

**Mitigation**: A dosage validation layer checks extracted doses against
known therapeutic ranges per drug. Out-of-range doses are flagged rather than
silently accepted.

---

## De-identification Errors

### Error Pattern 1: Contextual PHI in Clinical Content

**Severity**: Critical | **Frequency**: Occasional

```
"Patient is the daughter of John Smith, who has a history of Huntington's."
```

"John Smith" is PHI (family member name), but the family history of
Huntington's disease is clinically relevant. Over-redaction removes clinical
information; under-redaction leaks PHI.

**Root cause**: Named entity boundaries overlap between PHI (person names) and
clinical content (family history context). The de-identification module must
redact names while preserving the clinical relationship.

**Mitigation**: The surrogate replacement strategy generates realistic fake
names (preserving gender and ethnicity patterns) so the clinical narrative
remains readable. The de-identification module detects 12 PHI types and uses
confidence thresholds — low-confidence detections are flagged for human review
rather than auto-redacted.

### Error Pattern 2: Structured Data Leakage

**Severity**: Critical | **Frequency**: Rare

Medical record numbers, dates of service, and provider names embedded in
structured formats (tables, headers, fax lines) may be missed by pattern
matchers designed for narrative text.

```
"FAX FROM: Dr. Jane Doe, 555-123-4567, MRN: 12345678"
```

**Mitigation**: The input sanitisation middleware and de-identification module
handle both narrative and structured patterns. Phone number, MRN, and fax
number patterns are matched with dedicated regex rules. Header/footer
detection targets common structured PHI locations.

---

## Cross-Module Error Propagation

ClinIQ's enhanced pipeline chains 14 modules sequentially. Errors propagate
downstream:

```
Text Input
  → NER (extracts entities with errors)
    → ICD-10 (receives wrong entities → predicts wrong codes)
      → Risk Scoring (receives wrong codes → wrong risk level)
        → Summary (includes wrong risk in narrative)
```

### Propagation Analysis

| Upstream Error | Downstream Impact | Severity |
|---------------|-------------------|----------|
| NER misses medication | Medication list incomplete, drug interaction check fails | Critical |
| NER false positive entity | ICD predicts spurious code, risk inflated | Moderate |
| Abbreviation unexpanded | NER misses entity, cascading to ICD/risk | Moderate |
| Section parser mislabels | Summariser weights wrong section | Low |
| De-id over-redacts | Downstream modules receive `[REDACTED]` tokens | Moderate |

### Mitigation Strategy

1. **Fail-safe defaults**: When a module fails, downstream modules receive
   empty inputs rather than corrupted data. The enhanced pipeline catches
   per-module exceptions independently.
2. **Confidence propagation**: Low-confidence NER entities are flagged so
   downstream consumers can apply different thresholds.
3. **Independent modules**: Some modules (risk keyword scanner, comorbidity
   from ICD codes) have parallel paths that don't depend on NER.
4. **Circuit breaker**: If a model's error rate exceeds thresholds, the
   circuit breaker disables it and falls back to rule-based alternatives.

---

## Clinical Safety Taxonomy

### Critical Errors (Require Immediate Attention)

| ID | Error | Module | Impact |
|----|-------|--------|--------|
| C1 | Negated entity treated as present | NER + Assertion | Wrong diagnosis |
| C2 | Discontinued medication listed as active | Medications | Drug safety |
| C3 | Missing high-risk condition in risk score | Risk | Undertriage |
| C4 | PHI leakage in de-identified output | De-id | Privacy violation |
| C5 | Hallucinated medication dose | Summarisation | Drug safety |

### Moderate Errors (Should Be Addressed)

| ID | Error | Module | Impact |
|----|-------|--------|--------|
| M1 | Wrong ICD-10 specificity | ICD | Billing accuracy |
| M2 | Anatomical term ambiguity | NER | Documentation |
| M3 | Tooth numbering confusion | Dental | Wrong tooth |
| M4 | Abbreviation misexpansion | Abbreviations | Documentation |

### Low Errors (Monitor Over Time)

| ID | Error | Module | Impact |
|----|-------|--------|--------|
| L1 | Entity span boundary off by word | NER | Evaluation metrics |
| L2 | Section label mismatch | Sections | Summary weighting |
| L3 | Periodontal measurement parsing | Dental | Data entry |

---

## Recommendations

### Short-term (Next Sprint)

1. **Increase negation test coverage**: Add adversarial test cases with
   double negatives, long-distance negation, and scope ambiguity.
2. **ICD combination code rules**: Implement post-processing layer for
   hypertensive CKD, diabetic complications, and other common combinations.
3. **Medication status validation**: Cross-reference medication status with
   assertion detection output for independent verification.

### Medium-term (1–3 Months)

4. **Confidence calibration**: Run ECE analysis on NER and ICD modules to
   verify that stated confidence scores match actual accuracy. The evaluation
   framework already supports calibration metrics.
5. **Error logging pipeline**: Structured logging of prediction errors with
   clinical severity tags for ongoing monitoring.
6. **Adversarial test suite**: Curated set of clinically tricky inputs
   (abbreviation collisions, negation scope, cross-system tooth numbers).

### Long-term (3–6 Months)

7. **Active learning**: Use clinician feedback to identify and retrain on
   error-prone input patterns.
8. **Ensemble arbitration**: Replace majority voting with learned weights
   per entity type based on historical precision/recall per module.
9. **Clinical validation study**: Structured comparison against manual
   clinician annotation on a held-out dataset with inter-annotator agreement
   measurement via Cohen's Kappa.

---

*Analysis authored: March 2026*
*Based on: ClinIQ v0.1.0, 14 ML modules, 3094 tests*
*Review cadence: Monthly, aligned with model retraining cycles*
