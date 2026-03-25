# Model Card: Clinical Risk Scoring

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | ClinIQ Risk Scorer |
| **Version** | 1.0.0 |
| **Type** | Rule-based + ML hybrid |
| **Task** | Clinical document risk stratification |
| **Output** | Risk score (0–1), risk level, category scores, recommendations |
| **Inference Time** | < 5 ms per document (rule-based) |
| **Dependencies** | None (rule-based); scikit-learn (ML scorer) |

## Description

The ClinIQ Risk Scorer assesses clinical documents for patient risk across
five categories: medication, cardiovascular, infection, surgical, and
follow-up compliance.  It combines entity-level signals (polypharmacy,
high-risk medications), ICD code chapter mapping, and text pattern matching
for urgency keywords and high-risk conditions.

## Architecture

```
Clinical Note ─┬─► Text Risk Factors (urgency keywords, conditions)
               ├─► Entity Risk Factors (medication count, high-risk meds)
               ├─► ICD Risk Factors (chapter-based severity mapping)
               │
               ▼
         Category Score Calculator
               │
               ▼
         Overall Score = 0.7 × category_weighted_avg + 0.3 × factor_boost
               │
               ▼
         Risk Level Classifier
         ├── critical  ≥ 0.8
         ├── high      ≥ 0.6
         ├── moderate  ≥ 0.4
         └── low       < 0.4
               │
               ▼
         Recommendation Generator
```

## Risk Categories

| Category | Weight | Factors Assessed |
|---|---|---|
| Medication | 0.25 | Polypharmacy (≥5 meds), high-risk drug list (12 medications), interaction signals |
| Cardiovascular | 0.25 | Hypertension, cardiac history, coronary disease, atrial conditions |
| Infection | 0.20 | Active infection, sepsis, immunosuppression, recent hospitalisation |
| Surgical | 0.15 | Prior surgery, bleeding risk, anaesthesia risk, post-operative status |
| Follow-up | 0.15 | Non-compliance, missed appointments, social barriers |

## High-Risk Condition Map

20 conditions are mapped with severity weights ranging from 0.30 (UTI) to
0.95 (cardiac arrest).  Critical conditions (MI, stroke, PE, sepsis) carry
weights ≥ 0.90.

## High-Risk Medication List

12 medications with weights from 0.50 (benzodiazepines) to 0.75 (fentanyl),
including anticoagulants (warfarin, heparin), insulin, digoxin, lithium,
methotrexate, and opioids.

## Scoring Algorithm

1. **Text risk factors**: Match urgency keywords (critical/high/moderate
   tiers) and known high-risk conditions against document text.
2. **Entity risk factors**: Count medications for polypharmacy detection;
   check each medication against the high-risk list; map non-negated
   disease entities to the condition severity table.
3. **ICD risk factors**: Map top-10 ICD predictions to chapter-based
   severity weights (Circulatory 0.6, Neoplasms 0.7, Endocrine 0.5, etc.),
   scaled by prediction confidence.
4. **Category scores**: Keyword matching within each category; normalised
   to [0, 1].
5. **Overall score**: Weighted average of category scores (70%) plus
   boost from high-weight risk factors (30%).
6. **Recommendations**: Generated based on overall risk level and
   category-specific thresholds.

## ML Risk Scorer (Optional)

The `MLRiskScorer` uses a trained scikit-learn classifier (logistic
regression or gradient boosting) with features extracted from:
- Entity counts per type
- High-risk medication presence (binary features)
- ICD chapter indicators
- Text length and section count
- Urgency keyword density

Falls back to the rule-based scorer when no trained model is available.

## Recommendations Generation

| Risk Level | Recommendation |
|---|---|
| Critical (≥ 0.8) | Immediate clinical review; consider specialist escalation |
| High (≥ 0.6) | Urgent follow-up within 48–72 hours |
| Moderate (≥ 0.4) | Routine follow-up within 1–2 weeks |
| Low (< 0.4) | Standard care pathway |

Additional category-specific recommendations trigger when individual
category scores exceed 0.6 (medication review, cardiology consult,
infectious disease workup).

## Limitations

- **Rule-based**: No learning from labelled outcomes; risk weights are
  expert-assigned, not data-driven.
- **Urgency keyword sensitivity**: Exact string matching may miss
  paraphrased urgency signals.
- **No temporal reasoning**: Does not distinguish current vs. historical
  conditions (e.g. "history of MI" scored same as "acute MI").
- **English only**: All pattern matching is English-language.

## Ethical Considerations

- Risk scores are **decision support**, not clinical decisions.
- Scores should be reviewed by qualified clinicians before acting.
- The system may exhibit bias toward conditions well-represented in the
  high-risk condition and medication maps.
- Rare conditions or non-standard terminology may be under-scored.

## Intended Use

- Triage support for clinical note review prioritisation.
- Automated flagging of high-risk patients for follow-up.
- Quality assurance auditing of discharge documentation.

## Not Intended For

- Autonomous clinical decision-making without human oversight.
- Diagnostic purposes.
- Insurance risk underwriting.
