# Model Card: Clinical Document Type Classifier

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ Document Type Classifier |
| **Version** | 1.0.0 |
| **Task** | Multi-class document classification |
| **Architecture** | Rule-based (primary) + Transformer (optional) |
| **Input** | Raw clinical document text |
| **Output** | Ranked document type predictions with confidence scores |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Input Clinical Document                    │
└─────────────────────────┬────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Transformer Model   │──── unavailable? ───┐
              │  (HuggingFace seq-    │                     │
              │   classification)     │                     │
              └───────────┬───────────┘                     │
                          │ fallback                        │
              ┌───────────▼───────────────────────▼─────────┐
              │      Rule-Based Classifier                   │
              │                                              │
              │  ┌─────────────────┐  weight: 0.45          │
              │  │ Section Header  │  Pattern matching for   │
              │  │ Detection       │  type-specific headers  │
              │  └────────┬────────┘                         │
              │           │                                  │
              │  ┌────────▼────────┐  weight: 0.30          │
              │  │ Keyword Density │  Term frequency in      │
              │  │ Analysis        │  lower-cased text       │
              │  └────────┬────────┘                         │
              │           │                                  │
              │  ┌────────▼────────┐  weight: 0.25          │
              │  │ Structural      │  Line count, section    │
              │  │ Features        │  count profiles         │
              │  └────────┬────────┘                         │
              └───────────┼──────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Weighted Score      │
              │   Aggregation         │
              │   + Ranking           │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  ClassificationResult │
              │  predicted_type       │
              │  scores[]             │
              │  evidence[]           │
              └───────────────────────┘
```

## Supported Document Types (13)

| Type | Description | Key Indicators |
|------|-------------|----------------|
| `discharge_summary` | Hospital discharge documentation | "Hospital Course", "Discharge Diagnosis", "Discharge Medications" |
| `progress_note` | Daily SOAP-format clinical notes | "Subjective", "Objective", "Assessment", "Plan" |
| `history_physical` | Comprehensive H&P examination | "Chief Complaint", "HPI", "ROS", "Physical Exam" |
| `operative_note` | Surgical procedure documentation | "Pre-operative Diagnosis", "Procedure Performed", "Blood Loss" |
| `consultation_note` | Specialist consultation report | "Reason for Consult", "Recommendations" |
| `radiology_report` | Imaging study interpretation | "Findings", "Impression", "Technique", CT/MRI/X-ray mentions |
| `pathology_report` | Histopathological analysis | "Gross Description", "Microscopic", "Specimen" |
| `laboratory_report` | Clinical lab test results | "Reference Range", "Flag", CBC/CMP/BMP mentions |
| `nursing_note` | Nursing assessment documentation | "Vitals", "Pain Assessment", "Intake/Output" |
| `emergency_note` | ED encounter documentation | "Triage", "Mode of Arrival", "Acuity" |
| `dental_note` | Dental examination/procedure notes | Tooth numbers, "Probing Depth", CDT codes |
| `prescription` | Medication prescription | "Sig", "Dispense", "Refills", dosage units |
| `referral` | Referral letter/request | "Reason for Referral", "Referred to", "Specialist" |

## Scoring Formula

```
score(type) = 0.45 × section_score + 0.30 × keyword_score + 0.25 × structural_score
```

### Section Score (weight: 0.45)
- Fraction of type-specific section header regex patterns matched
- +0.1 bonus if first match appears in opening 500 characters
- Capped at 1.0

### Keyword Score (weight: 0.30)
- Fraction of type-specific keywords found in lower-cased text
- Capped at 1.0

### Structural Score (weight: 0.25)
- 0.5 if line count falls within type's expected range
- 0.5 if section count meets type's profile expectations
- Unstructured types score higher with fewer sections

## Evidence Attribution

Each prediction includes evidence strings showing:
- **Section headers**: Matched header patterns from the text
- **Keywords**: Matched domain-specific terms

This supports interpretability and clinical validation of the classification.

## Performance Characteristics

| Metric | Rule-Based | Transformer |
|--------|-----------|-------------|
| Latency (per doc) | < 1 ms | 50–200 ms |
| Dependencies | None (regex only) | PyTorch, Transformers |
| Accuracy (structured docs) | High | Very High |
| Accuracy (ambiguous docs) | Moderate | High |
| GPU required | No | Optional |

## Limitations

1. **Ambiguous documents**: Notes combining multiple types (e.g., H&P with embedded operative note) may produce multiple high-confidence scores rather than a single clear winner
2. **Free-text notes**: Unstructured narrative without section headers relies more heavily on keyword density, which is less discriminative
3. **Language**: English clinical text only; no multilingual support
4. **Template variation**: Accuracy depends on documents following common clinical formatting conventions
5. **Rule-based ceiling**: Pattern-based classification cannot capture semantic nuances that a fine-tuned transformer model would recognise

## Ethical Considerations

- Document classification does **not** process or expose PHI; it operates on structural features
- Misclassification could affect downstream routing in clinical workflows — always pair with human review in production
- The classifier is designed as an assistive tool, not a replacement for clinical documentation specialists

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classify` | Classify a single document |
| `POST` | `/classify/batch` | Classify up to 50 documents |
| `GET`  | `/classify/types` | List all 13 supported document types |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-25 | Initial release with 13 document types, rule-based + transformer architecture |
