# Model Card — PHI De-identification

## Model Details

| Field | Value |
|---|---|
| **Model Name** | ClinIQ PHI De-identifier |
| **Version** | 1.0.0 |
| **Type** | Rule-based pattern matching with contextual heuristics |
| **Task** | Protected Health Information detection and redaction |
| **Framework** | Python stdlib (re module) |
| **License** | MIT |

## Intended Use

### Primary Use Cases

- **Pre-processing clinical text** before feeding into NLP pipelines (NER, ICD-10, summarization) to remove PHI that could pose compliance risks.
- **Training data preparation** — de-identify raw clinical notes to create safe training corpora for ML models (use SURROGATE strategy for realistic synthetic replacements).
- **Audit and compliance** — scan clinical documents to identify and catalog PHI locations for HIPAA compliance reviews.
- **Research data sharing** — redact PHI from clinical notes before sharing with collaborators or external research teams.

### Out of Scope

- **Sole HIPAA compliance mechanism** — this module is one layer of defence; it does NOT guarantee full HIPAA compliance on its own.
- **Non-English clinical text** — patterns are tuned for US English clinical documents.
- **Novel name formats** — highly unusual names, transliterated names, or names with non-Latin characters may not be detected.
- **Handwritten or OCR'd notes** — text with OCR artefacts may reduce pattern matching accuracy.

## HIPAA Safe Harbor Coverage

All 18 identifier categories from 45 CFR §164.514(b)(2):

| # | Category | Detection Method | Base Confidence |
|---|---|---|---|
| 1 | Names | Titled patterns (Dr./Mr./Mrs./etc.) | 0.90 |
| 2 | Dates | 4 format patterns (MM/DD/YYYY, ISO, Month DD YYYY, DD Month YYYY) | 0.95 |
| 3 | Phone numbers | US format with optional country code | 0.90 |
| 4 | Fax numbers | Classified via phone pattern + "fax" context | 0.90 |
| 5 | Email addresses | RFC-style pattern | 0.98 |
| 6 | SSN | 9-digit with exclusion rules (000, 666, 9xx) | 0.85 |
| 7 | MRN | Prefix-based (MRN, Medical Record) | 0.97 |
| 8 | Health plan numbers | Prefix-based | 0.90 |
| 9 | Account numbers | Prefix-based (Account, Acct) | 0.90 |
| 10 | Certificate/license | Prefix-based (License, DEA, NPI) | 0.90 |
| 11 | Vehicle identifiers | Tag placeholder | — |
| 12 | Device identifiers | Tag placeholder | — |
| 13 | URLs | HTTP/HTTPS pattern | 0.98 |
| 14 | IP addresses | IPv4 dotted-quad | 0.90 |
| 15 | Biometric identifiers | Custom detector hook | — |
| 16 | Photographs | Custom detector hook | — |
| 17 | Geographic data | ZIP code (5-digit, ZIP+4) with context validation | 0.70 |
| 18 | Ages over 89 | Numeric + age suffix pattern | 0.92 |

## Architecture

```
Input Text
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ PhiDetector │────▶│ Pattern Registry │  (compiled regex, ordered by priority)
│             │     │  • name_titled   │
│             │     │  • date_0..3     │
│             │     │  • phone_us      │
│             │     │  • email         │
│             │     │  • ssn           │
│             │     │  • mrn           │
│             │     │  • url           │
│             │     │  • ip            │
│             │     │  • zip           │
│             │     │  • age_over_89   │
│             │     │  • account       │
│             │     │  • license       │
│             │     └──────────────────┘
│             │
│             │────▶ Custom Detectors (optional transformer NER)
│             │
│  Confidence │────▶ Context Window Adjustment
│  Adjustment │     (prefix/suffix heuristics per PhiType)
│             │
│   Overlap   │────▶ Longest-match-wins resolution
│  Resolution │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Deidentifier │──▶ REDACT:    [TYPE] tags
│              │──▶ MASK:      ****  (length-preserving)
│              │──▶ SURROGATE: Synthetic values (deterministic)
└──────────────┘
       │
       ▼
  Output Text + Entity Metadata
```

## Replacement Strategies

| Strategy | Example Input | Example Output | Use Case |
|---|---|---|---|
| **REDACT** | `Dr. Smith on 01/15/2024` | `[NAME] on [DATE]` | Logging, display, analysis |
| **MASK** | `Dr. Smith on 01/15/2024` | `********* on **********` | Quick visual redaction |
| **SURROGATE** | `Dr. Smith on 01/15/2024` | `Dr. Johnson on 01/01/2000` | Training data preparation |

## Limitations and Biases

- **Name recall is context-dependent**: untitled names (e.g. "sent to John for review") will not be detected without a title prefix. This is a deliberate precision-over-recall trade-off to avoid false positives on medical terms.
- **ZIP code false positives**: 5-digit numbers in clinical text (medication dosages like "10000 units", lab values) may trigger ZIP code detection. The context-based confidence adjustment mitigates this by requiring address-like context for high-confidence ZIP detection.
- **SSN false positives**: 9-digit numbers without "SSN" or "Social Security" context receive a confidence penalty. Use the confidence threshold parameter to tune the precision/recall balance.
- **Date format coverage**: only US and ISO date formats are supported. European DD/MM/YYYY dates may be incorrectly parsed.

## Evaluation

| Metric | Value | Notes |
|---|---|---|
| Test count | 83 | Unit + API route tests |
| PHI types tested | 15/18 | Vehicle, device, biometric via custom detector hook |
| Strategies tested | 3/3 | REDACT, MASK, SURROGATE |
| Edge cases | ✓ | Empty text, no PHI, overlaps, batch, thresholds |

## Ethical Considerations

- This module processes sensitive health information. It should be used as part of a comprehensive privacy programme, not as a standalone solution.
- Surrogate values are deliberately unrealistic (e.g. "(555) 000-0001") to prevent re-identification.
- The module does NOT store or transmit any PHI — all processing is in-memory and stateless.
