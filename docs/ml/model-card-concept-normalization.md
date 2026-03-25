# Model Card: Clinical Concept Normalization

## Overview

The **Clinical Concept Normalizer** maps extracted entity mentions (from NER or manual input) to standardised medical ontology codes. It bridges the gap between raw clinical text and structured, interoperable medical data.

**Module**: `app.ml.normalization.normalizer`
**Type**: Rule-based entity linker (dictionary + fuzzy matching)
**Version**: 1.0.0

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Entity Mention Input                     │
│          ("HTN", "heart attack", "Lipitor")              │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│              Text Normalization                          │
│   • Case folding                                         │
│   • Whitespace collapsing                                │
│   • Trailing period removal                              │
└──────────────────┬───────────────────────────────────────┘
                   │
          ┌────────┼────────────────┐
          ▼        │                ▼
   ┌──────────┐    │     ┌───────────────┐
   │  Exact   │────┘     │    Alias      │
   │  Match   │          │    Match      │
   │ (conf=1) │          │  (conf=0.95)  │
   └──────────┘          └───────────────┘
          │                      │
          │    (both miss)       │
          └──────┬───────────────┘
                 ▼
   ┌──────────────────────────────┐
   │      Fuzzy Matching          │
   │   (SequenceMatcher ratio)    │
   │   conf = similarity ratio   │
   │   threshold ≥ 0.80          │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │    Type-Aware Filtering      │
   │   (optional entity_type →    │
   │    EntityTypeGroup mapping)  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   NormalizationResult        │
   │   • CUI, SNOMED, RxNorm     │
   │   • ICD-10-CM, LOINC        │
   │   • confidence, match_type  │
   │   • alternatives             │
   └──────────────────────────────┘
```

## Concept Dictionary

### Coverage by Type Group

| Type Group  | Concepts | Examples |
|-------------|----------|----------|
| CONDITION   | ~55      | Hypertension, Diabetes, COPD, Dental Caries, Depression |
| MEDICATION  | ~40      | Metformin, Lisinopril, Atorvastatin, Amoxicillin |
| PROCEDURE   | ~18      | ECG, MRI, CT Scan, CABG, Root Canal, SRP |
| ANATOMY     | ~7       | Heart, Lung, Liver, Kidney, Brain |
| LAB         | ~20      | CBC, BMP, HbA1c, Troponin, TSH, LFTs |

### Ontology Coverage

| Ontology    | Mapped Concepts | Use Case |
|-------------|-----------------|----------|
| UMLS CUI    | All (~140)      | Universal concept identity |
| SNOMED-CT   | ~60             | Clinical findings, procedures |
| RxNorm      | ~40             | Medication normalisation |
| ICD-10-CM   | ~45             | Diagnostic coding |
| LOINC       | ~20             | Laboratory observations |

### Alias Library

- **Total aliases**: 350+
- **Abbreviations**: HTN, MI, CHF, COPD, DM, CVA, PE, DVT, UTI, etc.
- **Brand names**: Lipitor→Atorvastatin, Zoloft→Sertraline, Ozempic→Semaglutide
- **Synonyms**: heart attack→MI, shortness of breath→Dyspnea
- **Dental**: TMJ, SRP, RCT, caries, perio
- **Lab abbreviations**: CBC, BMP, CMP, TSH, A1c, BNP, PT/INR

## Resolution Strategies

### 1. Exact Match (confidence = 1.00)
Direct hit on case-folded preferred term in O(1) dictionary lookup.

### 2. Alias Match (confidence = 0.95)
Hit on any registered alias (abbreviation, synonym, brand name).

### 3. Fuzzy Match (confidence = similarity ratio)
Levenshtein-ratio scoring via `SequenceMatcher`. Catches typos and partial mentions.
- Default threshold: 0.80
- Returns ranked alternatives for ambiguous matches

## Type-Aware Filtering

When an entity_type is provided, the normalizer constrains matches to compatible ontology groups:

| NER Entity Type | Concept Group |
|----------------|---------------|
| DISEASE        | CONDITION     |
| SYMPTOM        | CONDITION     |
| MEDICATION     | MEDICATION    |
| DOSAGE         | MEDICATION    |
| PROCEDURE      | PROCEDURE     |
| TREATMENT      | PROCEDURE     |
| ANATOMY        | ANATOMY       |
| BODY_PART      | ANATOMY       |
| LAB_VALUE      | LAB           |
| TEST           | LAB           |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/normalize` | Single entity normalization |
| POST   | `/normalize/batch` | Batch (up to 500 entities) |
| GET    | `/normalize/lookup/{cui}` | Reverse CUI lookup |
| GET    | `/normalize/dictionary/stats` | Dictionary coverage stats |

## Performance Characteristics

- **Exact/alias lookup**: O(1), <0.1ms per entity
- **Fuzzy matching**: O(n) scan of dictionary, ~1-5ms per entity
- **Batch throughput**: 500 entities in <50ms (mostly exact/alias)
- **Memory**: ~50KB for dictionary + indices
- **Thread safety**: Read-only dictionary, safe for concurrent access

## Limitations

1. **Dictionary size** — Curated dictionary (~140 concepts) does not cover the full breadth of UMLS (>3M concepts). Best suited for common clinical entities.
2. **No contextual disambiguation** — "PE" could be Pulmonary Embolism or Physical Examination; resolved by first-registered-wins without context.
3. **Fuzzy matching precision** — Short abbreviations (2-3 chars) may produce false matches at lower similarity thresholds.
4. **No UMLS API integration** — All mappings are local; no live lookup against NLM APIs.
5. **English only** — All terms and aliases are in English.

## Ethical Considerations

- Concept normalization errors could lead to incorrect clinical coding
- Should be used as a decision-support tool, not as a definitive coding system
- All CUI/SNOMED/RxNorm/ICD mappings should be validated by qualified clinical coders
- The module does not store or transmit PHI

## References

- [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/)
- [SNOMED CT](https://www.snomed.org/)
- [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/)
- [ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)
- [LOINC](https://loinc.org/)
