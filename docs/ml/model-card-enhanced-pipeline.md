# Model Card — Enhanced Clinical Pipeline

## Overview

The **Enhanced Clinical Pipeline** is a unified orchestration layer that integrates all 14+ ClinIQ clinical NLP modules into a single analysis call. It processes clinical documents through two phases: pre-processing (structural analysis) and extraction/scoring (clinical information extraction).

## Architecture

```
Input Text
    │
    ├─── Phase 1: Pre-processing ──────────────────────────────────────┐
    │    ├── Document Classification (13 types)                       │
    │    ├── Section Parsing (35 categories)                          │
    │    ├── Quality Analysis (5 dimensions)                          │
    │    ├── De-identification (18 HIPAA categories) [optional]       │
    │    └── Abbreviation Expansion (230+ entries)                    │
    │                                                                 │
    ├─── Phase 2: Extraction & Scoring ────────────────────────────────┐
    │    ├── Medication Extraction (220+ drugs)                        │
    │    ├── Allergy Extraction (150+ allergens)                       │
    │    ├── Vital Signs Extraction (9 types)                          │
    │    ├── Temporal Extraction (dates, durations, frequencies)        │
    │    ├── Assertion Detection (6 statuses) [needs NER entities]     │
    │    ├── Concept Normalization (UMLS/SNOMED/RxNorm) [needs NER]    │
    │    ├── SDoH Extraction (8 domains)                               │
    │    ├── Relation Extraction (12 types) [needs NER entities]       │
    │    └── Charlson Comorbidity Index (17 categories)                │
    │                                                                   │
    └─── EnhancedPipelineResult (JSON-serializable) ───────────────────┘
```

## Integrated Modules

| # | Module | Input | Output | Dependencies |
|---|--------|-------|--------|--------------|
| 1 | Document Classifier | text | type + confidence | None |
| 2 | Section Parser | text | sections with offsets | None |
| 3 | Quality Analyzer | text | score, grade, findings | None |
| 4 | De-identifier | text | redacted text + entities | None |
| 5 | Abbreviation Expander | text | expanded text + matches | None |
| 6 | Medication Extractor | text | structured medications | None |
| 7 | Allergy Extractor | text | allergens + reactions | None |
| 8 | Vital Signs Extractor | text | measurements + interpretation | None |
| 9 | Temporal Extractor | text | dates, durations, frequencies | None |
| 10 | Assertion Detector | text + entities | assertion statuses | NER entities |
| 11 | Concept Normalizer | entities | CUI, SNOMED, RxNorm codes | NER entities |
| 12 | SDoH Extractor | text | social factors + Z-codes | None |
| 13 | Relation Extractor | text + entities | semantic relations | NER entities |
| 14 | Charlson Calculator | text + ICD codes | CCI score + mortality | Optional ICD codes |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze/enhanced` | Single document analysis with all modules |
| POST | `/analyze/enhanced/batch` | Batch analysis (up to 20 documents) |
| GET | `/analyze/enhanced/modules` | Module catalogue with descriptions |

## Configuration

All modules are independently toggleable via request body:

```json
{
  "text": "Clinical note text...",
  "enable_classification": true,
  "enable_sections": true,
  "enable_quality": true,
  "enable_deidentification": false,
  "enable_abbreviations": true,
  "enable_medications": true,
  "enable_allergies": true,
  "enable_vitals": true,
  "enable_temporal": true,
  "enable_assertions": true,
  "enable_normalization": true,
  "enable_sdoh": true,
  "enable_relations": true,
  "enable_comorbidity": true
}
```

## Design Decisions

1. **Composition over inheritance** — Wraps `ClinicalPipeline` via delegation rather than subclassing
2. **Fault isolation** — Each module's failure is captured in `component_errors` without aborting remaining stages
3. **Lazy initialization** — Modules are imported and instantiated on first use
4. **De-identification off by default** — It modifies text (destructive operation)
5. **Consistent serialization** — All results are dict-based for direct JSON serialization

## Performance

- **Typical latency**: 10–50ms per document (all rule-based modules)
- **Zero ML dependencies** for rule-based modules
- **Memory**: ~50MB for all module dictionaries and patterns

## Limitations

1. Assertion detection, concept normalization, and relation extraction require NER entities from the base pipeline — they produce empty results without a loaded NER model
2. De-identification uses rule-based patterns; may miss novel PHI formats
3. No cross-module optimization (each module processes text independently)
4. Batch processing is sequential (not parallelized)

## Test Coverage

- 60 unit tests for the pipeline module
- 19 API route tests
- 79 total new tests
- Full test suite: 2806 passing, 0 failures
