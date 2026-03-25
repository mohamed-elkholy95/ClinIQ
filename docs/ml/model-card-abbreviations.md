# Model Card: Clinical Abbreviation Expansion

## Overview

Rule-based clinical abbreviation detection and expansion engine that identifies
medical abbreviations in clinical free text and expands them to full forms.
Features context-aware disambiguation for ambiguous abbreviations with multiple
possible meanings.

## Architecture

```
Input Text
    │
    ├─→ Word Boundary Scan (compiled regex patterns)
    │       │
    │       ├─→ Unambiguous Match → Dictionary Lookup → Confidence from DB
    │       │
    │       └─→ Ambiguous Match → Disambiguation Pipeline
    │               │
    │               ├─→ 1. Section Header Override (highest priority)
    │               ├─→ 2. Context Keyword Matching (±50 char window)
    │               └─→ 3. Default First Sense (fallback)
    │
    ├─→ Overlapping Span Deduplication (confidence tie-breaking)
    ├─→ Confidence Threshold Filtering
    └─→ Expanded Text Generation ("abbrev (expansion)" format)
```

## Dictionary Coverage

| Domain | Unambiguous | Ambiguous Entries | Total |
|--------|------------|-------------------|-------|
| Cardiology | ~30 | — | ~30 |
| Pulmonology | ~18 | 3 senses | ~18 |
| Endocrine | ~12 | — | ~12 |
| Neurology | ~13 | 3 senses | ~13 |
| Gastroenterology | ~15 | — | ~15 |
| Renal | ~10 | 2 senses | ~10 |
| Infectious | ~10 | — | ~10 |
| Musculoskeletal | ~7 | 2 senses | ~7 |
| Hematology | ~15 | 1 sense | ~15 |
| General | ~35 | 4 senses | ~35 |
| Dental | ~14 | 1 sense | ~14 |
| Pharmacy | ~40 | 2 senses | ~40 |
| **Total** | **~220** | **~10 entries (25 senses)** | **~230** |

## Ambiguous Abbreviations

| Abbreviation | Possible Meanings | Disambiguation Method |
|-------------|-------------------|----------------------|
| PE | Pulmonary embolism, Physical exam | Context keywords + section headers |
| PT | Patient, Physical therapy, Prothrombin time | Context keywords + section headers |
| MS | Multiple sclerosis, Mental status, Morphine sulfate | Context keywords |
| OR | Operating room | Context keywords |
| CR | Creatinine | Context keywords + section headers |
| CAP | Community-acquired pneumonia, Capsule | Context keywords + section headers |
| RA | Rheumatoid arthritis, Room air | Context keywords |
| PD | Peritoneal dialysis, Probing depth | Context keywords + section headers |
| ED | Emergency department, Erectile dysfunction | Context keywords |

## Disambiguation Strategies

### 1. Section Header Override (Priority 1)
When an abbreviation appears under a known section header, the section
context determines the meaning:
- **Physical Exam:** PE → physical exam
- **Labs/Laboratory:** PT → prothrombin time, CR → creatinine
- **Medications:** CAP → capsule
- **Periodontal:** PD → probing depth

### 2. Context Keyword Matching (Priority 2)
Examines ±50 characters around the abbreviation for domain-specific
keywords. Each sense has 7–13 keywords. More keyword matches → higher
confidence (0.70 + 0.05 per match, capped at 0.90).

### 3. Default Sense (Priority 3)
Falls back to the first listed sense at confidence 0.60 when no context
signals are present.

## Confidence Scoring

| Match Type | Confidence Range |
|-----------|-----------------|
| Unambiguous (from dictionary) | 0.70 – 0.95 |
| Ambiguous: Section resolved | 0.90 |
| Ambiguous: Context resolved | 0.70 – 0.90 |
| Ambiguous: Default sense | 0.60 |

## Performance Characteristics

- **Processing time:** <5ms per document (typical clinical note)
- **Dependencies:** Zero ML dependencies, pure regex/dictionary
- **Memory:** ~50KB for compiled patterns and dictionary

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/abbreviations` | Expand abbreviations in single document |
| POST | `/abbreviations/batch` | Batch expansion (up to 50 documents) |
| GET | `/abbreviations/lookup/{abbrev}` | Look up a specific abbreviation |
| GET | `/abbreviations/dictionary/stats` | Dictionary coverage statistics |
| GET | `/abbreviations/domains` | List 12 clinical domains |

## Limitations

1. **English only** — US clinical English abbreviations; no support for other languages
2. **Context window fixed** — ±50 characters may miss relevant signals in long sentences
3. **No learning** — Dictionary is static; cannot learn new abbreviations from data
4. **Ambiguity coverage** — Only ~10 ambiguous entries; many clinical abbreviations have context-dependent meanings not yet modeled
5. **No negation awareness** — Does not distinguish "no PE" from "PE diagnosed"
6. **Specialty bias** — General medical focus; specialty-specific abbreviations (ophthalmology, dermatology) are underrepresented

## Ethical Considerations

- Abbreviation expansion is a **preprocessing aid**, not a clinical decision tool
- Incorrect disambiguation of ambiguous abbreviations could mislead downstream NLP
- Should be validated against institution-specific abbreviation conventions
- Does not replace clinical judgment in interpreting notes

## References

- Xu, H., et al. (2007). "A study of abbreviations in clinical notes." AMIA Annual Symposium.
- Moon, S., et al. (2014). "A sense inventory for clinical abbreviations and acronyms." AMIA Annual Symposium.
- Wu, Y., et al. (2017). "Clinical abbreviation disambiguation using neural word embeddings." BioNLP Workshop.
