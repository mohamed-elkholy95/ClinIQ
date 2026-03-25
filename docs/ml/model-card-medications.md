# Model Card — Clinical Medication Extraction

## Overview

Structured medication extraction module that parses free-text clinical notes into normalized, machine-readable medication components following HL7 FHIR MedicationStatement conventions.

## Architecture

```
Clinical text ─► SectionDetector ──────────┐
             ├► DrugNameExtractor ──────────┤
             ├► DosageExtractor ────────────┤
             ├► RouteExtractor ─────────────┼─► MedicationExtractionResult
             ├► FrequencyExtractor ─────────┤
             ├► DurationExtractor ──────────┤
             ├► IndicationExtractor ────────┤
             └► StatusDetector ─────────────┘
```

### Dual-Path Design

| Variant | Implementation | Latency | Dependencies |
|---------|---------------|---------|-------------|
| Rule-Based | Compiled regex + drug dictionary | <5ms | None |
| Transformer | HuggingFace token-classification | ~50ms | transformers, torch |

The transformer variant automatically falls back to rule-based on load failure.

## Drug Dictionary

**220+ entries** covering:

| Specialty | Example Medications | Count |
|-----------|-------------------|-------|
| Cardiovascular | lisinopril, metoprolol, warfarin, atorvastatin | ~50 |
| Endocrine/Diabetes | metformin, insulin, semaglutide, levothyroxine | ~25 |
| Psychiatry/CNS | sertraline, quetiapine, gabapentin, zolpidem | ~35 |
| Pain/Analgesia | acetaminophen, ibuprofen, oxycodone, tramadol | ~25 |
| Pulmonary | albuterol, tiotropium, fluticasone, montelukast | ~12 |
| GI | omeprazole, ondansetron, docusate, lactulose | ~16 |
| Antibiotics | amoxicillin, azithromycin, vancomycin, ciprofloxacin | ~25 |
| Dental | lidocaine, articaine, chlorhexidine, fluoride | ~12 |
| Other | aspirin, enoxaparin, allopurinol, tamsulosin | ~20 |

### Brand-to-Generic Normalization

Examples:
- Lipitor → atorvastatin
- Zoloft → sertraline
- Glucophage → metformin
- Nexium → esomeprazole
- Ozempic → semaglutide

## Extracted Components

### 1. Dosage
- **Value + Unit**: "500 mg", "2 puffs", "30 units"
- **Range doses**: "1-2 tablets" → value=1.0, value_high=2.0
- **Decimal**: "0.125 mg"
- **Supported units**: mg, mcg, g, mL, units, IU, mEq, tablets, capsules, puffs, drops, patches

### 2. Route of Administration (15 routes)
| Route | Patterns |
|-------|----------|
| PO (oral) | "by mouth", "orally", "PO", "p.o." |
| IV | "intravenous", "IV", "i.v." |
| IM | "intramuscular", "IM" |
| SQ | "subcutaneous", "SQ", "subq" |
| SL | "sublingual", "SL" |
| Topical | "topically", "applied to skin" |
| Inhaled | "inhaled", "via inhaler", "MDI" |
| PR | "rectal", "per rectum" |
| Transdermal | "patch", "transdermal" |
| Nebulized | "nebulizer", "neb" |
| Ophthalmic | "eye drops", "ophthalmic" |
| Otic | "ear drops", "otic" |
| Nasal | "nasal spray", "intranasal" |
| Vaginal | "vaginal", "PV" |

### 3. Frequency
| Pattern | Normalized |
|---------|-----------|
| QD, daily, once daily | daily |
| BID, twice daily | BID |
| TID, three times daily | TID |
| QID, four times daily | QID |
| q6h, every 6 hours | q6h |
| qhs, at bedtime, nightly | at bedtime |
| weekly, once a week | weekly |
| PRN, as needed | (sets PRN flag) |
| STAT | STAT |
| AC, with meals | with meals |

### 4. Duration
- "for 10 days", "x 7 days", "× 5 days"
- "for 2 weeks", "for 3 months"
- Range: "5-7 days"

### 5. Indication
- "for pain", "for hypertension", "for diabetes"
- Extracted from "for [indication]" patterns

### 6. Status
| Status | Trigger Patterns |
|--------|-----------------|
| Active | "continue", "maintain", "currently", "taking" |
| Discontinued | "discontinued", "d/c'd", "stopped" |
| Held | "hold", "on hold", "withheld" |
| New | "started", "initiated", "begin" |
| Changed | "increased", "decreased", "adjusted", "titrated" |
| Allergic | "allergic", "allergy", "adverse reaction" |

## Confidence Scoring

```
Score = base + dosage + route + frequency + section + brand_bonus

base           = 0.50  (drug dictionary match)
dosage         = 0.15  (dosage pattern found)
route          = 0.10  (route detected)
frequency      = 0.10  (frequency detected)
section_header = 0.10  (inside "Medications:" section)
brand_bonus    = 0.05  (brand→generic mapping exists)

Maximum = 1.00
```

## Section Header Detection

The extractor recognizes medication section headers for confidence boosting:
- "Medications:", "Current Medications:", "Home Medications:"
- "Discharge Medications:", "Admission Medications:"
- "Pre-op Meds:", "Post-op Meds:"
- "Drug List:", "Rx:", "Prescriptions:"

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/medications` | POST | Single text extraction |
| `/medications/batch` | POST | Batch (up to 50 texts) |
| `/medications/lookup/{name}` | GET | Drug dictionary lookup |
| `/medications/dictionary/stats` | GET | Dictionary statistics |

## Limitations

1. **Dictionary-dependent**: Only detects drugs in the curated dictionary (~220 entries). Rare or newly-approved medications may be missed.
2. **Context window**: Attributes (dosage, route) are extracted from a 120-character window after the drug name. Attributes mentioned earlier or far later may be missed.
3. **Cross-line contamination**: In densely packed medication lists, status signals from adjacent lines may bleed into a medication's context.
4. **No drug interaction detection**: Does not identify drug-drug interactions.
5. **No allergy cross-referencing**: Allergy status is detected textually but not cross-referenced against prescribed medications.
6. **English only**: Patterns are designed for English clinical text.

## Test Coverage

- 88 unit tests for the extraction module
- 19 API endpoint tests
- 107 total tests, 0 failures
