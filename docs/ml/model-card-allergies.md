# Model Card: Clinical Allergy Extractor

## Overview

Rule-based extraction engine that identifies drug, food, and environmental allergens from clinical free text with associated reaction detection, severity classification, NKDA recognition, and negation handling.

## Architecture

```
Input Text
    │
    ├─ NKDA Detection (NKDA/NKA/NKFA/"no known allergies"/etc.)
    │
    ├─ Allergy Section Detection (header-bounded confidence boosting)
    │
    ├─ Allergen Matching (compiled regex, ~150 entries, longest-first)
    │   │
    │   ├─ Drug (80+ entries: antibiotics, NSAIDs, opioids, CV, psych, etc.)
    │   ├─ Food (15 entries: peanuts, shellfish, dairy, eggs, etc.)
    │   └─ Environmental (10 entries: pollen, dust, latex, bee stings, etc.)
    │
    ├─ Negation/Toleration Detection (60-char prefix window)
    │
    ├─ Reaction Detection (30+ patterns within same line)
    │   └─ Severity Modifier Override (severe/life-threatening/mild/moderate)
    │
    ├─ Severity Classification (max of reaction severities)
    │
    └─ Deduplication (highest confidence per canonical allergen)
    │
    ▼
ExtractionResult (allergies + NKDA status + metadata)
```

## Allergen Dictionary

| Category | Count | Examples |
|----------|-------|---------|
| Drug | 80+ | Penicillin (PCN), Sulfonamides (sulfa), Aspirin (ASA), Codeine, Iodine contrast |
| Food | 15 | Peanuts, Shellfish, Tree nuts, Dairy (milk), Eggs, Wheat/Gluten |
| Environmental | 10 | Pollen, Dust mites, Latex, Bee stings, Animal dander, Mold |

**Total surface forms**: ~250 (including aliases and brand names)

## Reaction Detection (30+ patterns)

| Severity | Reactions |
|----------|-----------|
| Life-threatening | Anaphylaxis, anaphylactic shock, airway compromise, laryngeal edema |
| Severe | Angioedema, bronchospasm, Stevens-Johnson syndrome, hypotension, dyspnea |
| Moderate | Urticaria, hives, rash, pruritus, swelling, edema, erythema |
| Mild | Nausea, vomiting, diarrhea, GI upset, headache, dizziness, flushing |

## NKDA Patterns

Recognises: NKDA, NKA, NKFA, "no known drug allergies", "no known allergies", "denies allergies", "no drug allergies", "no medication allergies"

## Confidence Scoring

| Factor | Boost |
|--------|-------|
| Base detection | 0.70 |
| Drug allergen | +0.05 |
| Reaction found | +0.10 |
| Inside allergy section | +0.10 |
| **Maximum** | **1.00** |

## Severity Classification

Four levels: **mild** → **moderate** → **severe** → **life-threatening**

Determined by: (1) reaction type inherent severity, (2) explicit severity modifiers in context ("severe rash"), (3) max across multiple reactions.

## Assertion Status

| Status | Triggers |
|--------|----------|
| Active | Default for all detected allergies |
| Tolerated | "tolerates", "not allergic to", "no adverse reaction to" |
| Historical | "previously", "childhood", "resolved", "outgrown" |

## Performance

- **Latency**: <3 ms per document
- **Dependencies**: Zero ML dependencies
- **Thread safety**: Fully stateless

## Limitations

1. Rule-based — cannot detect novel or misspelled allergens not in dictionary
2. Reaction window is line-bounded, may miss reactions on separate lines
3. Cannot distinguish allergy from adverse drug reaction (ADR) clinically
4. Food allergens may produce false positives in dietary/nutrition notes
5. Does not validate clinical plausibility of allergen-reaction pairs
6. Not designed for non-English clinical documents

## Ethical Considerations

- Allergy information is safety-critical; false negatives could lead to adverse drug events
- This module should supplement, not replace, structured allergy documentation
- NKDA detection may conflict with detected allergens in contradictory notes
- Should be validated against institutional allergy documentation standards
- Must not be used as sole source of allergy information for prescribing decisions
