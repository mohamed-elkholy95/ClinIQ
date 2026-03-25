# Model Card: Clinical Relation Extraction

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | ClinIQ Relation Extractor |
| **Version** | 1.0.0 |
| **Type** | Rule-based pattern matching + optional transformer classifier |
| **Task** | Semantic relation extraction between medical entities |
| **Output** | Directed relations with type, confidence, and evidence |
| **Inference Time** | < 1 ms per entity pair (rule-based) |
| **Dependencies** | None (rule-based); transformers, torch (transformer) |

## Description

Identifies semantic relationships between pairs of medical entities in
clinical text.  Given pre-extracted entities with character offsets, the
module determines how entities relate to each other — whether a medication
treats a disease, a test diagnoses a condition, a symptom is a side effect
of a drug, etc.

## Relation Types (12)

| Relation | Direction | Example |
|---|---|---|
| `treats` | MEDICATION → DISEASE/SYMPTOM | "metformin for diabetes" |
| `causes` | DISEASE/PROCEDURE → SYMPTOM | "pneumonia causing fever" |
| `diagnoses` | TEST → DISEASE | "CT scan revealed pneumonia" |
| `contraindicates` | DISEASE → MEDICATION | "renal failure, avoid metformin" |
| `administered_for` | PROCEDURE → DISEASE | "CABG for coronary artery disease" |
| `dosage_of` | DOSAGE → MEDICATION | "500mg metformin" |
| `location_of` | ANATOMY → DISEASE/PROCEDURE | "pain in the left knee" |
| `result_of` | LAB_VALUE → TEST | "Hgb 7.2 on CBC" |
| `worsens` | DISEASE → DISEASE/SYMPTOM | "diabetes worsened by infection" |
| `prevents` | MEDICATION → DISEASE | "aspirin for prevention of CVD" |
| `monitors` | TEST → DISEASE/MEDICATION | "INR to monitor warfarin" |
| `side_effect_of` | SYMPTOM → MEDICATION | "nausea from methotrexate" |

## Architecture

```
Entities ─► Sort by position
         ─► Generate candidate pairs (within max_distance)
         ─► Filter by type compatibility (RELATION_TYPE_CONSTRAINTS)
         ─► Match intervening text against pattern library
         ─► Apply confidence bonuses (proximity, co-sentence)
         ─► De-duplicate (highest confidence wins per pair+type)
         ─► Sort by confidence descending
```

## Pattern Library

Each relation type has 2–3 regex patterns with base confidence scores:

- **Treat patterns**: "for", "to treat", "prescribed for", "started on",
  "managed with" (base 0.75–0.85)
- **Cause patterns**: "caused by", "due to", "secondary to", "leads to",
  "associated with" (base 0.60–0.85)
- **Diagnose patterns**: "revealed", "showed", "positive for", "confirmed",
  "consistent with" (base 0.70–0.85)

## Confidence Scoring

```
final_confidence = min(1.0, base_confidence + proximity_bonus + sentence_bonus)

proximity_bonus = max(0.0, 0.1 × (1 - distance / 150))
sentence_bonus  = 0.05 if no period between entities, else 0.0
```

## Type Constraints

Each relation type specifies which (subject_type, object_type) pairs are
valid.  Invalid type combinations are rejected before pattern matching,
preventing nonsensical relations like "DOSAGE treats ANATOMY".

## Transformer Variant

An optional `TransformerRelationExtractor` wraps a fine-tuned HuggingFace
sequence-classification model trained on entity-pair contexts:

- **Input format**: `[CLS] subject [SEP] context [SEP] object [CLS]`
- **Max length**: 128 tokens
- **Fallback**: Automatically uses rule-based extraction if model loading fails

## Limitations

- **Pattern coverage**: Only detects relations matching known syntactic
  patterns; novel phrasings may be missed.
- **Windowed pairs only**: Entities more than `max_distance` characters
  apart are not evaluated.
- **No cross-sentence relations**: Relations spanning multiple sentences
  receive no co-sentence bonus and may have lower confidence.
- **No negation awareness**: Does not distinguish "metformin treats
  diabetes" from "metformin does not treat diabetes" (relies on upstream
  entity-level negation detection).
- **English only**: All patterns are English-language.

## Ethical Considerations

- Extracted relations are for **informational support**, not clinical
  decision-making.
- False positive relations (e.g. incorrectly linking a medication to a
  disease) could mislead if not reviewed by a clinician.
- The system may exhibit bias toward common clinical phrasings and
  standard terminology.
