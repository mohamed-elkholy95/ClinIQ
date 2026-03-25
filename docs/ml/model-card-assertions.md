# Model Card — Clinical Assertion Detection

## Overview

Clinical assertion detection module that classifies the contextual status
of medical entities (conditions, symptoms, medications, procedures) extracted
from clinical text. Determines whether a concept is **present**, **absent**
(negated), **possible** (uncertain), **conditional**, **hypothetical** (future),
or attributed to **family history**.

## Architecture

```
Clinical Text + Entity Span
         │
         ▼
┌─────────────────────┐
│ Sentence Segmentation│  ← Regex-based, handles clinical formatting
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Section Detection   │  ← "Family History:", "Plan:", "Pertinent Negatives:"
│  (ConText only)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────┐
│      Trigger Matching Pipeline       │
│                                      │
│  1. Pseudo-trigger check (block FP) │
│  2. Pre-entity triggers (→ entity)  │
│  3. Post-entity triggers (entity →) │
│  4. Scope terminator check          │
│  5. Priority-based selection        │
│  6. Confidence calculation          │
└─────────────────┬───────────────────┘
                  │
                  ▼
         AssertionResult
         (status, confidence, trigger, sentence)
```

## Assertion Statuses

| Status | Description | Example |
|--------|-------------|---------|
| `present` | Condition is affirmed | "Patient has diabetes" |
| `absent` | Condition is negated | "No evidence of malignancy" |
| `possible` | Uncertain/suspected | "Rule out pulmonary embolism" |
| `conditional` | Depends on future event | "If symptoms worsen, start antibiotics" |
| `hypothetical` | Planned/future | "Will start metformin tomorrow" |
| `family` | Attributed to family | "Family history of breast cancer" |

## Trigger Library (97 total)

### Pre-entity Triggers (70 patterns)

**Negation (24 patterns)**:
`no`, `not`, `denies`, `without`, `negative for`, `absence of`, `no signs of`,
`no evidence of`, `free of`, `resolved`, `no further`, `no acute/new/significant`,
`no longer has`, `failed to reveal`, `not demonstrate`, `no radiographic evidence of`,
`ruled out`, `was not`, `are not`, `does not have/show`, `no history of`,
`never had`, `unremarkable`

**Uncertainty (20 patterns)**:
`possible`, `probable`, `likely`, `suspected`, `suspicious for`, `may have`,
`might be`, `concern for`, `rule out`, `r/o`, `questionable`, `uncertain`,
`equivocal`, `indeterminate`, `cannot be excluded`, `differential includes`,
`suggestive of`, `consistent with`, `clinically consistent with`, `appears to be`

**Family (8 patterns)**:
`family history of`, `familial`, `mother/father/sibling had/with`,
`maternal/paternal history`, `FH:`, `family hx of`, `inherited`, `hereditary`

**Hypothetical (12 patterns)**:
`will start/begin/need`, `plan to/for`, `planned`, `scheduled for`,
`to be started/given`, `should be started`, `consider starting`,
`if patient develops`, `would recommend`, `pending`, `awaiting`, `follow-up with`

**Conditional (6 patterns)**:
`if symptoms worsen/persist/recur`, `should symptoms worsen`,
`in the event of`, `unless`, `provided that`, `as needed for`

### Post-entity Triggers (9 patterns)

**Negation**: `was absent/negative/ruled out`, `has been ruled out`,
`has resolved`, `was not seen/found/detected`, `is unlikely`

**Uncertainty**: `is suspected/questionable`, `cannot be excluded`, `?`

**Family**: `runs in the family`

### Pseudo Triggers (7 patterns — prevent false positives)

`no increase/decrease/change in`, `not causing`, `not only/just`,
`no doubt/question`, `not necessarily`, `gram negative`, `not certain if`

### Scope Terminators (11 patterns)

`but`, `however`, `yet`, `though`, `although`, `aside from`,
`except for`, `other than`, `which`, `that is/was`, `;` / `:`

## Confidence Scoring

```
confidence = base (0.90)
           + priority × 0.02
           - (distance / 10) × 0.01
clamped to [0.50, 1.00]
```

- **Priority bonus**: Higher-priority triggers (more specific patterns) get higher confidence
- **Distance penalty**: Triggers farther from the entity are penalized
- **Default (no trigger)**: 0.80 confidence for PRESENT status

## Section-Aware Detection (ConTextAssertionDetector)

The extended ConText detector recognizes clinical section headers and
applies section-level assertion overrides:

| Section Header | Implied Assertion |
|---------------|-------------------|
| `Family History:` / `FH:` | FAMILY |
| `Pertinent Negatives:` | ABSENT |
| `Plan:` / `Assessment and Plan:` / `A/P:` | HYPOTHETICAL |

**Override rules**:
- Section headers only override **default (PRESENT)** status
- Explicit triggers always take precedence over section context
- A new section header cancels the previous section's context

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Latency (single entity) | < 0.1ms |
| Latency (batch, 50 entities) | < 2ms |
| Memory footprint | ~200KB (compiled patterns) |
| ML dependencies | None (pure regex) |
| Thread safety | Yes (instance-level state only) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/assertions` | POST | Single entity assertion detection |
| `/assertions/batch` | POST | Batch detection (up to 200 entities) |
| `/assertions/statuses` | GET | List all 6 assertion status types |
| `/assertions/stats` | GET | Detection statistics and trigger count |

## Limitations

1. **No coreference resolution**: "The patient's mother has diabetes. She also has hypertension." — "She" referent is not resolved
2. **Scope heuristics**: Trigger scope is character-distance-based, not syntactic; complex sentences may cause errors
3. **No discourse-level reasoning**: Multi-sentence negation scope is not tracked
4. **Limited conditional detection**: Only recognizes common conditional phrasing patterns
5. **English only**: Trigger library is English-specific
6. **No learning**: Rule-based; does not adapt to domain-specific usage patterns

## References

- Harkema H, et al. "ConText: An algorithm for determining negation, experiencer, and temporal status from clinical reports." *J Biomed Inform.* 2009;42(5):839-851.
- Chapman WW, et al. "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries." *J Biomed Inform.* 2001;34(5):301-310.
- Uzuner Ö, et al. "2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text." *JAMIA.* 2011;18(5):552-556.
