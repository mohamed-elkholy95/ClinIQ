# Model Card: Social Determinants of Health (SDoH) Extractor

## Model Overview

| Field | Value |
|---|---|
| **Name** | ClinIQ SDoH Extractor |
| **Version** | 1.0.0 |
| **Type** | Rule-based pattern matching |
| **Task** | Social determinant factor extraction from clinical text |
| **Domains** | 8 (Housing, Employment, Education, Food Security, Transportation, Social Support, Substance Use, Financial) |
| **Dependencies** | Python `re` (stdlib only, zero ML dependencies) |
| **Latency** | <5ms per document |

## Architecture

```
Clinical Text
     │
     ▼
┌─────────────────────────┐
│  Section Detection       │──── Social History header identification
│  (regex header matching) │     with scope termination
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Pattern Matching        │──── 100+ compiled regex triggers
│  (per-domain scan)       │     across 8 SDoH domains
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Negation Detection      │──── 20+ negation cues in
│  (context window scan)   │     configurable character window
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Sentiment Assignment    │──── Adverse / Protective / Neutral
│  (with negation flip)    │     based on trigger + context
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Confidence Scoring      │──── Base confidence ± section boost
│  + Deduplication         │     ± negation penalty
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Z-Code Mapping          │──── ICD-10-CM Z55–Z65 assignment
│  + Result Assembly       │     per domain
└──────────────────────────┘
```

## Domain Coverage

### 8 SDoH Domains

| Domain | Adverse Triggers | Protective Triggers | Z-Code Range | Example Finding |
|---|---|---|---|---|
| **Housing** | 10 | 3 | Z59.00–Z59.81 | "currently homeless" |
| **Employment** | 7 | 3 | Z56.0–Z56.9 | "currently unemployed" |
| **Education** | 6 | 2 | Z55.0–Z55.9 | "limited English proficiency" |
| **Food Security** | 7 | 1 | Z59.41–Z59.48 | "food insecurity" |
| **Transportation** | 4 | 1 | Z59.82 | "no reliable transportation" |
| **Social Support** | 9 | 3 | Z60.2–Z65.3 | "socially isolated" |
| **Substance Use** | 15 | 6 | Z71.41–Z72.0 | "current smoker, 1 PPD" |
| **Financial** | 9 | 0 | Z59.5–Z59.7 | "cannot afford medication" |

### Sentiment Classification

| Sentiment | Description | Example |
|---|---|---|
| **Adverse** | Risk factor present | "Patient is currently homeless" |
| **Protective** | Social strength present | "Strong family support network" |
| **Neutral** | Informational, unclear direction | "Lives alone", "Retired" |

### Negation-Aware Sentiment Flipping

When a negation cue precedes a trigger:
- Adverse → Protective (e.g., "denies homelessness" → protective)
- Protective → Adverse (e.g., "no stable housing" → adverse)
- Confidence reduced by 15% for inferred sentiment

### Negation Cues (20+)

`no`, `not`, `never`, `denies`, `denied`, `without`, `negative for`,
`no history of`, `no hx of`, `does not`, `has not`, `hasn't`, `doesn't`,
`none`, `absent`, `quit`, `stopped`, `former`, `previously`

## Confidence Scoring

| Component | Effect |
|---|---|
| **Base confidence** | 0.72–0.92 per trigger (specificity-dependent) |
| **Section boost** | +0.05 when inside Social History section |
| **Negation penalty** | ×0.85 when sentiment is flipped |
| **Min threshold** | Configurable (default 0.50) |

## ICD-10-CM Z-Code Mapping

Each domain maps to validated Z-codes from the Z55–Z65 range:

- **Z55** — Problems related to education and literacy
- **Z56** — Problems related to employment and unemployment
- **Z59** — Problems related to housing and economic circumstances
- **Z60** — Problems related to social environment
- **Z63** — Problems in primary support group
- **Z65** — Problems related to other psychosocial circumstances
- **Z71** — Persons encountering health services for counseling
- **Z72** — Problems related to lifestyle

## Section-Aware Detection

The extractor identifies Social History sections via header patterns:
- `SOCIAL HISTORY:`, `Social History:`, `SH:`
- `PSYCHOSOCIAL`, `BEHAVIORAL HEALTH`
- `SUBSTANCE USE HISTORY`, `SOCIAL DETERMINANTS`

Section terminators: `FAMILY HISTORY`, `ROS`, `PHYSICAL EXAM`,
`ASSESSMENT`, `PLAN`, `MEDICATIONS`, `ALLERGIES`, `PMH`

Matches inside social history sections receive a configurable confidence
boost (default +0.05).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/sdoh` | Single document SDoH extraction |
| POST | `/sdoh/batch` | Batch extraction (up to 50 documents) |
| GET | `/sdoh/domains` | List all 8 domains with metadata |
| GET | `/sdoh/domains/{name}` | Detail for specific domain |
| GET | `/sdoh/z-codes` | Full Z-code catalogue |

## Performance Characteristics

- **Latency**: <5ms per document (rule-based, no ML inference)
- **Memory**: ~1MB (compiled regex library)
- **Thread safety**: Stateless extractor, safe for concurrent use
- **Batch processing**: Linear scaling, no overhead

## Limitations

1. **Pattern coverage**: Only detects phrasings present in the trigger
   library; novel or unusual documentation styles may be missed
2. **No coreference resolution**: Cannot link pronouns to entities
   (e.g., "She is homeless" if the subject is established earlier)
3. **Limited context understanding**: Cannot disambiguate complex
   scenarios requiring reasoning
4. **English only**: All patterns are English-language
5. **Section detection**: Relies on common header patterns; non-standard
   section labels may not be detected
6. **No severity grading**: Detects presence but does not grade severity
   (e.g., moderate vs. severe food insecurity)

## Ethical Considerations

- SDoH data is sensitive; findings should be used for care improvement,
  not discrimination
- False positives in adverse findings could lead to inappropriate
  interventions or stigma
- The extractor should augment — not replace — clinical judgment
- Z-code suggestions are advisory; clinicians should verify before
  documenting

## References

1. Healthy People 2030: Social Determinants of Health.
   https://health.gov/healthypeople/priority-areas/social-determinants-health
2. WHO Commission on Social Determinants of Health (2008).
3. ICD-10-CM Z55–Z65: Persons with potential health hazards related to
   socioeconomic and psychosocial circumstances.
4. Patra BG, et al. "Extracting social determinants of health from
   electronic health records using NLP." J Am Med Inform Assoc. 2021.
5. Bompelli A, et al. "Social determinants of health in electronic
   health records and their impact on health outcomes." NPJ Digital
   Medicine. 2021.
