# Model Card: Clinical Note Quality Analyzer

## Overview

The Clinical Note Quality Analyzer evaluates clinical notes across five quality dimensions before they enter the NLP inference pipeline.  It produces a composite score (0–100), a letter grade, per-dimension breakdowns, and actionable recommendations.

## Architecture

```
Raw Clinical Note
       │
       ▼
┌──────────────────────┐
│  Pre-compute Stats   │  word count, sentence count, section detection,
│                      │  abbreviation ratio, medical term ratio
└──────────┬───────────┘
           │
     ┌─────┼─────┬─────────┬─────────┐
     ▼     ▼     ▼         ▼         ▼
┌────────┐┌──────┐┌─────────┐┌───────┐┌──────────┐
│Complete-││Read- ││Structure││Info   ││Consisten-│
│ness    ││abil- ││         ││Density││cy        │
│Scorer  ││ity   ││         ││       ││          │
└────┬───┘└──┬───┘└────┬────┘└───┬───┘└────┬─────┘
     │       │         │         │          │
     └───────┴─────────┴─────────┴──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Weighted Composite │  Σ(score × weight)
            │  Score + Grade      │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Recommendations    │  sorted by severity
            │  Generation         │
            └─────────────────────┘
```

## Quality Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Completeness** | 0.20 | Word count threshold, expected section coverage (CC, HPI, Assessment, Plan), section count bonus |
| **Readability** | 0.20 | Sentence length distribution, abbreviation density, very long/short sentences |
| **Structure** | 0.20 | Section header presence and standardisation, whitespace ratio, list usage, line length variance |
| **Information Density** | 0.20 | Medical term concentration (suffix/prefix patterns), numeric measurement density |
| **Consistency** | 0.20 | Duplicate paragraph detection, contradictory assertion modifiers |

## Scoring

- **Overall score**: 0–100, weighted sum of dimension scores
- **Grade**: A (≥90), B (≥80), C (≥70), D (≥60), F (<60)
- Each dimension scorer starts at a base score and applies penalties/bonuses based on findings

### Finding Severities

| Severity | Description |
|----------|-------------|
| **Critical** | Major quality issue likely to impact NLP accuracy |
| **Warning** | Moderate issue that may reduce extraction quality |
| **Info** | Informational observation, no action required |

## Expected Sections

Default expected sections (configurable):
- Chief Complaint
- History of Present Illness
- Assessment
- Plan

35+ known clinical section headers recognised including dental-specific sections (Dental History, Periodontal Assessment, Oral Examination).

## Performance

- **Latency**: < 5ms for typical clinical notes (no ML dependencies)
- **Dependencies**: Zero — pure regex and statistics-based heuristics
- **Thread safety**: Stateless analysis, safe for concurrent use

## Limitations

1. **Heuristic-based** — Does not use ML models; may miss subtle quality issues
2. **English only** — Section detection and abbreviation patterns are English-specific
3. **Note type agnostic** — Default expected sections are general; deployers should configure `expected_sections` for specialty-specific notes
4. **No semantic understanding** — Cannot detect clinically incorrect content, only structural and statistical quality signals
5. **Abbreviation dictionary** — Covers ~60 common clinical abbreviations; rarer abbreviations may not be counted

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/quality` | Analyze single note quality |
| POST | `/quality/batch` | Analyze up to 100 notes with aggregate summary |
| GET | `/quality/dimensions` | List quality dimensions with descriptions |

## References

- Joint Commission standards for clinical documentation quality
- AHIMA best practices for health record documentation
- i2b2/n2c2 clinical NLP shared task annotation guidelines
