# Model Card: Clinical Temporal Information Extraction

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ Temporal Extractor |
| **Version** | 1.0.0 |
| **Task** | Temporal expression extraction and normalisation from clinical text |
| **Architecture** | Rule-based regex pattern matching with configurable reference date |
| **Input** | Clinical document text + optional reference date |
| **Output** | Extracted dates, durations, frequencies, ages, and temporal relations |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Input Clinical Text                        │
│                   + Reference Date (optional)                │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   Temporal Expression Engine   │
         │                               │
         │  ┌─────────────────────┐      │
         │  │ Date Extraction     │      │  4 format patterns
         │  │ (absolute dates)    │      │  MM/DD/YYYY, ISO,
         │  │                     │      │  Month DD YYYY, etc.
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Duration Extraction │      │  Simple: "3 days"
         │  │                     │      │  Range: "3 to 5 days"
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Relative Time       │      │  "3 days ago",
         │  │ Resolution          │      │  "yesterday",
         │  │                     │      │  "last week"
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Age Extraction      │      │  With 130-year
         │  │                     │      │  sanity check
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ POD Extraction      │      │  "POD #3",
         │  │ (post-op days)      │      │  "post-op day 5"
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Frequency           │      │  40+ clinical
         │  │ Normalisation       │      │  abbreviations
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Temporal Relation   │      │  before, after,
         │  │ Signal Detection    │      │  during, simultaneous
         │  └─────────┬───────────┘      │
         │            │                  │
         │  ┌─────────▼───────────┐      │
         │  │ Overlap             │      │  Confidence-based
         │  │ Deduplication       │      │  longest-match
         │  └─────────────────────┘      │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  TemporalExtractionResult     │
         │  dates, durations, relatives, │
         │  ages, frequencies, relations │
         └───────────────────────────────┘
```

## Extraction Categories

### Date Formats (4 patterns)

| Format | Example | Regex Pattern |
|--------|---------|---------------|
| US format | 03/25/2026 | `MM/DD/YYYY` |
| ISO format | 2026-03-25 | `YYYY-MM-DD` |
| Written (US) | March 25, 2026 | `Month DD, YYYY` |
| Written (EU) | 25 March 2026 | `DD Month YYYY` |

### Duration Types

| Type | Example | Output |
|------|---------|--------|
| Simple | "3 days" | `{value: 3, unit: "days"}` |
| Range | "3 to 5 days" | `{min: 3, max: 5, unit: "days"}` |
| Supported units | days, weeks, months, years, hours, minutes | — |

### Relative Time Expressions

| Pattern | Example | Resolution |
|---------|---------|------------|
| N units ago | "3 days ago" | reference_date − 3 days |
| Yesterday | "yesterday" | reference_date − 1 day |
| Last {period} | "last week" | reference_date − 7 days |
| Today | "today" | reference_date |
| Tomorrow | "tomorrow" | reference_date + 1 day |

### Clinical Frequency Abbreviations (40+)

| Category | Abbreviations |
|----------|--------------|
| Standard dosing | QD, BID, TID, QID |
| Hourly intervals | q2h, q4h, q6h, q8h, q12h, q24h, q48h, q72h |
| Conditional | PRN, STAT, AC, PC, HS |
| Written forms | "once daily", "twice daily", "every N hours" |

### Temporal Relation Signals

| Relation | Example Patterns |
|----------|-----------------|
| BEFORE | "prior to", "before", "preceding" |
| AFTER | "after", "following", "subsequent to", "post" |
| DURING | "during", "while", "throughout", "concurrent" |
| SIMULTANEOUS | "at the same time", "simultaneously", "concomitant" |

## Special Features

### Age Extraction
- Patterns: "67-year-old", "age 67", "67 y/o"
- Sanity check: ages > 130 are rejected as likely false positives

### Postoperative Day Extraction
- Patterns: "POD #3", "POD3", "post-op day 5", "postoperative day 2"
- Used for surgical timeline reconstruction

### Overlap Deduplication
- When multiple extractors match the same text span, the longest match wins
- Ties broken by confidence score

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/temporal` | Extract temporal information from text |
| `GET`  | `/temporal/frequency-map` | Get catalogue of 40+ frequency abbreviations |

## Limitations

1. **English only**: All patterns target English clinical text
2. **Ambiguous dates**: "03/04/2026" is parsed as MM/DD/YYYY (US convention); DD/MM/YYYY format not supported
3. **Implicit references**: "next Tuesday" or "in two weeks" relative to discharge date require explicit reference_date parameter
4. **Complex temporal reasoning**: Does not build full temporal graphs or resolve event ordering across multiple sentences
5. **Non-standard abbreviations**: Facility-specific abbreviations not in the 40+ standard set will not be normalised

## Ethical Considerations

- Temporal extraction is used for patient timeline reconstruction — errors could affect clinical decision support
- Dates extracted may constitute PHI; apply de-identification before sharing extracted temporal data
- Always pair with clinical review when temporal information drives treatment decisions

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-25 | Initial release with 4 date formats, durations, relative times, frequencies, temporal relations |
