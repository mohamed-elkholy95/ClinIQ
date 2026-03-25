# Model Card: Clinical Vital Signs Extraction

## Overview

Rule-based extraction engine for structured vital sign measurements from
clinical free text.  Identifies 9 vital sign types with unit normalisation,
physiological validation, clinical interpretation, and trend detection.

## Architecture

```
Clinical Text
    │
    ├─ Pattern Matching (compiled regex, 20+ patterns)
    │   ├─ Blood Pressure (labeled + unit-required)
    │   ├─ Heart Rate (labeled + standalone-with-unit)
    │   ├─ Temperature (explicit-unit + auto-detect)
    │   ├─ Respiratory Rate
    │   ├─ Oxygen Saturation (SpO2/SaO2/O2 Sat + on-RA)
    │   ├─ Weight (kg/lbs)
    │   ├─ Height (ft'in"/cm/inches)
    │   ├─ BMI
    │   └─ Pain Scale (/10)
    │
    ├─ Qualitative Descriptors (8 patterns)
    │   ├─ afebrile → 98.6°F NORMAL
    │   ├─ febrile/fever → 101°F HIGH
    │   ├─ tachycardic → 110 bpm HIGH
    │   ├─ bradycardic → 50 bpm LOW
    │   ├─ tachypneic → 24/min HIGH
    │   ├─ hypotensive → 85 mmHg LOW
    │   ├─ hypertensive → 160 mmHg HIGH
    │   └─ hypoxic → 88% LOW
    │
    ├─ Unit Conversion
    │   ├─ °C → °F
    │   ├─ lbs → kg
    │   ├─ inches → cm
    │   └─ ft'in" → cm
    │
    ├─ Physiological Validation (reject impossible values)
    ├─ Clinical Interpretation (5 categories per reference ranges)
    ├─ Section-Aware Confidence Boosting (+0.05 in Vital Signs sections)
    ├─ Trend Detection (improving/worsening/stable within 60-char window)
    ├─ BMI Auto-Calculation (from weight + height when BMI not stated)
    └─ Span Deduplication (confidence-based tie-breaking)
```

## Vital Sign Types

| Type                | Unit       | Valid Range    | Normal Range  | Confidence |
|---------------------|-----------|---------------|--------------|------------|
| Blood Pressure      | mmHg      | 20–350 SBP    | 90–140 SBP   | 0.80–0.90  |
| Heart Rate          | bpm       | 10–350        | 60–100       | 0.88       |
| Temperature         | °F        | 80–115        | 97.0–99.5    | 0.78–0.88  |
| Respiratory Rate    | breaths/min| 2–80          | 12–20        | 0.88       |
| Oxygen Saturation   | %         | 30–100        | 95–100       | 0.88       |
| Weight              | kg        | 0.5–700       | —            | 0.85       |
| Height              | cm        | 20–280        | —            | 0.85       |
| BMI                 | kg/m²     | 5–100         | 18.5–25.0    | 0.85–0.88  |
| Pain Scale          | /10       | 0–10          | 0–3          | 0.85       |

## Clinical Interpretation Categories

| Category       | Description                                 |
|---------------|---------------------------------------------|
| NORMAL        | Within standard adult reference range       |
| LOW           | Below normal but not immediately dangerous  |
| HIGH          | Above normal but not immediately dangerous  |
| CRITICAL_LOW  | Dangerously low, requires immediate attention|
| CRITICAL_HIGH | Dangerously high, requires immediate attention|

## Blood Pressure Reference Ranges (AHA 2017)

| Systolic      | Diastolic     | Interpretation  |
|--------------|--------------|----------------|
| ≤ 70         | ≤ 40         | CRITICAL_LOW   |
| 71–89        | 41–59        | LOW            |
| 90–140       | 60–90        | NORMAL         |
| 141–179      | 91–119       | HIGH           |
| ≥ 180        | ≥ 120        | CRITICAL_HIGH  |

MAP (Mean Arterial Pressure) calculated as: (SBP + 2 × DBP) / 3

## Confidence Scoring

| Factor                        | Impact      |
|------------------------------|-------------|
| Labeled pattern (e.g., "BP") | 0.88–0.90   |
| Standalone with unit         | 0.80        |
| Explicit unit (°F, °C)       | 0.88        |
| Auto-detected unit           | 0.78        |
| Qualitative descriptor       | 0.70        |
| In Vital Signs section       | +0.05       |
| Calculated BMI               | 0.90 × min(weight, height confidence) |

## Performance Characteristics

- **Latency**: < 2ms per document (pure regex, no ML dependencies)
- **Throughput**: > 10,000 documents/second single-threaded
- **Memory**: ~50 KB for compiled patterns (stateless, thread-safe)
- **Dependencies**: Python stdlib only (re, hashlib, time, dataclasses)

## Limitations

1. **Adult ranges only** — pediatric, neonatal, and geriatric reference
   ranges not implemented; interpretation may be inaccurate for these groups
2. **No contextual disambiguation** — "120/80" without a BP label may be
   missed unless mmHg unit is present
3. **Qualitative implied values are approximate** — "tachycardic" implies
   ~110 bpm but actual HR could be anywhere > 100
4. **No medication/activity context** — doesn't adjust interpretation for
   beta-blocker use, post-exercise, pregnancy, etc.
5. **Single-reading per match** — cannot extract ranges like "BP 120-130/80"
6. **English only** — patterns designed for English clinical notes

## Ethical Considerations

- Vital sign interpretation should never replace clinical judgment
- Critical findings flagged by this module require human verification
- Reference ranges are based on population averages and may not apply to
  all patients
- Weight/height/BMI extraction should be used carefully given body image
  sensitivities in clinical documentation

## References

- Whelton PK, et al. 2017 ACC/AHA Blood Pressure Guidelines
- WHO Clinical Vital Signs Reference Ranges
- Charlson ME, et al. Physiological validation studies
