# Model Card: Dental-Specific NER Module

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ Dental NER |
| **Version** | 1.0.0 |
| **Task** | Dental-specific named entity recognition and periodontal risk assessment |
| **Architecture** | Rule-based pattern matching + periodontal risk scoring |
| **Input** | Dental clinical note text |
| **Output** | Dental entities (tooth numbers, surfaces, procedures, measurements) + risk assessment |

## Architecture

```
┌──────────────────────────────────────────────┐
│              Input Dental Note                │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │     DentalNERModel          │
    │                             │
    │  ┌────────────────────┐     │
    │  │ Tooth Number       │     │  Universal (1–32)
    │  │ Extraction         │     │  FDI (11–48)
    │  │                    │     │  Primary (A–T)
    │  └────────┬───────────┘     │
    │           │                 │
    │  ┌────────▼───────────┐     │
    │  │ Surface Detection  │     │  M, D, B, L, O, I, F, P
    │  │ (12 surfaces)      │     │  MB, ML, DB, DL
    │  └────────┬───────────┘     │
    │           │                 │
    │  ┌────────▼───────────┐     │
    │  │ Procedure          │     │  CDT code mapping
    │  │ Extraction         │     │  40+ procedure patterns
    │  └────────┬───────────┘     │
    │           │                 │
    │  ┌────────▼───────────┐     │
    │  │ Measurement        │     │  Probing depths, CAL,
    │  │ Extraction         │     │  recession, mobility
    │  └────────┬───────────┘     │
    │           │                 │
    │  ┌────────▼───────────┐     │
    │  │ Negation Detection │     │  "no", "without",
    │  │                    │     │  "negative for"
    │  └────────────────────┘     │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  PeriodontalRiskAssessor    │
    │                             │
    │  Probing depth scoring      │
    │  Attachment loss grading    │
    │  Bleeding/mobility factors  │
    │  Risk level classification  │
    │  CDT code recommendations   │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  DentalAssessment           │
    │  entities, risk_score,      │
    │  cdt_codes, recommendations │
    └─────────────────────────────┘
```

## Tooth Numbering Systems

| System | Range | Description |
|--------|-------|-------------|
| Universal | 1–32 | US/ADA system; 1 = upper-right third molar |
| FDI | 11–48 | ISO/international two-digit system |
| Primary | A–T | Primary (deciduous) teeth |

The module detects numbering system from context and supports cross-system references.

## Dental Surfaces (12)

| Code | Surface | Code | Surface |
|------|---------|------|---------|
| M | Mesial | MB | Mesiobuccal |
| D | Distal | ML | Mesiolingual |
| B | Buccal | DB | Distobuccal |
| L | Lingual | DL | Distolingual |
| O | Occlusal | F | Facial |
| I | Incisal | P | Palatal |

Multi-surface notations (e.g., "MOD", "DO") are parsed into individual surface components.

## Procedure Categories

| Category | Example Patterns | CDT Range |
|----------|-----------------|-----------|
| Restorative | amalgam, composite, crown, inlay, onlay | D2000–D2999 |
| Endodontic | root canal, pulpotomy, apicoectomy | D3000–D3999 |
| Periodontic | SRP, scaling, osseous surgery, graft | D4000–D4999 |
| Prosthodontic | denture, bridge, implant, pontic | D5000–D6999 |
| Oral Surgery | extraction, biopsy, alveoloplasty | D7000–D7999 |
| Orthodontic | brackets, archwire, retainer | D8000–D8999 |
| Preventive | prophylaxis, sealant, fluoride | D1000–D1999 |
| Diagnostic | exam, radiograph, bitewing, panoramic | D0100–D0999 |

## Periodontal Risk Assessment

### Risk Factors and Scoring

| Factor | Low (0–1) | Moderate (2–3) | High (4–5) |
|--------|-----------|----------------|------------|
| Probing Depth | ≤ 3 mm | 4–5 mm | ≥ 6 mm |
| Attachment Loss | ≤ 2 mm | 3–4 mm | ≥ 5 mm |
| Bleeding on Probing | < 10% sites | 10–30% sites | > 30% sites |
| Tooth Mobility | Grade 0 | Grade I | Grade II–III |
| Furcation | None | Class I | Class II–III |

### Risk Levels

| Level | Score Range | Interpretation |
|-------|------------|----------------|
| Low | 0.0–0.30 | Healthy periodontium, routine maintenance |
| Moderate | 0.31–0.60 | Early/moderate periodontitis, closer monitoring |
| High | 0.61–0.80 | Advanced periodontitis, treatment planning needed |
| Critical | 0.81–1.00 | Severe disease, immediate intervention recommended |

## Limitations

1. **Rule-based only**: No ML-based dental NER; relies on pattern matching which may miss unusual phrasing
2. **English only**: Dental terminology patterns are English-language specific
3. **Abbreviation ambiguity**: Some dental abbreviations overlap with medical ones (e.g., "RCT" = root canal therapy vs. randomised controlled trial)
4. **Quadrant context**: Quadrant assignment requires explicit mention; implicit context from surrounding text is not inferred
5. **Periodontal scoring**: Risk assessment uses heuristic scoring, not a validated periodontal risk calculator

## Ethical Considerations

- Dental risk scores are assistive tools — they do not replace clinical periodontal examination
- CDT code suggestions are for documentation assistance, not for billing submission without clinician verification
- Tooth numbering conversion between systems should be verified for complex cases

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-24 | Initial release with tooth/surface/procedure extraction and periodontal risk assessment |
