# Model Card: Clinical Section Parser

## Overview

The Clinical Section Parser is a rule-based engine that segments clinical free-text documents into their constituent sections by detecting header patterns and mapping them to a taxonomy of 35+ clinical section categories.

## Architecture

```
Input Text
    │
    ├─ Strategy 1: Colon Headers (HEADER:, Title Case:, **Bold**)  → confidence 1.0
    ├─ Strategy 2: ALL-CAPS Lines (≤60 chars, no colon)            → confidence 0.85
    └─ Strategy 3: Numbered Headers (1. Header, 2) Header)         → confidence 0.80
    │
    ▼
Candidate Deduplication (position-based, keep highest confidence)
    │
    ▼
Header Normalisation → Category Lookup (~60 canonical mappings)
    │
    ▼
SectionParseResult (sections with character offsets + categories)
```

## Section Categories (35)

| Category | Description |
|----------|-------------|
| chief_complaint | Primary reason for encounter |
| history_of_present_illness | Current illness narrative |
| past_medical_history | Prior medical conditions |
| past_surgical_history | Prior surgical procedures |
| family_history | Family medical history |
| social_history | Social/lifestyle factors |
| review_of_systems | Systematic symptom review |
| medications | Current medications |
| allergies | Drug/food/environmental allergies |
| vital_signs | Vital sign measurements |
| physical_exam | Examination findings |
| assessment | Clinical assessment |
| plan | Treatment plan |
| assessment_and_plan | Combined A&P |
| laboratory | Lab results |
| imaging | Radiological findings |
| procedures | Procedures performed |
| hospital_course | Hospitalisation events |
| discharge_medications | Discharge meds |
| discharge_instructions | Discharge instructions |
| discharge_diagnosis | Final diagnoses |
| follow_up | Follow-up care |
| operative_findings | Surgical findings |
| dental_history | Dental/oral history |
| periodontal_assessment | Periodontal exam |
| oral_examination | Intraoral exam |
| pertinent_negatives | Relevant negatives |
| pertinent_positives | Relevant positives |
| immunizations | Vaccination history |
| problem_list | Active diagnoses |
| reason_for_visit | Encounter reason |
| subjective | SOAP S |
| objective | SOAP O |
| recommendations | Clinical recommendations |
| addendum | Note addendum |

## Header Dictionary

~60 canonical header names mapped to categories, including:
- Standard headers: "Chief Complaint", "HPI", "PMH", "Medications"
- Abbreviations: "CC", "FH", "SH", "ROS", "A/P", "VS", "PE"
- Dental: "Dental History", "Periodontal Assessment", "Oral Examination"
- Discharge: "Discharge Medications", "Discharge Instructions"
- SOAP: "Subjective", "Objective"

## Performance

- **Latency**: <1 ms per document
- **Dependencies**: Zero ML dependencies (pure regex/string matching)
- **Thread safety**: Fully stateless, safe for concurrent use

## Integration Points

Used by downstream modules for section-aware confidence boosting:
- Vital signs extractor (+0.05 in Vital Signs section)
- SDoH extractor (+0.05 in Social History section)
- Medication extractor (+0.10 in Medications section)
- Assertion detector (section-level assertion overrides)
- Allergy extractor (+0.10 in Allergies section)

## Limitations

1. Requires well-formatted clinical notes with recognisable headers
2. Unstructured narratives without headers produce no sections
3. Ambiguous short abbreviations (e.g., "PE") may conflict with content
4. Does not handle nested sub-sections (e.g., ROS organ systems)
5. Not designed for non-English clinical documents

## Ethical Considerations

- Section parsing is a structural analysis tool and does not interpret clinical content
- Incorrect section boundaries could cause downstream modules to apply wrong confidence adjustments
- Should be validated against institution-specific note templates before clinical deployment
