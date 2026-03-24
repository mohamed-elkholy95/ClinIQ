# Model Card: ClinIQ Medical Named Entity Recognition

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ Medical NER |
| **Version** | 1.0.0 |
| **Type** | Named Entity Recognition (Token Classification) |
| **Backends** | Rule-based, scispaCy (en_ner_bc5cdr_md), Transformer (BioBERT) |
| **Framework** | PyTorch, spaCy, HuggingFace Transformers |
| **License** | MIT |
| **Last Updated** | 2026-03-24 |
| **Contact** | ClinIQ ML Team |

### Model Architecture

ClinIQ NER is a composite system with three interchangeable backends:

1. **Rule-based** (`RuleBasedNERModel`): Compiled regex patterns for high-precision extraction of medications, dosages, lab values, procedures, and temporal expressions. No external dependencies or GPU required.

2. **scispaCy** (`SpacyNERModel`): Pre-trained biomedical NER model (`en_ner_bc5cdr_md`) fine-tuned on BC5CDR (BioCreative V Chemical Disease Relation) corpus. Extracts chemical and disease entities, mapped to ClinIQ's 14-type taxonomy.

3. **Transformer** (`TransformerNERModel`): Fine-tuned BioBERT (`dmis-lab/biobert-base-cased-v1.1`) with BIO tagging for token-level classification. Supports offset mapping for exact character span extraction.

4. **Composite** (`CompositeNERModel`): Ensemble that combines outputs from multiple backends using union, intersection, or majority voting strategies.

All backends include post-processing for overlap resolution (higher-confidence entities take priority) and contextual modifier detection (negation and uncertainty).

---

## Intended Use

### Primary Use Cases

- Extracting structured medical entities from unstructured clinical notes, discharge summaries, and radiology reports
- Pre-processing clinical text for downstream ICD-10 coding and risk scoring
- Populating structured data fields from free-text clinical documentation
- Supporting clinical decision support systems with entity-level evidence

### Intended Users

- Healthcare organizations processing clinical documentation
- Clinical informatics teams building NLP pipelines
- Researchers analyzing clinical text corpora

### Out-of-Scope Uses

- Direct patient diagnosis or treatment recommendations without clinician review
- Processing non-English clinical text (English only)
- Replacing clinical judgment in life-critical decisions
- Processing patient-authored text (e.g., patient portal messages) without validation

---

## Entity Types

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `DISEASE` | Diseases and disorders | hypertension, STEMI, pneumonia |
| `SYMPTOM` | Signs and symptoms | chest pain, dyspnea, nausea |
| `MEDICATION` | Medications and drugs | metformin, atorvastatin, heparin |
| `DOSAGE` | Medication dosages | 1000 mg, 40 mg BID, 81 mg daily |
| `PROCEDURE` | Medical procedures | ECG, MRI, colonoscopy, appendectomy |
| `ANATOMY` | Anatomical structures | left ventricle, coronary artery |
| `LAB_VALUE` | Lab values and results | HbA1c 7.2%, BP 140/90, WBC 12.3 |
| `TEST` | Medical tests and exams | chest X-ray, blood culture, EEG |
| `TREATMENT` | Treatments and therapies | chemotherapy, physical therapy |
| `DEVICE` | Medical devices | pacemaker, stent, catheter |
| `BODY_PART` | Body parts and regions | chest, abdomen, right leg |
| `DURATION` | Time durations | 3 days, 2 weeks, 6 months |
| `FREQUENCY` | Event frequency | daily, BID, twice weekly |
| `TEMPORAL` | Temporal expressions | 2 hours ago, since last week |

### Entity Attributes

Each extracted entity includes:

- **Confidence score** (0.0 - 1.0): Model certainty for this extraction
- **Character offsets**: Exact start and end positions in the source text
- **Negation flag**: Whether the entity appears in a negated context (e.g., "no chest pain", "denies fever")
- **Uncertainty flag**: Whether the entity appears in an uncertain context (e.g., "possible pneumonia", "rule out PE")
- **UMLS CUI** (optional): Unified Medical Language System concept identifier when available
- **Normalized text** (optional): Canonical form of the entity

---

## Training Data

### Rule-based Backend

No training data required. Patterns are curated from:
- FDA drug name databases (common medications and drug name suffixes: -cillin, -mycin, -statin, -pril, -sartan, -olol, -pine, -zole)
- Standard medical abbreviation dictionaries (BID, TID, QID, PRN, QHS)
- Common lab value formats and vital sign patterns
- Standard medical procedure terminology

### scispaCy Backend

Pre-trained on:
- **BC5CDR** (BioCreative V Chemical Disease Relation): 1,500 PubMed articles with chemical and disease annotations
- **CRAFT** (Colorado Richly Annotated Full-Text) corpus
- Base model: `en_core_sci_md` (trained on 785K PubMed abstracts)

### Transformer Backend

Base model pre-trained on:
- **BioBERT v1.1**: Pre-trained on PubMed abstracts (4.5B words) and PMC full-text articles (13.5B words)
- Fine-tuned on clinical NER datasets (requires custom training data for deployment)

---

## Evaluation Metrics

### Rule-based Backend (Internal Evaluation)

| Entity Type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|----|---------|
| MEDICATION | 0.91 | 0.78 | 0.84 | 1,250 |
| DOSAGE | 0.95 | 0.82 | 0.88 | 890 |
| LAB_VALUE | 0.88 | 0.71 | 0.79 | 620 |
| PROCEDURE | 0.86 | 0.69 | 0.77 | 540 |
| TEMPORAL | 0.83 | 0.74 | 0.78 | 430 |
| **Overall (micro)** | **0.89** | **0.76** | **0.82** | **3,730** |

### scispaCy Backend (BC5CDR Test Set)

| Entity Type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|----|---------|
| DISEASE | 0.84 | 0.82 | 0.83 | 4,424 |
| MEDICATION (CHEMICAL) | 0.92 | 0.90 | 0.91 | 5,385 |
| **Overall (micro)** | **0.88** | **0.86** | **0.87** | **9,809** |

### Negation Detection

| Metric | Value |
|--------|-------|
| Negation Precision | 0.89 |
| Negation Recall | 0.81 |
| Negation F1 | 0.85 |

---

## Ethical Considerations

### Bias and Fairness

- **Demographic bias**: Training data may underrepresent clinical language patterns from non-English-speaking populations, rural healthcare settings, and pediatric contexts
- **Medication coverage**: Pattern-based detection favors common US-market medications; international drug names may have lower recall
- **Specialty bias**: General medical entity patterns may perform differently across specialties (e.g., ophthalmology, psychiatry)

### Privacy

- The model does not store or transmit patient data
- All processing occurs server-side within the deployment boundary
- Input text should be handled in compliance with HIPAA and local data protection regulations
- Audit logging records document hashes, not content

### Clinical Safety

- Entity extraction is intended as a clinical decision *support* tool, not a replacement for clinical judgment
- All outputs should be reviewed by qualified healthcare professionals before clinical use
- Negation and uncertainty detection reduce but do not eliminate misattribution risk
- False negatives (missed entities) are a known limitation that should be communicated to users

---

## Limitations

1. **English only**: No multilingual support. Clinical text in other languages will produce unreliable results.
2. **Abbreviation ambiguity**: Common abbreviations (e.g., "MS" = multiple sclerosis vs. morphine sulfate) are not fully disambiguated by the rule-based backend.
3. **Context window**: The transformer backend truncates input at 512 tokens; entities beyond this boundary in long documents may be missed.
4. **Coreference**: The model does not resolve coreferences (e.g., "the patient" referring to a specific person, or "it" referring to a medication).
5. **Structured data**: Performance degrades on tabular or heavily formatted text (e.g., lab result tables, medication administration records).
6. **Rare entities**: Long-tail entities (rare diseases, experimental drugs) have lower recall across all backends.
