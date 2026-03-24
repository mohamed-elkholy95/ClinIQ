# Model Card: ClinIQ ICD-10 Code Prediction

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ ICD-10 Predictor |
| **Version** | 1.0.0 |
| **Type** | Multi-label Text Classification |
| **Backends** | scikit-learn baseline, Transformer (ClinicalBERT), Hierarchical |
| **Framework** | scikit-learn, PyTorch, HuggingFace Transformers |
| **License** | MIT |
| **Last Updated** | 2026-03-24 |
| **Contact** | ClinIQ ML Team |

### Model Architecture

ClinIQ ICD-10 prediction uses three interchangeable classifier backends:

1. **scikit-learn Baseline** (`SklearnICDClassifier`): TF-IDF feature extraction via `ClinicalFeatureExtractor` (unigram + bigram, clinical abbreviation normalization, section-aware features) followed by a multi-label scikit-learn classifier. Suitable for deployment without GPU.

2. **Transformer** (`TransformerICDClassifier`): Fine-tuned ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) with multi-label classification head using sigmoid activation. Handles long documents via sliding window with overlapping chunks (stride = window_size / 2) and max-pooling aggregation across windows.

3. **Hierarchical** (`HierarchicalICDClassifier`): Two-stage approach that first predicts the ICD-10 chapter (21 chapters, A00-Z99), then runs chapter-specific classifiers for fine-grained code prediction. Reduces the label space at each stage.

All backends support:
- Configurable top-k prediction with confidence thresholds (minimum 0.1)
- Batch prediction for throughput optimization
- ICD-10 chapter mapping and code description lookup
- Contributing text segments identification

---

## Intended Use

### Primary Use Cases

- Suggesting ICD-10 diagnosis codes from clinical notes and discharge summaries to assist medical coders
- Pre-populating coding worksheets to reduce manual review time
- Auditing existing code assignments against clinical documentation
- Clinical research: automated cohort identification by diagnosis code

### Intended Users

- Medical coders and health information management professionals
- Clinical documentation improvement specialists
- Healthcare IT teams integrating automated coding into EHR workflows
- Clinical researchers building diagnosis-based cohorts

### Out-of-Scope Uses

- Autonomous billing code assignment without human review
- Primary source for clinical decision-making
- Coding for procedure codes (CPT/HCPCS) -- this model covers diagnosis codes only
- Processing non-English clinical documentation

---

## Training Approach

### scikit-learn Baseline

**Feature Engineering:**
- TF-IDF vectors (unigram + bigram, max 5,000 features, sublinear TF)
- Clinical feature extraction: section detection, abbreviation normalization, medication and lab value counts
- Feature matrix: sparse, typically 5,000-10,000 dimensions

**Classifier:**
- Multi-label binary relevance with per-label classifiers
- Sigmoid calibration for probability outputs
- Minimum confidence threshold of 0.1 to filter low-confidence predictions

### Transformer

**Base Model:**
- `emilyalsentzer/Bio_ClinicalBERT`: BERT-base architecture pre-trained on MIMIC-III clinical notes (2M+ notes)
- Fine-tuned with multi-label binary cross-entropy loss
- Max sequence length: 512 tokens

**Long Document Handling:**
- Documents exceeding 512 tokens are split into overlapping windows (stride = 206 tokens)
- Each window is classified independently
- Predictions are aggregated via max-pooling across windows

### Hierarchical

**Stage 1 -- Chapter Prediction:**
- Predicts top-3 ICD-10 chapters from the full document
- Uses the same architecture as the transformer backend but with 21 output labels

**Stage 2 -- Code Prediction:**
- Runs chapter-specific classifiers on the predicted chapters
- Each chapter classifier has a smaller label space (50-200 codes)
- Final predictions are merged and sorted by confidence

---

## Evaluation Metrics

### scikit-learn Baseline

| Metric | Value |
|--------|-------|
| F1 (micro) | 0.68 |
| F1 (macro) | 0.52 |
| Precision@5 | 0.71 |
| Precision@10 | 0.58 |
| Recall@10 | 0.64 |
| Hamming Loss | 0.018 |
| Subset Accuracy | 0.31 |

### Transformer (ClinicalBERT)

| Metric | Value |
|--------|-------|
| F1 (micro) | 0.76 |
| F1 (macro) | 0.61 |
| Precision@5 | 0.79 |
| Precision@10 | 0.65 |
| Recall@10 | 0.72 |
| Hamming Loss | 0.012 |
| Subset Accuracy | 0.38 |

### Hierarchical

| Metric | Value |
|--------|-------|
| F1 (micro) | 0.74 |
| F1 (macro) | 0.59 |
| Precision@5 | 0.77 |
| Precision@10 | 0.63 |
| Chapter Accuracy | 0.89 |

### Performance by ICD-10 Chapter (Transformer, F1 micro)

| Chapter | Description | F1 |
|---------|-------------|-----|
| I00-I99 | Circulatory system | 0.81 |
| E00-E89 | Endocrine/metabolic | 0.78 |
| J00-J99 | Respiratory system | 0.77 |
| C00-D49 | Neoplasms | 0.75 |
| K00-K95 | Digestive system | 0.73 |
| N00-N99 | Genitourinary system | 0.71 |
| M00-M99 | Musculoskeletal | 0.68 |
| F01-F99 | Mental/behavioral | 0.64 |
| R00-R99 | Symptoms/signs | 0.59 |

High-prevalence chapters (circulatory, endocrine, respiratory) achieve stronger performance due to greater training data representation.

---

## Ethical Considerations

### Bias and Fairness

- **Prevalence bias**: High-frequency codes (e.g., I10 Hypertension, E11 Type 2 Diabetes) have significantly better prediction performance than rare codes. This may systematically under-code rare diseases.
- **Demographic bias**: Training data from US hospital systems may not generalize to other healthcare contexts, coding conventions, or patient populations.
- **Documentation style**: Performance varies with documentation quality. Well-structured notes with clear assessment/plan sections yield better predictions than brief or fragmented notes.
- **Specialty bias**: Models trained primarily on inpatient discharge summaries may underperform on outpatient clinic notes or specialty-specific documentation.

### Financial Impact

- ICD-10 codes directly affect healthcare billing. Incorrect predictions, if accepted without review, could result in upcoding (billing for more complex conditions than documented) or undercoding (lost revenue and incomplete clinical records).
- This system is designed as a coding *assistant*, not an autonomous coder.

### Privacy

- The model processes clinical text that may contain Protected Health Information (PHI)
- All processing occurs within the deployment boundary
- No training data or inference inputs are transmitted externally
- Audit logging records processing metadata, not clinical content

---

## Known Limitations

1. **Rare code coverage**: Codes with fewer than 50 training examples have poor recall (F1 < 0.30). The long tail of ICD-10 (~70,000 codes) cannot be fully covered.
2. **Multi-code attribution**: When multiple conditions are described in the same sentence, the model may attribute a code to the wrong condition.
3. **Laterality and specificity**: The model may predict the correct base code but miss specificity digits (e.g., predicting E11.9 instead of E11.22).
4. **Negated conditions**: Conditions mentioned as ruled out (e.g., "No evidence of pneumonia") may still generate false-positive predictions. The NER negation detection helps but is not integrated into all ICD backends.
5. **Document length**: The sklearn baseline processes the full document as a single feature vector, losing positional information. The transformer backend truncates or chunks documents, potentially splitting relevant context.
6. **Temporal context**: The model does not distinguish between current, historical, and family history conditions.
7. **External cause codes**: V00-Y99 (external causes) and Z00-Z99 (health status factors) have lower performance due to their reliance on context that is often absent from clinical narratives.
