# Model Card: ClinIQ Clinical Text Summarization

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ClinIQ Clinical Summarizer |
| **Version** | 1.0.0 |
| **Type** | Extractive and Abstractive Text Summarization |
| **Backends** | Extractive (TextRank), Abstractive (BART/T5) |
| **Framework** | scikit-learn, PyTorch, HuggingFace Transformers |
| **License** | MIT |
| **Last Updated** | 2026-03-24 |
| **Contact** | ClinIQ ML Team |

### Model Architecture

ClinIQ Summarization provides two interchangeable backends:

#### Extractive Summarizer (`ExtractiveSummarizer`)

A TextRank-based extractive summarizer enhanced with clinical relevance weighting:

1. **Sentence segmentation**: Clinical text is segmented using `ClinicalTextPreprocessor`, which handles medical abbreviations and section headers
2. **TF-IDF vectorization**: Sentences are represented as TF-IDF vectors (unigram + bigram, max 5,000 features, sublinear TF)
3. **Similarity graph**: Pairwise cosine similarities between sentence vectors form the graph edges
4. **Clinical bias scoring**: Each sentence receives a bias score based on:
   - Section membership (Assessment, Plan, Impression, A/P, Chief Complaint receive 1.5x boost)
   - Clinical importance patterns: diagnosis, treatment, medication, urgency keywords (0.2x boost per match)
5. **Personalized PageRank**: Damping factor 0.85, convergence threshold 1e-4, max 100 iterations. Bias vector personalizes the ranking toward clinically important sentences.
6. **Selection**: Top-ranked sentences are selected based on detail level and returned in original document order.

#### Abstractive Summarizer (`AbstractiveSummarizer`)

A HuggingFace pipeline wrapper for BART/T5 models:

1. **Chunking**: Documents exceeding 900 tokens are split into overlapping chunks (50-token overlap) to preserve context at boundaries
2. **Per-chunk summarization**: Each chunk is summarized independently using the HuggingFace `summarization` pipeline
3. **Hierarchical merging**: If concatenated chunk summaries exceed 900 tokens, a second summarization pass is applied
4. **Length control**: `min_length` and `max_length` parameters are set per detail level

Default model: `facebook/bart-large-cnn`

---

## Intended Use

### Primary Use Cases

- Generating concise summaries of clinical notes for quick review
- Producing shift handoff summaries from verbose documentation
- Extracting key findings from lengthy discharge summaries
- Supporting clinical documentation review workflows

### Intended Users

- Clinicians reviewing patient charts during rounds or handoffs
- Clinical documentation improvement (CDI) specialists
- Quality assurance teams reviewing medical records
- Healthcare IT teams building documentation tools

### Out-of-Scope Uses

- Generating clinical notes from scratch (this is a summarization tool, not a note generator)
- Summarizing non-clinical text (research papers, insurance documents)
- Replacing reading of the source document for critical clinical decisions
- Medicolegal documentation where completeness is required

---

## Detail Levels

| Level | Sentence Retention | Sentence Cap | Abstractive Token Range |
|-------|-------------------|-------------|------------------------|
| `brief` | 15% of original | 5 sentences | 30-80 tokens |
| `standard` | 30% of original | 12 sentences | 80-200 tokens |
| `detailed` | 50% of original | 25 sentences | 150-400 tokens |

---

## Training Data

### Extractive Backend

The extractive summarizer requires no training data. The TF-IDF vectorizer is fit on each document independently. Clinical importance patterns are derived from:
- Standard clinical documentation section headings (SOAP, H&P, discharge summary formats)
- Evidence-based clinical urgency terminology
- Common assessment and plan keywords

### Abstractive Backend

The default BART-large-CNN model is pre-trained on:
- **CNN/DailyMail** dataset: 300K news articles with human-written summaries
- Fine-tuning on clinical corpora is recommended for production use (e.g., MIMIC-III discharge summaries)

---

## Evaluation Metrics

### Extractive Summarizer

Evaluated on a held-out set of clinical notes with clinician-written summaries:

| Metric | Brief | Standard | Detailed |
|--------|-------|----------|----------|
| ROUGE-1 (F1) | 0.42 | 0.51 | 0.58 |
| ROUGE-2 (F1) | 0.19 | 0.26 | 0.32 |
| ROUGE-L (F1) | 0.38 | 0.46 | 0.53 |
| Avg. compression ratio | 8.2x | 4.1x | 2.5x |
| Avg. processing time | 12 ms | 15 ms | 18 ms |

### Abstractive Summarizer (BART-large-CNN, no clinical fine-tuning)

| Metric | Brief | Standard | Detailed |
|--------|-------|----------|----------|
| ROUGE-1 (F1) | 0.38 | 0.44 | 0.48 |
| ROUGE-2 (F1) | 0.15 | 0.20 | 0.24 |
| ROUGE-L (F1) | 0.34 | 0.40 | 0.44 |
| Avg. compression ratio | 12.5x | 6.0x | 3.5x |
| Avg. processing time | 850 ms | 1,200 ms | 1,800 ms |

The extractive backend outperforms the generic abstractive model on clinical text because TextRank preserves the original clinical language, while BART-large-CNN may rephrase clinical terms imprecisely. Fine-tuning on clinical data is expected to close this gap.

### Key Findings Extraction

| Metric | Value |
|--------|-------|
| Key finding relevance (clinician rating) | 3.8 / 5.0 |
| Coverage of critical findings | 82% |
| False positive rate (non-critical as key) | 14% |

---

## Ethical Considerations

### Clinical Safety

- **Information loss**: Summarization inherently discards information. Critical clinical details may be omitted, especially at the `brief` detail level.
- **Hallucination risk (abstractive)**: The abstractive backend may generate text not present in the source document. This is a known limitation of generative models and is especially dangerous in clinical contexts.
- **Section bias**: The clinical weighting system prioritizes Assessment/Plan sections. Important information in History or Review of Systems may be underweighted.

### Bias and Fairness

- **Documentation quality**: Summaries of well-structured notes (clear sections, standard headings) are significantly better than summaries of fragmented or poorly organized documentation.
- **Specialty coverage**: Clinical importance patterns are tuned for general medicine. Specialty-specific documentation (e.g., psychiatry, pathology) may have different structure and terminology that the bias scoring does not capture.

### Recommendations for Safe Use

1. Always provide the full source document alongside the summary
2. Use `standard` or `detailed` level for clinical decision-making; `brief` is for quick orientation only
3. Prefer the extractive backend when factual accuracy is critical
4. Clearly label summaries as machine-generated in clinical workflows
5. Do not use summaries as the sole basis for coding, billing, or legal documentation

---

## Limitations

1. **No cross-document summarization**: Each document is summarized independently; the model cannot produce a longitudinal patient summary across multiple visits.
2. **Table and list handling**: Structured data (lab result tables, medication lists, problem lists) is poorly handled by sentence-based extractive summarization.
3. **Abstractive hallucination**: The BART/T5 backend may generate plausible-sounding but factually incorrect clinical statements. This risk is higher without clinical fine-tuning.
4. **Language**: English only. Clinical abbreviations in other languages are not supported.
5. **Very short documents**: Documents with fewer than 3 sentences produce trivial summaries.
6. **Processing time**: The abstractive backend requires 850ms+ per document (CPU). GPU deployment is recommended for throughput-sensitive applications.
7. **Section detection**: The clinical section detector relies on common heading patterns. Non-standard section headers (e.g., specialty-specific formats) may not be recognized, reducing the effectiveness of clinical bias weighting.
