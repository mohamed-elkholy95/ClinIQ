# Model Card: Evaluation & Benchmarking Framework

## Overview

The ClinIQ evaluation framework provides standardised metrics for assessing clinical NLP model performance across all pipeline components. Unlike general-purpose evaluation libraries, these metrics are tailored for clinical NLP challenges: extreme class imbalance in ICD-10 coding, strict span requirements for entity extraction, and the critical importance of calibrated confidence scores in clinical decision support.

## Metrics Catalogue

| Metric | Endpoint | Use Case | Range |
|--------|----------|----------|-------|
| Cohen's Kappa | `/evaluate/agreement` | Inter-annotator agreement | −1 to 1 |
| MCC | `/evaluate/classification` | Balanced binary classification | −1 to 1 |
| ECE | `/evaluate/classification` | Confidence calibration | 0 to 1 |
| Brier Score | `/evaluate/classification` | Probability accuracy | 0 to 1 |
| Partial NER F1 | `/evaluate/ner` | Entity extraction with span credit | 0 to 1 |
| ROUGE-1/2/L | `/evaluate/rouge` | Summarisation quality (P/R/F1) | 0 to 1 |
| Hierarchical ICD | `/evaluate/icd` | ICD-10 code prediction at 3 levels | 0 to 1 |
| AUPRC | `/evaluate/auprc` | Imbalanced binary classification | 0 to 1 |

## Architecture

```
┌────────────────────────────────────────────────┐
│                REST API Layer                   │
│  POST /evaluate/{classification,agreement,      │
│       ner,rouge,icd,auprc}                     │
│  GET  /evaluate/metrics                        │
├────────────────────────────────────────────────┤
│           Pydantic Request Validation           │
├────────────────────────────────────────────────┤
│        advanced_metrics.py (core logic)         │
│  ┌──────────┬──────────┬──────────────────┐    │
│  │  Kappa   │   MCC    │  Calibration     │    │
│  ├──────────┼──────────┼──────────────────┤    │
│  │Partial   │  ROUGE   │ Hierarchical ICD │    │
│  │NER Match │  P/R/F1  │ (chapter/block)  │    │
│  ├──────────┴──────────┴──────────────────┤    │
│  │            AUPRC (trapezoidal)         │    │
│  └────────────────────────────────────────┘    │
├────────────────────────────────────────────────┤
│         numpy (only external dependency)        │
└────────────────────────────────────────────────┘
```

## Design Decisions

### Why Cohen's Kappa over raw agreement?
Raw inter-annotator agreement inflates scores when one label dominates (e.g., "absent" entities). Kappa corrects for chance agreement and is the standard metric for i2b2/n2c2 clinical NLP shared tasks.

### Why MCC over F1 for binary classification?
F1 ignores true negatives. In ICD-10 coding, most codes are absent for any encounter, so TN is the largest quadrant. MCC uses all four confusion matrix cells, providing a balanced assessment even under extreme class imbalance.

### Why partial NER span matching?
Exact span matching (also provided) is the gold standard, but it treats a prediction of "diabetes" for the gold "type 2 diabetes" the same as a complete miss. Partial matching via Jaccard overlap reveals how close the model is, enabling more nuanced error analysis.

### Why full ROUGE P/R/F1?
Recall-only ROUGE (as in `metrics.py`) rewards long outputs that contain the reference. Precision penalises irrelevant content. Clinical summaries must be both complete (high recall) and concise (high precision).

### Why hierarchical ICD-10 evaluation?
A prediction of E11.9 (Type 2 DM, unspecified) for gold E11.65 (Type 2 DM with hyperglycemia) is wrong at full-code level but correct at block (E11) and chapter (E) levels. Reporting all three levels reveals whether errors are in the right clinical neighbourhood.

### Why no sklearn dependency?
All metrics are implemented from scratch using only numpy and standard library. This keeps the evaluation module importable in lightweight environments and avoids version conflicts with the broader ML stack.

## Performance

All metrics compute in <1ms for typical clinical datasets (up to 100K samples per request). Memory usage is O(n) for all operations.

## Limitations

- **Cohen's Kappa** assumes nominal categories; not suitable for ordinal agreement
- **MCC** is binary-only; for multi-class use the base `metrics.py` module
- **Calibration** uses equal-width bins; adaptive binning may be more appropriate for extreme distributions
- **Partial NER matching** uses greedy assignment; the Hungarian algorithm would give optimal matching but is more complex
- **ROUGE** operates on whitespace-tokenised text; clinical text with abbreviations may benefit from specialised tokenisation
- **AUPRC** uses trapezoidal integration; interpolated precision may differ slightly from sklearn's implementation

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Matthews, B.W. (1975). Comparison of the predicted and observed secondary structure. *Biochimica et Biophysica Acta*, 405(2), 442–451.
- Naeini, M.P. et al. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI*.
- Uzuner, Ö. et al. (2011). 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text. *JAMIA*, 18(5), 552–556.
- Lin, C.Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL Workshop on Text Summarization*.
