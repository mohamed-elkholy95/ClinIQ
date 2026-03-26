# ClinIQ — Inference Benchmarks

Documented performance characteristics for all ML modules.  All benchmarks
measured on a single-threaded Python 3.12 process without GPU unless noted.

## Rule-Based Modules (No ML Dependencies)

| Module | Avg Latency | p95 Latency | Throughput | Notes |
|--------|-------------|-------------|------------|-------|
| Text Preprocessing | <1 ms | 1 ms | ~5,000 docs/s | Regex-based normalisation |
| Section Parser | <1 ms | 1 ms | ~5,000 docs/s | 3-strategy header detection |
| Allergy Extractor | <3 ms | 5 ms | ~1,500 docs/s | 150 allergens, 250 surface forms |
| Medication Extractor | <3 ms | 5 ms | ~1,500 docs/s | ~200 drugs + INN suffix patterns |
| Vital Signs Extractor | <2 ms | 3 ms | ~2,000 docs/s | 9 vital types, 20+ regex patterns |
| Abbreviation Expander | <5 ms | 8 ms | ~1,000 docs/s | 220+ unambiguous + 10 ambiguous entries |
| Quality Analyzer | <5 ms | 8 ms | ~1,000 docs/s | 5-dimension heuristic scoring |
| De-identification | <5 ms | 10 ms | ~800 docs/s | 12 PHI types, 3 replacement strategies |
| Assertion Detector | <3 ms | 5 ms | ~1,500 docs/s | NegEx/ConText approximation |
| Temporal Extractor | <3 ms | 5 ms | ~1,500 docs/s | 6 temporal expression types |
| SDoH Extractor | <5 ms | 8 ms | ~1,000 docs/s | 8 domains, 100+ trigger patterns |
| Comorbidity Calculator | <2 ms | 3 ms | ~2,000 docs/s | 17 Charlson categories, ~200 ICD prefixes |
| Relation Extractor (rule) | <3 ms | 5 ms | ~1,500 docs/s | 12 relation types, proximity + sentence |
| Concept Normalizer | <5 ms | 10 ms | ~800 docs/s | 140 concepts, exact/alias/fuzzy match |
| Rule-based NER | <5 ms | 8 ms | ~1,000 docs/s | Pattern + dictionary matching |

## ML-Based Modules

| Module | Avg Latency (CPU) | Avg Latency (GPU) | Notes |
|--------|-------------------|-------------------|-------|
| scispaCy NER | 30–50 ms | N/A (CPU only) | `en_ner_bc5cdr_md` pipeline |
| BioBERT NER | 80–150 ms | 15–25 ms | Token classification, 512 max tokens |
| ICD-10 (TF-IDF + LR) | 5–10 ms | N/A | Sklearn predict, sparse features |
| ICD-10 (ClinicalBERT) | 80–150 ms | 15–25 ms | Multi-label sigmoid head |
| Document Classifier (rule) | <3 ms | N/A | Keyword + section + structural scoring |
| Document Classifier (transformer) | 60–120 ms | 10–20 ms | 14-type sequence classification |
| Extractive Summarizer | 10–30 ms | N/A | TextRank + clinical relevance |
| Risk Scorer (rule) | <5 ms | N/A | Additive factor model |
| Risk Scorer (ML) | 50–100 ms | 10–20 ms | Depends on model backend |

## Search & Retrieval

| Component | Avg Latency | Notes |
|-----------|-------------|-------|
| BM25 Search | 5–15 ms | In-memory inverted index |
| Query Expansion | <2 ms | 120+ abbreviation/synonym mappings |
| Neural Reranking | 50–100 ms | Cross-encoder scoring (CPU) |
| Hybrid Search (full) | 60–130 ms | BM25 + expansion + rerank pipeline |

## Enhanced Pipeline (Full 14-Module)

| Configuration | Avg Latency | Notes |
|---------------|-------------|-------|
| All rule-based modules | 30–60 ms | 14 modules, zero ML dependencies |
| Rule-based + scispaCy NER | 80–120 ms | Adding biomedical NER |
| Full pipeline (all ML) | 200–400 ms | All transformer models on CPU |
| Full pipeline (GPU) | 50–100 ms | All transformer models on GPU |

## ONNX Runtime Optimisation

When models are exported to ONNX format via ``app.ml.serving.onnx_serving``:

| Model | PyTorch CPU | ONNX CPU | Speedup |
|-------|-------------|----------|---------|
| BioBERT NER | 80–150 ms | 30–60 ms | ~2.5× |
| ClinicalBERT ICD | 80–150 ms | 30–60 ms | ~2.5× |
| Document Classifier | 60–120 ms | 25–50 ms | ~2.4× |

## Memory Usage

| Component | Approximate Memory |
|-----------|-------------------|
| Rule-based modules (all) | ~50 MB |
| scispaCy `en_ner_bc5cdr_md` | ~200 MB |
| BioBERT (PyTorch) | ~420 MB |
| ClinicalBERT (PyTorch) | ~420 MB |
| ONNX Runtime model | ~200 MB per model |
| Concept normalization dict | ~5 MB |
| BM25 index (10K docs) | ~30 MB |

## Scaling Characteristics

- **Horizontal scaling**: Celery workers process documents in parallel;
  HPA scales 2–8 replicas at 75% CPU utilisation.
- **Batch processing**: Most modules support `process_batch()` with
  linear scaling (2× documents ≈ 2× time, no per-batch overhead).
- **Inference caching**: Circuit-breaker-protected Redis cache for
  repeated documents (cache hit = <1 ms).
- **Graceful degradation**: Each enhanced pipeline module is independently
  fault-tolerant — a failing transformer falls back to rule-based
  without aborting the pipeline.

## Methodology

Benchmarks are approximate and based on:
- Average clinical note length: 500–2,000 words
- Hardware: 4-core CPU, 16 GB RAM (typical API server)
- Python 3.12, PyTorch 2.x, scikit-learn 1.x
- Cold-start latency excluded (first request includes model loading)

For production latency monitoring, see the Grafana ML Inference dashboard
(22 panels) with real-time p50/p95/p99 tracking per model.
