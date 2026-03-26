# ClinIQ Python SDK

Typed Python client for the **ClinIQ Clinical NLP API** — covering all 31 endpoint groups including core analysis, 14 specialized clinical NLP modules, evaluation framework, conversation memory, document search, and infrastructure monitoring.

## Installation

```bash
pip install -e ./sdk
```

## Quick Start

```python
from cliniq_client import ClinIQClient

client = ClinIQClient(base_url="http://localhost:8000", api_key="cliniq_xxx")

# Full pipeline analysis
result = client.analyze("Patient has diabetes and takes metformin 1000mg BID.")
print(result.entities)        # typed Entity objects
print(result.icd_predictions)  # typed ICDPrediction objects

# Enhanced pipeline (all 14 modules)
enhanced = client.analyze_enhanced(
    "Patient has hypertension, diabetes. Takes metoprolol 50mg BID.",
    enable_medications=True,
    enable_vitals=True,
    enable_sdoh=True,
)
print(enhanced.medications)
print(enhanced.sdoh)
```

## Available Methods

### Core Analysis
| Method | Endpoint | Returns |
|--------|----------|---------|
| `analyze()` | `POST /analyze` | `AnalysisResult` |
| `analyze_enhanced()` | `POST /analyze/enhanced` | `EnhancedAnalysisResult` |
| `extract_entities()` | `POST /ner` | `list[dict]` |
| `predict_icd()` | `POST /icd-predict` | `list[dict]` |
| `summarize()` | `POST /summarize` | `dict` |
| `assess_risk()` | `POST /risk-score` | `dict` |

### Specialized Clinical Modules
| Method | Endpoint | Returns |
|--------|----------|---------|
| `classify_document()` | `POST /classify` | `ClassificationResult` |
| `extract_medications()` | `POST /medications` | `MedicationResult` |
| `extract_allergies()` | `POST /allergies` | `AllergyResult` |
| `extract_vitals()` | `POST /vitals` | `VitalSignResult` |
| `parse_sections()` | `POST /sections` | `SectionResult` |
| `expand_abbreviations()` | `POST /abbreviations` | `AbbreviationResult` |
| `analyze_quality()` | `POST /quality` | `QualityReport` |
| `extract_sdoh()` | `POST /sdoh` | `SDoHResult` |
| `calculate_comorbidity()` | `POST /comorbidity` | `ComorbidityResult` |
| `extract_relations()` | `POST /relations` | `RelationResult` |
| `normalize_concept()` | `POST /normalize` | `dict` |
| `search()` | `POST /search` | `SearchResult` |

### Evaluation Framework
| Method | Endpoint | Returns |
|--------|----------|---------|
| `evaluate_classification()` | `POST /evaluate/classification` | `ClassificationEvalResult` |
| `evaluate_agreement()` | `POST /evaluate/agreement` | `KappaResult` |
| `evaluate_ner()` | `POST /evaluate/ner` | `NEREvalResult` |
| `evaluate_rouge()` | `POST /evaluate/rouge` | `ROUGEEvalResult` |
| `evaluate_icd()` | `POST /evaluate/icd` | `ICDEvalResult` |
| `evaluate_auprc()` | `POST /evaluate/auprc` | `AUPRCResult` |
| `list_evaluation_metrics()` | `GET /evaluate/metrics` | `list[dict]` |

### Conversation Memory
| Method | Endpoint | Returns |
|--------|----------|---------|
| `add_conversation_turn()` | `POST /conversation/turns` | `ConversationTurnResult` |
| `get_conversation_context()` | `POST /conversation/context` | `ConversationContext` |
| `clear_conversation()` | `DELETE /conversation/{id}` | `dict` |
| `get_conversation_stats()` | `GET /conversation/stats` | `ConversationStats` |
| `list_conversation_sessions()` | `GET /conversation/sessions` | `list[ConversationSessionInfo]` |

### Infrastructure
| Method | Endpoint | Returns |
|--------|----------|---------|
| `health()` | `GET /health` | `dict` |
| `list_models()` | `GET /models` | `list[dict]` |
| `get_metrics()` | `GET /metrics` | `dict` |
| `get_drift_status()` | `GET /drift/status` | `dict` |

### Batch Processing
| Method | Endpoint | Returns |
|--------|----------|---------|
| `submit_batch()` | `POST /batch` | `BatchJob` |
| `get_batch_status()` | `GET /batch/{id}` | `BatchJob` |
| `wait_for_batch()` | polling | `BatchJob` |

## Authentication

```python
# API key auth
client = ClinIQClient(api_key="cliniq_xxx")

# Bearer token auth
client = ClinIQClient(token="jwt.token.here")

# Login to get token
client = ClinIQClient()
token = client.login("user@example.com", "password")
```

## Models

All response models are Python dataclasses with type hints:

- `AnalysisResult`, `EnhancedAnalysisResult` — Full pipeline results
- `Entity`, `ICDPrediction`, `Summary`, `RiskAssessment` — Core types
- `MedicationResult`, `AllergyResult`, `VitalSignResult` — Extraction results
- `SectionResult`, `AbbreviationResult`, `QualityReport` — Document analysis
- `SDoHResult`, `ComorbidityResult`, `RelationResult` — Clinical scoring
- `ClassificationResult`, `SearchResult`, `BatchJob` — Utilities
- `ClassificationEvalResult`, `KappaResult`, `NEREvalResult`, `ROUGEEvalResult`, `ICDEvalResult`, `AUPRCResult` — Evaluation metrics
- `ConversationTurnResult`, `ConversationContext`, `ConversationStats`, `ConversationSessionInfo` — Conversation memory

## Development

```bash
cd sdk
pip install -e ".[dev]"
pytest tests/ -v
```

## Version

v0.3.0 — Expanded from 29 to 31 endpoint groups with evaluation framework (7 metric types) and conversation memory (5 endpoints).
