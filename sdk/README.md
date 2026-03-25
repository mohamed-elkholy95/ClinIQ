# ClinIQ Python SDK

Typed Python client for the **ClinIQ Clinical NLP API** — covering all 29 endpoint groups including core analysis, 14 specialized clinical NLP modules, document search, and infrastructure monitoring.

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

## Development

```bash
cd sdk
pip install -e ".[dev]"
pytest tests/ -v
```

## Version

v0.2.0 — Expanded from 8 to 29 endpoint groups with full typed model coverage.
