# ClinIQ Python SDK

Typed Python client for the ClinIQ Clinical NLP API.

## Installation

```bash
pip install cliniq-client
```

## Quick Start

```python
from cliniq_client import ClinIQClient

client = ClinIQClient(
    base_url="http://localhost:8000",
    api_key="cliniq_your_api_key_here"
)

# Full pipeline analysis
result = client.analyze("Patient has type 2 diabetes managed with metformin 1000mg BID.")

# Access structured results
for entity in result.entities:
    print(f"{entity.text} ({entity.entity_type}) - {entity.confidence:.0%}")

for pred in result.icd_predictions:
    print(f"{pred.code}: {pred.description} ({pred.confidence:.0%})")

if result.risk_assessment:
    print(f"Risk: {result.risk_assessment.risk_level} ({result.risk_assessment.overall_score})")

# Individual endpoints
entities = client.extract_entities("Aspirin 81mg daily for CAD prophylaxis")
codes = client.predict_icd("Acute appendicitis with peritonitis")
summary = client.summarize(long_clinical_note, detail_level="brief")

# Batch processing
job = client.submit_batch([
    {"text": "Patient note 1...", "document_id": "doc-1"},
    {"text": "Patient note 2...", "document_id": "doc-2"},
])
result = client.wait_for_batch(job.job_id)
```
