"""Celery worker for background task processing."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "cliniq",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minute hard limit
    task_soft_time_limit=540,  # 9 minute soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


@celery_app.task(bind=True, name="cliniq.process_batch")
def process_batch_task(self, job_id: str, documents: list[dict], config: dict) -> dict:
    """Process a batch of clinical documents.

    This task runs NER, ICD-10 prediction, summarization, and risk scoring
    on each document in the batch.
    """
    from app.ml.pipeline import ClinicalPipeline, PipelineConfig

    pipeline_config = PipelineConfig(
        enable_ner=config.get("enable_ner", True),
        enable_icd=config.get("enable_icd", True),
        enable_summarization=config.get("enable_summarization", True),
        enable_risk=config.get("enable_risk", True),
        enable_dental=config.get("enable_dental", False),
        confidence_threshold=config.get("confidence_threshold", 0.5),
        top_k_icd=config.get("top_k_icd", 10),
        detail_level=config.get("detail_level", "standard"),
    )

    pipeline = ClinicalPipeline()
    pipeline.load()

    results = []
    total = len(documents)

    for i, doc in enumerate(documents):
        try:
            result = pipeline.process(doc["text"], pipeline_config)
            results.append({
                "document_id": doc.get("document_id", str(i)),
                "status": "completed",
                "result": {
                    "entities": [e.to_dict() for e in (result.entities or [])],
                    "icd_predictions": [p.to_dict() for p in (result.icd_predictions or [])],
                    "summary": result.summary.to_dict() if result.summary else None,
                    "risk_assessment": result.risk_assessment.to_dict() if result.risk_assessment else None,
                },
            })
        except Exception as exc:
            results.append({
                "document_id": doc.get("document_id", str(i)),
                "status": "failed",
                "error": str(exc),
            })

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": total, "job_id": job_id},
        )

    return {"job_id": job_id, "results": results, "total": total}


@celery_app.task(name="cliniq.health_check")
def health_check() -> dict:
    """Simple health check task for worker monitoring."""
    return {"status": "healthy", "worker": "cliniq"}
