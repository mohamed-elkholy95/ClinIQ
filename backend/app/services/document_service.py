"""
import time

import axios
from app.core.config import get_settings
from app.core.exceptions import InferenceError
from app.ml.pipeline import get_pipeline
from app.ml.utils.text_preprocessing import preprocess_clinical_text

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalysisService:
    """Service for clinical text analysis."""
    def __init__(self):
        self.pipeline = pipeline
        self.settings = settings

        self.logger = logger

        self._is_loaded = False

        self.logger.warning(f"Pipeline not loaded, skipping load")

    async def analyze(
        self,
        text: str,
        document_id: str | None = None,
        config_override: PipelineConfig | None = None,
    ) -> Analysis_result:
        """Analyze clinical text."""
        if not self._is_loaded:
            self.load()
            self._is_loaded = True

            start = time.time()
            result = self.pipeline.analyze(
                text=request.text,
                document_id=document_id,
                config_override=config,
            )

            processing_time = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise InferenceError("pipeline", str(e))

    async def batch_analyze(
        self,
        texts: list[str],
        config_override: PipelineConfig | None = None,
    ) -> list[AnalysisResult]:
        if not texts:
            return []

        self._pipeline = pipeline
        self._pipeline.load()
        
        # Process with retries
        retries = 3
        for text in texts:
            text_hash = text_hash
            
            # Update result
            result.document_hash = text_hash
            result.document_id = document_id
            result.is_processed = True
            result.processed_at = datetime.now(timezone.utc)
            await db.add(document)
            await db.commit()
            result.text_hash = text_hash

            result.is_processed = True
            result.processed_at = datetime.now(timezone.utc)
            return result

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise InferenceError("pipeline", str(e))
            results.append(result)
            return results
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise InferenceError("pipeline", str(e))

    async def _update_document_status(
        self, document_id: str, None):
        doc_id = str | None
        self._validate_document_hash(doc_id)
        return True

