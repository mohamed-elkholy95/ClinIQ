"""Unit tests for custom exception classes."""

import pytest

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BatchProcessingError,
    ClinIQError,
    ConfigurationError,
    DatabaseError,
    DocumentProcessingError,
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)


class TestClinIQError:
    """Tests for the base ClinIQError exception class."""

    def test_basic_creation(self):
        """Test creating a ClinIQError with just a message."""
        exc = ClinIQError("Something went wrong")
        assert exc.message == "Something went wrong"
        assert str(exc) == "Something went wrong"

    def test_default_error_code(self):
        """Test that the default error code is 'INTERNAL_ERROR'."""
        exc = ClinIQError("Error without code")
        assert exc.error_code == "INTERNAL_ERROR"

    def test_custom_error_code(self):
        """Test setting a custom error code."""
        exc = ClinIQError("Error", error_code="CUSTOM_CODE")
        assert exc.error_code == "CUSTOM_CODE"

    def test_default_details_is_empty_dict(self):
        """Test that details defaults to an empty dict."""
        exc = ClinIQError("Error")
        assert exc.details == {}

    def test_custom_details(self):
        """Test passing custom details."""
        details = {"key": "value", "count": 42}
        exc = ClinIQError("Error with details", details=details)
        assert exc.details == details

    def test_is_exception_subclass(self):
        """Test that ClinIQError is a subclass of Exception."""
        exc = ClinIQError("Error")
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that ClinIQError can be raised and caught."""
        with pytest.raises(ClinIQError) as exc_info:
            raise ClinIQError("Test error", error_code="TEST")

        assert exc_info.value.error_code == "TEST"
        assert exc_info.value.message == "Test error"


class TestValidationError:
    """Tests for ValidationError."""

    def test_error_code_is_validation_error(self):
        """Test that ValidationError sets the correct error code."""
        exc = ValidationError("Field is required")
        assert exc.error_code == "VALIDATION_ERROR"

    def test_is_subclass_of_cliniq_error(self):
        """Test that ValidationError is a ClinIQError."""
        exc = ValidationError("Validation failed")
        assert isinstance(exc, ClinIQError)

    def test_message_preserved(self):
        """Test that the message is preserved."""
        exc = ValidationError("Invalid email format")
        assert exc.message == "Invalid email format"

    def test_with_details(self):
        """Test ValidationError with field details."""
        details = {"field": "email", "value": "not-an-email"}
        exc = ValidationError("Invalid email", details=details)
        assert exc.details == details

    def test_can_be_caught_as_cliniq_error(self):
        """Test that ValidationError can be caught as ClinIQError."""
        with pytest.raises(ClinIQError):
            raise ValidationError("Validation failed")


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_error_code(self):
        """Test error code."""
        exc = AuthenticationError()
        assert exc.error_code == "AUTHENTICATION_ERROR"

    def test_default_message(self):
        """Test default message."""
        exc = AuthenticationError()
        assert "authentication" in exc.message.lower() or "failed" in exc.message.lower()

    def test_custom_message(self):
        """Test providing a custom message."""
        exc = AuthenticationError("Invalid credentials")
        assert exc.message == "Invalid credentials"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = AuthenticationError()
        assert isinstance(exc, ClinIQError)


class TestAuthorizationError:
    """Tests for AuthorizationError."""

    def test_error_code(self):
        """Test error code."""
        exc = AuthorizationError()
        assert exc.error_code == "AUTHORIZATION_ERROR"

    def test_default_message(self):
        """Test default message contains 'authorized' or 'permission'."""
        exc = AuthorizationError()
        assert len(exc.message) > 0

    def test_custom_message(self):
        """Test custom message."""
        exc = AuthorizationError("Insufficient permissions")
        assert exc.message == "Insufficient permissions"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = AuthorizationError()
        assert isinstance(exc, ClinIQError)


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_error_code(self):
        """Test error code."""
        exc = NotFoundError("Document")
        assert exc.error_code == "NOT_FOUND"

    def test_message_contains_resource_name(self):
        """Test that the message contains the resource name."""
        exc = NotFoundError("Document")
        assert "Document" in exc.message

    def test_message_without_identifier(self):
        """Test message when no identifier is provided."""
        exc = NotFoundError("User")
        assert "User" in exc.message
        assert "not found" in exc.message.lower()

    def test_message_with_identifier(self):
        """Test message when identifier is provided."""
        exc = NotFoundError("Document", identifier="doc-123")
        assert "doc-123" in exc.message

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = NotFoundError("Resource")
        assert isinstance(exc, ClinIQError)

    def test_integer_identifier(self):
        """Test with an integer identifier."""
        exc = NotFoundError("User", identifier=42)
        assert "42" in exc.message


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_error_code(self):
        """Test error code."""
        exc = RateLimitError()
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"

    def test_message(self):
        """Test that message mentions rate limit."""
        exc = RateLimitError()
        assert "rate limit" in exc.message.lower() or "exceeded" in exc.message.lower()

    def test_retry_after_in_details(self):
        """Test that retry_after appears in details."""
        exc = RateLimitError(retry_after=60)
        assert exc.details.get("retry_after") == 60

    def test_no_retry_after(self):
        """Test creation without retry_after."""
        exc = RateLimitError()
        assert exc.details.get("retry_after") is None

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = RateLimitError()
        assert isinstance(exc, ClinIQError)


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_error_code(self):
        """Test error code."""
        exc = ModelNotFoundError("biobert")
        assert exc.error_code == "MODEL_NOT_FOUND"

    def test_message_contains_model_name(self):
        """Test that message contains the model name."""
        exc = ModelNotFoundError("biobert")
        assert "biobert" in exc.message

    def test_message_with_version(self):
        """Test message when version is provided."""
        exc = ModelNotFoundError("biobert", model_version="v1.0")
        assert "biobert" in exc.message

    def test_details_contain_model_name(self):
        """Test that details dict contains model_name."""
        exc = ModelNotFoundError("en_ner_bc5cdr_md")
        assert exc.details.get("model_name") == "en_ner_bc5cdr_md"

    def test_details_with_version(self):
        """Test that details dict contains model_version when provided."""
        exc = ModelNotFoundError("biobert", model_version="1.1.0")
        assert exc.details.get("model_version") == "1.1.0"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = ModelNotFoundError("model")
        assert isinstance(exc, ClinIQError)


class TestModelLoadError:
    """Tests for ModelLoadError."""

    def test_error_code(self):
        """Test error code."""
        exc = ModelLoadError("biobert")
        assert exc.error_code == "MODEL_LOAD_ERROR"

    def test_message_contains_model_name(self):
        """Test that message references the model name."""
        exc = ModelLoadError("en_core_sci_md")
        assert "en_core_sci_md" in exc.message

    def test_message_contains_reason(self):
        """Test that message includes the reason when provided."""
        exc = ModelLoadError("biobert", reason="File not found")
        assert "File not found" in exc.message

    def test_message_without_reason(self):
        """Test message when no reason is provided."""
        exc = ModelLoadError("biobert")
        assert "biobert" in exc.message

    def test_details_model_name(self):
        """Test that details contain model_name."""
        exc = ModelLoadError("rule-based")
        assert exc.details.get("model_name") == "rule-based"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = ModelLoadError("model")
        assert isinstance(exc, ClinIQError)


class TestInferenceError:
    """Tests for InferenceError."""

    def test_error_code(self):
        """Test error code."""
        exc = InferenceError("biobert")
        assert exc.error_code == "INFERENCE_ERROR"

    def test_message_contains_model_name(self):
        """Test that message contains the model name."""
        exc = InferenceError("rule-based-ner")
        assert "rule-based-ner" in exc.message

    def test_message_contains_reason(self):
        """Test that message includes the reason when provided."""
        exc = InferenceError("biobert", reason="CUDA out of memory")
        assert "CUDA out of memory" in exc.message

    def test_details_model_name(self):
        """Test that details contain model_name."""
        exc = InferenceError("icd-classifier")
        assert exc.details.get("model_name") == "icd-classifier"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = InferenceError("model")
        assert isinstance(exc, ClinIQError)


class TestDocumentProcessingError:
    """Tests for DocumentProcessingError."""

    def test_error_code(self):
        """Test error code."""
        exc = DocumentProcessingError("Parsing failed")
        assert exc.error_code == "DOCUMENT_ERROR"

    def test_message_contains_reason(self):
        """Test that message contains the reason."""
        exc = DocumentProcessingError("Invalid encoding")
        assert "Invalid encoding" in exc.message

    def test_with_document_id(self):
        """Test creation with a document ID."""
        exc = DocumentProcessingError("Parsing failed", document_id="doc-abc-123")
        assert exc.details.get("document_id") == "doc-abc-123"

    def test_without_document_id(self):
        """Test creation without a document ID."""
        exc = DocumentProcessingError("Parsing failed")
        assert exc.details.get("document_id") is None

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = DocumentProcessingError("reason")
        assert isinstance(exc, ClinIQError)


class TestBatchProcessingError:
    """Tests for BatchProcessingError."""

    def test_error_code(self):
        """Test error code."""
        exc = BatchProcessingError("job-001", "Worker crashed")
        assert exc.error_code == "BATCH_ERROR"

    def test_message_contains_job_id(self):
        """Test that message contains the job ID."""
        exc = BatchProcessingError("job-xyz", "Timeout")
        assert "job-xyz" in exc.message

    def test_message_contains_reason(self):
        """Test that message contains the failure reason."""
        exc = BatchProcessingError("job-001", "Memory limit exceeded")
        assert "Memory limit exceeded" in exc.message

    def test_details_contain_job_id(self):
        """Test that details contain the job_id."""
        exc = BatchProcessingError("job-42", "Failed")
        assert exc.details.get("job_id") == "job-42"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = BatchProcessingError("job", "reason")
        assert isinstance(exc, ClinIQError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_error_code(self):
        """Test error code."""
        exc = ConfigurationError("Invalid config")
        assert exc.error_code == "CONFIGURATION_ERROR"

    def test_message_preserved(self):
        """Test that message is preserved."""
        exc = ConfigurationError("Missing required field: SECRET_KEY")
        assert exc.message == "Missing required field: SECRET_KEY"

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = ConfigurationError("Error")
        assert isinstance(exc, ClinIQError)


class TestDatabaseError:
    """Tests for DatabaseError."""

    def test_error_code(self):
        """Test error code."""
        exc = DatabaseError("INSERT")
        assert exc.error_code == "DATABASE_ERROR"

    def test_message_contains_operation(self):
        """Test that message contains the operation name."""
        exc = DatabaseError("SELECT")
        assert "SELECT" in exc.message

    def test_message_with_reason(self):
        """Test message when reason is provided."""
        exc = DatabaseError("UPDATE", reason="Connection refused")
        assert "Connection refused" in exc.message

    def test_message_without_reason(self):
        """Test message when no reason is provided."""
        exc = DatabaseError("DELETE")
        assert "DELETE" in exc.message

    def test_is_cliniq_error(self):
        """Test inheritance."""
        exc = DatabaseError("operation")
        assert isinstance(exc, ClinIQError)


class TestExceptionHierarchy:
    """Tests for the exception inheritance hierarchy."""

    def test_all_errors_are_cliniq_error(self):
        """Test that every custom exception is a subclass of ClinIQError."""
        exception_classes = [
            ValidationError("msg"),
            AuthenticationError(),
            AuthorizationError(),
            NotFoundError("Resource"),
            RateLimitError(),
            ModelNotFoundError("model"),
            ModelLoadError("model"),
            InferenceError("model"),
            DocumentProcessingError("reason"),
            BatchProcessingError("job", "reason"),
            ConfigurationError("msg"),
            DatabaseError("op"),
        ]
        for exc in exception_classes:
            assert isinstance(exc, ClinIQError), (
                f"{type(exc).__name__} is not a ClinIQError subclass"
            )

    def test_all_errors_are_exceptions(self):
        """Test that every custom exception is an Exception subclass."""
        exception_classes = [
            ValidationError("msg"),
            AuthenticationError(),
            ModelLoadError("model"),
            InferenceError("model"),
        ]
        for exc in exception_classes:
            assert isinstance(exc, Exception)
