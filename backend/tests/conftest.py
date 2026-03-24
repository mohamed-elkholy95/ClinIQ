"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base, User
from app.db.session import get_db_session
from app.main import app
from app.ml.pipeline import ClinicalPipeline


# Test settings
@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide test settings."""
    return Settings(
        app_name="ClinIQ-Test",
        environment="development",
        debug=True,
        database_url="postgresql+asyncpg://test:test@localhost:5432/cliniq_test",
        secret_key="test-secret-key-for-testing-only",
        cors_origins=["http://localhost:3000"],
    )


# Test database engine
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide test database session."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create test user."""
    from app.core.security import get_password_hash

    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
        is_active=True,
        is_superuser=False,
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create admin user."""
    from app.core.security import get_password_hash

    user = User(
        email="admin@example.com",
        hashed_password=get_password_hash("adminpassword123"),
        full_name="Admin User",
        is_active=True,
        is_superuser=True,
        role="admin",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# Test client
@pytest.fixture
def client(db_session: AsyncSession, test_user: User) -> TestClient:
    """Provide test client with database override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide async test client."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# ML Pipeline mocks
@pytest.fixture
def mock_ner_model() -> MagicMock:
    """Mock NER model."""
    from app.ml.ner.model import Entity

    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.extract_entities.return_value = [
        Entity(
            text="metformin",
            entity_type="MEDICATION",
            start_char=10,
            end_char=19,
            confidence=0.95,
        ),
        Entity(
            text="diabetes",
            entity_type="DISEASE",
            start_char=30,
            end_char=38,
            confidence=0.90,
        ),
    ]
    return model


@pytest.fixture
def mock_icd_model() -> MagicMock:
    """Mock ICD prediction model."""
    from app.ml.icd.model import ICDCodePrediction, ICDPredictionResult

    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.predict.return_value = ICDPredictionResult(
        predictions=[
            ICDCodePrediction(
                code="E11.9",
                description="Type 2 diabetes mellitus without complications",
                confidence=0.85,
                chapter="Endocrine, nutritional and metabolic diseases",
            ),
        ],
        processing_time_ms=50.0,
        model_name="test-model",
        model_version="1.0.0",
    )
    return model


@pytest.fixture
def mock_summarizer() -> MagicMock:
    """Mock summarizer."""
    from app.ml.summarization.model import SummarizationResult

    summarizer = MagicMock()
    summarizer.is_loaded = True
    summarizer.version = "1.0.0"
    summarizer.summarize.return_value = SummarizationResult(
        summary="Patient presents with diabetes managed with metformin.",
        original_length=100,
        summary_length=50,
        compression_ratio=2.0,
        processing_time_ms=100.0,
        model_name="test-summarizer",
        model_version="1.0.0",
        summary_type="extractive",
    )
    return summarizer


@pytest.fixture
def mock_pipeline(
    mock_ner_model: MagicMock,
    mock_icd_model: MagicMock,
    mock_summarizer: MagicMock,
) -> ClinicalPipeline:
    """Provide mock ML pipeline."""
    pipeline = ClinicalPipeline()
    pipeline._ner_model = mock_ner_model
    pipeline._icd_model = mock_icd_model
    pipeline._summarizer = mock_summarizer
    pipeline._is_loaded = True
    return pipeline


# Sample data
@pytest.fixture
def sample_clinical_text() -> str:
    """Provide sample clinical text for testing."""
    return """
    CHIEF COMPLAINT: Follow-up for diabetes mellitus type 2

    HISTORY OF PRESENT ILLNESS:
    Patient is a 55-year-old male with a history of type 2 diabetes mellitus,
    hypertension, and hyperlipidemia presenting for routine follow-up.
    He reports good compliance with medications including metformin 1000mg twice daily
    and lisinopril 10mg daily. Blood glucose levels have been well controlled.
    No symptoms of hypoglycemia. Patient denies chest pain, shortness of breath,
    or peripheral edema.

    PAST MEDICAL HISTORY:
    - Type 2 diabetes mellitus
    - Hypertension
    - Hyperlipidemia
    - Obesity

    MEDICATIONS:
    - Metformin 1000mg PO BID
    - Lisinopril 10mg PO daily
    - Atorvastatin 20mg PO daily
    - Aspirin 81mg PO daily

    ALLERGIES: NKDA

    ASSESSMENT AND PLAN:
    1. Type 2 DM - well controlled, continue current regimen
    2. HTN - well controlled, continue lisinopril
    3. Hyperlipidemia - continue atorvastatin
    4. Obesity - counseled on diet and exercise

    Follow up in 3 months.
    """


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide sample documents for batch testing."""
    return [
        "Patient has diabetes and takes metformin.",
        "History of hypertension, currently on lisinopril.",
        "Acute appendicitis, scheduled for appendectomy.",
    ]
