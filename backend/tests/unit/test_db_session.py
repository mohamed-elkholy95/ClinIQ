"""Unit tests for database session management.

Covers get_db_session dependency, get_db_context context manager,
init_db, and close_db lifecycle functions.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# get_db_session (async generator dependency)
# ---------------------------------------------------------------------------


class TestGetDbSession:
    """Test the FastAPI dependency that yields a database session."""

    @pytest.mark.asyncio
    async def test_yields_session_and_commits(self) -> None:
        """Normal flow: yield session → commit → close."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_factory = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_ctx.__aexit__.return_value = False
        mock_factory.return_value = mock_ctx

        with patch("app.db.session.async_session_factory", mock_factory):
            from app.db.session import get_db_session

            gen = get_db_session()
            session = await gen.__anext__()

            assert session is mock_session

            # Simulate normal exit
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

            mock_session.commit.assert_awaited_once()
            mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rollback_on_exception(self) -> None:
        """When code raises inside the yield, session is rolled back."""
        mock_session = AsyncMock(spec=AsyncSession)

        mock_factory = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_ctx.__aexit__.return_value = False
        mock_factory.return_value = mock_ctx

        with patch("app.db.session.async_session_factory", mock_factory):
            from app.db.session import get_db_session

            gen = get_db_session()
            await gen.__anext__()

            # Simulate exception
            with pytest.raises(ValueError):
                await gen.athrow(ValueError("boom"))

            mock_session.rollback.assert_awaited_once()
            mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_db_context (async context manager)
# ---------------------------------------------------------------------------


class TestGetDbContext:
    """Test the standalone context manager for outside-FastAPI usage."""

    @pytest.mark.asyncio
    async def test_context_commits_on_success(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)

        mock_factory = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_ctx.__aexit__.return_value = False
        mock_factory.return_value = mock_ctx

        with patch("app.db.session.async_session_factory", mock_factory):
            from app.db.session import get_db_context

            async with get_db_context() as session:
                assert session is mock_session

            mock_session.commit.assert_awaited_once()
            mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_rollback_on_exception(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)

        mock_factory = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_session
        mock_ctx.__aexit__.return_value = False
        mock_factory.return_value = mock_ctx

        with patch("app.db.session.async_session_factory", mock_factory):
            from app.db.session import get_db_context

            with pytest.raises(RuntimeError):
                async with get_db_context() as session:
                    raise RuntimeError("DB error")

            mock_session.rollback.assert_awaited_once()
            mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# init_db / close_db
# ---------------------------------------------------------------------------


class TestInitCloseDb:
    """Test database lifecycle functions."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self) -> None:
        mock_conn = AsyncMock()

        # engine.begin() is NOT a coroutine — it returns an async context manager directly
        mock_engine = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__.return_value = mock_conn
        ctx.__aexit__.return_value = False
        mock_engine.begin.return_value = ctx
        mock_engine.dispose = AsyncMock()

        with patch("app.db.session.engine", mock_engine):
            from app.db.session import init_db
            await init_db()

        mock_conn.run_sync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self) -> None:
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with patch("app.db.session.engine", mock_engine):
            from app.db.session import close_db
            await close_db()

        mock_engine.dispose.assert_awaited_once()
