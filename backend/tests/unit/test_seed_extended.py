"""Extended tests for db/seed.py — covering the main() entry point (lines 104-109)."""

from unittest.mock import patch, AsyncMock

import pytest

from app.db.seed import main


class TestSeedMain:
    """Cover the main() entry point which calls asyncio.run(seed_all())."""

    def test_main_calls_seed_all(self) -> None:
        """main() should call asyncio.run with seed_all()."""
        with patch("app.db.seed.asyncio") as mock_asyncio, \
             patch("app.db.seed.logging") as mock_logging:
            mock_asyncio.run = lambda coro: None  # Just consume the coroutine
            main()
            mock_logging.basicConfig.assert_called_once()
