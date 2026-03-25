"""Unit tests for database seeding script.

Tests ICD-10 code seeding, admin user creation, idempotency (skip duplicates),
and the seed_all orchestrator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.seed import ICD10_SEED_DATA, seed_admin_user, seed_all, seed_icd_codes


class TestICD10SeedData:
    """Validate the ICD-10 seed data structure."""

    def test_has_entries(self) -> None:
        assert len(ICD10_SEED_DATA) > 30

    def test_entry_structure(self) -> None:
        for code, description, chapter, category in ICD10_SEED_DATA:
            assert isinstance(code, str) and len(code) > 0
            assert isinstance(description, str) and len(description) > 0
            assert isinstance(chapter, str)
            assert isinstance(category, str)

    def test_dental_codes_present(self) -> None:
        codes = [entry[0] for entry in ICD10_SEED_DATA]
        assert "K02.9" in codes  # Dental caries
        assert "K05.3" in codes  # Chronic periodontitis

    def test_common_codes_present(self) -> None:
        codes = [entry[0] for entry in ICD10_SEED_DATA]
        assert "I10" in codes     # Hypertension
        assert "E11.9" in codes   # T2DM
        assert "J18.9" in codes   # Pneumonia

    def test_no_duplicate_codes(self) -> None:
        codes = [entry[0] for entry in ICD10_SEED_DATA]
        assert len(codes) == len(set(codes))


class TestSeedICDCodes:
    """Tests for seed_icd_codes function."""

    @pytest.mark.asyncio
    async def test_seeds_new_codes(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No existing code
        mock_db.execute.return_value = mock_result

        with patch("app.db.seed.get_db_context") as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            count = await seed_icd_codes()

        assert count == len(ICD10_SEED_DATA)
        assert mock_db.add.call_count == len(ICD10_SEED_DATA)

    @pytest.mark.asyncio
    async def test_skips_existing_codes(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Already exists
        mock_db.execute.return_value = mock_result

        with patch("app.db.seed.get_db_context") as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            count = await seed_icd_codes()

        assert count == 0
        mock_db.add.assert_not_called()


class TestSeedAdminUser:
    """Tests for seed_admin_user function."""

    @pytest.mark.asyncio
    async def test_creates_admin_when_missing(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with patch("app.db.seed.get_db_context") as mock_ctx, \
             patch("app.db.seed.get_password_hash", return_value="hashed"):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_admin_user()

        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_admin_exists(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch("app.db.seed.get_db_context") as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_admin_user()

        mock_db.add.assert_not_called()


class TestSeedAll:
    """Tests for the seed_all orchestrator."""

    @pytest.mark.asyncio
    async def test_calls_both_seed_functions(self) -> None:
        with patch("app.db.seed.seed_icd_codes", new_callable=AsyncMock) as mock_icd, \
             patch("app.db.seed.seed_admin_user", new_callable=AsyncMock) as mock_admin:
            await seed_all()
            mock_icd.assert_called_once()
            mock_admin.assert_called_once()
