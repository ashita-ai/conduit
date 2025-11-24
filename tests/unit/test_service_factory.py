"""Unit tests for service factory."""

import pytest
from unittest.mock import AsyncMock, patch

from conduit.core.database import Database
from conduit.utils.service_factory import create_service


class TestServiceFactory:
    """Tests for create_service factory function."""

    @pytest.mark.asyncio
    async def test_create_service_with_default_database(self):
        """Test create_service creates new database if not provided."""
        with patch("conduit.utils.service_factory.Database") as MockDatabase:
            mock_db = AsyncMock(spec=Database)
            mock_db.connect = AsyncMock()
            MockDatabase.return_value = mock_db

            service = await create_service()

            # Verify database was created and connected
            MockDatabase.assert_called_once()
            mock_db.connect.assert_called_once()

            # Verify service components are initialized
            assert service.database is mock_db
            assert service.router is not None
            assert service.router.hybrid_router is not None
            assert service.executor is not None

    @pytest.mark.asyncio
    async def test_create_service_with_provided_database(self):
        """Test create_service uses provided database."""
        mock_db = AsyncMock(spec=Database)

        service = await create_service(database=mock_db)

        # Verify provided database was used (no connect called since already provided)
        assert service.database is mock_db

        # Verify service components are initialized
        assert service.router is not None
        assert service.router.hybrid_router is not None
        assert service.executor is not None

    @pytest.mark.asyncio
    async def test_create_service_with_custom_result_type(self):
        """Test create_service accepts custom result type."""
        from pydantic import BaseModel

        class CustomResult(BaseModel):
            result: str

        mock_db = AsyncMock(spec=Database)

        service = await create_service(
            database=mock_db,
            default_result_type=CustomResult
        )

        # Verify custom result type was passed through
        assert service.default_result_type is CustomResult

    @pytest.mark.asyncio
    async def test_create_service_initializes_hybrid_router(self):
        """Test create_service initializes router with hybrid routing."""
        mock_db = AsyncMock(spec=Database)

        service = await create_service(database=mock_db)

        # Verify hybrid router components
        assert service.router.hybrid_router is not None
        assert service.router.hybrid_router.ucb1 is not None
        assert service.router.hybrid_router.linucb is not None
        assert service.router.analyzer is not None

    @pytest.mark.asyncio
    async def test_create_service_uses_default_models(self):
        """Test create_service uses default models from settings."""
        mock_db = AsyncMock(spec=Database)

        service = await create_service(database=mock_db)

        # Verify models are set (from settings.default_models)
        assert len(service.router.hybrid_router.models) > 0
