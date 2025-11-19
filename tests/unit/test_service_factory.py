"""Unit tests for service factory."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
            mock_db.get_model_states = AsyncMock(return_value={})
            MockDatabase.return_value = mock_db

            service = await create_service()

            # Verify database was created and connected
            MockDatabase.assert_called_once()
            mock_db.connect.assert_called_once()
            mock_db.get_model_states.assert_called_once()

            # Verify service components are initialized
            assert service.database is mock_db
            assert service.analyzer is not None
            assert service.bandit is not None
            assert service.executor is not None
            assert service.router is not None

    @pytest.mark.asyncio
    async def test_create_service_with_provided_database(self):
        """Test create_service uses provided database."""
        mock_db = AsyncMock(spec=Database)
        mock_db.get_model_states = AsyncMock(return_value={})

        service = await create_service(database=mock_db)

        # Verify provided database was used
        assert service.database is mock_db
        mock_db.get_model_states.assert_called_once()

        # Verify service components are initialized
        assert service.analyzer is not None
        assert service.bandit is not None
        assert service.executor is not None
        assert service.router is not None

    @pytest.mark.asyncio
    async def test_create_service_loads_model_states(self):
        """Test create_service loads model states from database."""
        mock_db = AsyncMock(spec=Database)

        # Mock model states return
        from conduit.core.models import ModelState
        mock_states = {
            "gpt-4o-mini": ModelState(model_id="gpt-4o-mini", alpha=5.0, beta=2.0),
            "gpt-4o": ModelState(model_id="gpt-4o", alpha=3.0, beta=1.0),
        }
        mock_db.get_model_states = AsyncMock(return_value=mock_states)

        with patch("conduit.utils.service_factory.logger") as mock_logger:
            service = await create_service(database=mock_db)

            # Verify states were loaded into bandit
            mock_db.get_model_states.assert_called_once()
            mock_logger.info.assert_called_once()
            assert "Loaded 2 model states" in mock_logger.info.call_args[0][0]

            # Verify bandit has the states
            assert "gpt-4o-mini" in service.bandit.model_states
            assert "gpt-4o" in service.bandit.model_states
            assert service.bandit.model_states["gpt-4o-mini"].alpha == 5.0
            assert service.bandit.model_states["gpt-4o"].alpha == 3.0

    @pytest.mark.asyncio
    async def test_create_service_handles_model_state_load_failure(self):
        """Test create_service continues if model state loading fails."""
        mock_db = AsyncMock(spec=Database)
        mock_db.get_model_states = AsyncMock(side_effect=Exception("Database error"))

        with patch("conduit.utils.service_factory.logger") as mock_logger:
            service = await create_service(database=mock_db)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to load model states" in mock_logger.warning.call_args[0][0]

            # Verify service was still created successfully
            assert service.database is mock_db
            assert service.analyzer is not None
            assert service.bandit is not None

    @pytest.mark.asyncio
    async def test_create_service_with_custom_result_type(self):
        """Test create_service accepts custom result type."""
        from pydantic import BaseModel

        class CustomResult(BaseModel):
            result: str

        mock_db = AsyncMock(spec=Database)
        mock_db.get_model_states = AsyncMock(return_value={})

        service = await create_service(
            database=mock_db,
            default_result_type=CustomResult
        )

        # Verify custom result type was passed through
        assert service.default_result_type is CustomResult
