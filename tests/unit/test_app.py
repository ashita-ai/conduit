"""Unit tests for FastAPI app factory."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from conduit.api.app import create_app, lifespan


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self):
        """Test lifespan successfully initializes all components on startup."""
        from fastapi import FastAPI

        app = FastAPI()

        with patch("conduit.api.app.Database") as MockDatabase:
            with patch("conduit.api.app.QueryAnalyzer") as MockAnalyzer:
                with patch("conduit.api.app.ContextualBandit") as MockBandit:
                    with patch("conduit.api.app.ModelExecutor") as MockExecutor:
                        with patch("conduit.api.app.RoutingEngine") as MockRoutingEngine:
                            with patch("conduit.api.app.RoutingService") as MockService:
                                # Setup mocks
                                mock_db = AsyncMock()
                                mock_db.connect = AsyncMock()
                                mock_db.get_model_states = AsyncMock(return_value={})
                                mock_db.get_model_prices = AsyncMock(return_value={})
                                MockDatabase.return_value = mock_db

                                mock_analyzer = MagicMock()
                                MockAnalyzer.return_value = mock_analyzer

                                mock_bandit = MagicMock()
                                mock_bandit.load_states = MagicMock()
                                MockBandit.return_value = mock_bandit

                                mock_executor = MagicMock()
                                MockExecutor.return_value = mock_executor

                                mock_routing_engine = MagicMock()
                                MockRoutingEngine.return_value = mock_routing_engine

                                mock_service = MagicMock()
                                MockService.return_value = mock_service

                                # Run lifespan
                                async with lifespan(app):
                                    # Verify startup sequence
                                    MockDatabase.assert_called_once()
                                    mock_db.connect.assert_called_once()
                                    mock_db.get_model_states.assert_called_once()
                                    mock_db.get_model_prices.assert_called_once()

                                    MockAnalyzer.assert_called_once()
                                    MockBandit.assert_called_once()
                                    MockExecutor.assert_called_once()
                                    MockRoutingEngine.assert_called_once()
                                    MockService.assert_called_once()

                                    mock_bandit.load_states.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_loads_model_states(self):
        """Test lifespan loads model states from database."""
        from fastapi import FastAPI
        from conduit.core.models import ModelState

        app = FastAPI()
        mock_states = {
            "gpt-4o-mini": ModelState(model_id="gpt-4o-mini", alpha=5.0, beta=2.0),
        }

        with patch("conduit.api.app.Database") as MockDatabase:
            with patch("conduit.api.app.QueryAnalyzer"):
                with patch("conduit.api.app.ContextualBandit") as MockBandit:
                    with patch("conduit.api.app.ModelExecutor"):
                        with patch("conduit.api.app.RoutingEngine"):
                            with patch("conduit.api.app.RoutingService"):
                                with patch("conduit.api.app.logger") as mock_logger:
                                    mock_db = AsyncMock()
                                    mock_db.connect = AsyncMock()
                                    mock_db.get_model_states = AsyncMock(return_value=mock_states)
                                    mock_db.get_model_prices = AsyncMock(return_value={})
                                    MockDatabase.return_value = mock_db

                                    mock_bandit = MagicMock()
                                    mock_bandit.load_states = MagicMock()
                                    MockBandit.return_value = mock_bandit

                                    async with lifespan(app):
                                        # Verify states were loaded
                                        mock_bandit.load_states.assert_called_once_with(mock_states)
                                        # Verify logging
                                        assert any("Loaded 1 model states" in str(call) for call in mock_logger.info.call_args_list)

    @pytest.mark.asyncio
    async def test_lifespan_handles_missing_model_states(self):
        """Test lifespan handles when no model states exist."""
        from fastapi import FastAPI

        app = FastAPI()

        with patch("conduit.api.app.Database") as MockDatabase:
            with patch("conduit.api.app.QueryAnalyzer"):
                with patch("conduit.api.app.ContextualBandit") as MockBandit:
                    with patch("conduit.api.app.ModelExecutor"):
                        with patch("conduit.api.app.RoutingEngine"):
                            with patch("conduit.api.app.RoutingService"):
                                with patch("conduit.api.app.logger") as mock_logger:
                                    mock_db = AsyncMock()
                                    mock_db.connect = AsyncMock()
                                    mock_db.get_model_states = AsyncMock(return_value={})
                                    mock_db.get_model_prices = AsyncMock(return_value={})
                                    MockDatabase.return_value = mock_db

                                    mock_bandit = MagicMock()
                                    mock_bandit.load_states = MagicMock()
                                    MockBandit.return_value = mock_bandit

                                    async with lifespan(app):
                                        # Verify empty states were loaded
                                        mock_bandit.load_states.assert_called_once_with({})
                                        # Verify info logged (empty states still logs success)
                                        assert any("Loaded 0 model states" in str(call) for call in mock_logger.info.call_args_list)


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test create_app returns a FastAPI application."""
        from fastapi import FastAPI

        app = create_app()

        assert isinstance(app, FastAPI)
        assert app.title == "Conduit"
        assert app.version == "0.1.0"

    def test_create_app_configures_middleware(self):
        """Test create_app sets up middleware."""
        with patch("conduit.api.app.setup_middleware") as mock_setup:
            app = create_app()

            # Verify middleware setup was called
            mock_setup.assert_called_once_with(app)

    def test_create_app_has_exception_handler(self):
        """Test create_app registers global exception handler."""
        app = create_app()

        # Verify exception handler is registered
        assert app.exception_handlers is not None
        assert Exception in app.exception_handlers
