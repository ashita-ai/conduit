"""Unit tests for conduit.api.app module.

Tests for FastAPI application factory and lifespan.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi(self):
        """Test create_app returns a FastAPI application."""
        with patch("conduit.api.app.setup_telemetry"):
            with patch("conduit.api.app.setup_middleware"):
                from conduit.api.app import create_app

                app = create_app()

                assert isinstance(app, FastAPI)
                assert app.title == "Conduit"
                assert app.version == "0.1.0"

    def test_create_app_has_exception_handler(self):
        """Test create_app registers global exception handler."""
        with patch("conduit.api.app.setup_telemetry"):
            with patch("conduit.api.app.setup_middleware"):
                from conduit.api.app import create_app

                app = create_app()

                # Check that exception handler is registered
                assert Exception in app.exception_handlers


class TestLifespan:
    """Tests for application lifespan management."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test lifespan manages startup and shutdown correctly."""
        from conduit.api.app import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with patch("conduit.api.app.Database") as MockDatabase:
            mock_db = AsyncMock()
            MockDatabase.return_value = mock_db

            with patch("conduit.api.app.Router") as MockRouter:
                mock_router = MagicMock()
                MockRouter.return_value = mock_router

                with patch("conduit.api.app.ModelExecutor") as MockExecutor:
                    mock_executor = MagicMock()
                    MockExecutor.return_value = mock_executor

                    with patch("conduit.api.app.create_routes") as mock_create_routes:
                        mock_api_router = MagicMock()
                        mock_create_routes.return_value = mock_api_router

                        with patch("conduit.api.app.shutdown_telemetry"):
                            with patch("conduit.api.app.settings") as mock_settings:
                                mock_settings.default_models = ["model-a"]
                                mock_settings.embedding_provider = "openai"
                                mock_settings.embedding_model = None
                                mock_settings.embedding_api_key = None

                                # Use lifespan as context manager
                                async with lifespan(mock_app):
                                    # Verify startup happened
                                    mock_db.connect.assert_called_once()
                                    MockRouter.assert_called_once()
                                    mock_create_routes.assert_called_once()
                                    mock_app.include_router.assert_called_once_with(
                                        mock_api_router
                                    )

                                # Verify shutdown happened
                                mock_db.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_loads_model_prices(self):
        """Test lifespan loads model prices from database."""
        from conduit.api.app import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with patch("conduit.api.app.Database") as MockDatabase:
            mock_db = AsyncMock()
            mock_db.get_model_prices.return_value = {"model-a": 0.01}
            MockDatabase.return_value = mock_db

            with patch("conduit.api.app.Router"):
                with patch("conduit.api.app.ModelExecutor") as MockExecutor:
                    with patch("conduit.api.app.create_routes") as mock_create_routes:
                        mock_create_routes.return_value = MagicMock()

                        with patch("conduit.api.app.shutdown_telemetry"):
                            with patch("conduit.api.app.settings") as mock_settings:
                                mock_settings.default_models = ["model-a"]
                                mock_settings.embedding_provider = "openai"
                                mock_settings.embedding_model = None
                                mock_settings.embedding_api_key = None

                                async with lifespan(mock_app):
                                    # Verify model prices were loaded
                                    mock_db.get_model_prices.assert_called_once()
                                    # Verify executor was created (pricing handled by LiteLLM)
                                    MockExecutor.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_lifespan_handles_price_loading_error(self):
        """Test lifespan handles model price loading errors gracefully."""
        from conduit.api.app import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with patch("conduit.api.app.Database") as MockDatabase:
            mock_db = AsyncMock()
            mock_db.get_model_prices.side_effect = Exception("DB error")
            MockDatabase.return_value = mock_db

            with patch("conduit.api.app.Router"):
                with patch("conduit.api.app.ModelExecutor") as MockExecutor:
                    with patch("conduit.api.app.create_routes") as mock_create_routes:
                        mock_create_routes.return_value = MagicMock()

                        with patch("conduit.api.app.shutdown_telemetry"):
                            with patch("conduit.api.app.settings") as mock_settings:
                                mock_settings.default_models = ["model-a"]
                                mock_settings.embedding_provider = "openai"
                                mock_settings.embedding_model = None
                                mock_settings.embedding_api_key = None

                                # Should not raise, gracefully handles error
                                async with lifespan(mock_app):
                                    MockExecutor.assert_called_once_with()


class TestExceptionHandler:
    """Tests for global exception handler."""

    @pytest.mark.asyncio
    async def test_exception_handler_returns_500(self):
        """Test exception handler returns 500 with error details."""
        with patch("conduit.api.app.setup_telemetry"):
            with patch("conduit.api.app.setup_middleware"):
                from conduit.api.app import create_app

                app = create_app()

                # Get the exception handler
                handler = app.exception_handlers[Exception]

                # Call it with a test exception
                response = await handler(MagicMock(), Exception("Test error"))

                assert response.status_code == 500
                body = response.body.decode()
                assert "Internal server error" in body
                assert "Test error" in body
