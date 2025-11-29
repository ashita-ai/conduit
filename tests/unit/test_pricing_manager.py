"""Unit tests for conduit.core.pricing_manager module.

Tests for PricingManager with three-tier fallback (Database → Cache → Fetch).
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from conduit.core.pricing import ModelPricing
from conduit.core.pricing_manager import PricingManager, CACHE_FILE


class TestPricingManagerCacheOnly:
    """Tests for PricingManager in cache-only mode (no database)."""

    @pytest.mark.asyncio
    async def test_get_pricing_from_cache(self, tmp_path):
        """Test loading pricing from local cache file."""
        # Create temporary cache file
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.1,
                    "output_cost_per_million": 4.4,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            manager = PricingManager(database=None)
            pricing = await manager.get_pricing()

            assert len(pricing) == 1
            assert "o4-mini" in pricing
            assert pricing["o4-mini"].input_cost_per_million == 1.1
            assert pricing["o4-mini"].output_cost_per_million == 4.4

    @pytest.mark.asyncio
    async def test_get_pricing_stale_cache_refreshes(self, tmp_path):
        """Test that stale cache triggers refresh from llm-prices.com."""
        # Create stale cache (> 24 hours old)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        cache_data = {
            "fetched_at": old_timestamp,
            "source_updated_at": "2025-11-23T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.0,
                    "output_cost_per_million": 4.0,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-23T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        # Mock direct fetch
        fresh_data = {
            "updated_at": "2025-11-24T00:00:00Z",
            "prices": [{
                "id": "o4-mini",
                "input": 1.1,
                "output": 4.4,
                "input_cached": None
            }]
        }

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            # Mock AsyncClient as async context manager
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = fresh_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                manager = PricingManager(database=None, cache_ttl_hours=24)
                pricing = await manager.get_pricing()

                # Should have fetched fresh data
                assert pricing["o4-mini"].input_cost_per_million == 1.1

    @pytest.mark.asyncio
    async def test_get_pricing_no_cache_fetches_direct(self, tmp_path):
        """Test direct fetch when cache file doesn't exist."""
        cache_file = tmp_path / "pricing.json"  # Doesn't exist

        fresh_data = {
            "updated_at": "2025-11-24T00:00:00Z",
            "prices": [{
                "id": "gpt-5",
                "input": 1.25,
                "output": 10.0,
                "input_cached": None
            }]
        }

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            # Mock AsyncClient as async context manager
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = fresh_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                manager = PricingManager(database=None)
                pricing = await manager.get_pricing()

                assert len(pricing) == 1
                assert "gpt-5" in pricing
                assert pricing["gpt-5"].input_cost_per_million == 1.25


class TestPricingManagerWithDatabase:
    """Tests for PricingManager with database fallback."""

    @pytest.mark.asyncio
    async def test_get_pricing_from_database(self):
        """Test loading pricing from database."""
        mock_db = AsyncMock()
        mock_db.pool = AsyncMock()

        # Mock database returning pricing
        db_pricing = {
            "claude-sonnet-4.5": ModelPricing(
                model_id="claude-sonnet-4.5",
                input_cost_per_million=3.0,
                output_cost_per_million=15.0,
                cached_input_cost_per_million=None,
                source="llm-prices.com",
                snapshot_at=datetime.now(timezone.utc)
            )
        }
        mock_db.get_latest_pricing.return_value = db_pricing

        manager = PricingManager(database=mock_db)
        pricing = await manager.get_pricing()

        assert len(pricing) == 1
        assert "claude-sonnet-4.5" in pricing
        assert pricing["claude-sonnet-4.5"].input_cost_per_million == 3.0
        mock_db.get_latest_pricing.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pricing_database_stale_warns(self):
        """Test database staleness warning when pricing > 7 days old."""
        mock_db = AsyncMock()
        mock_db.pool = AsyncMock()

        # Mock database returning stale pricing (8 days old)
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
        db_pricing = {
            "gpt-5": ModelPricing(
                model_id="gpt-5",
                input_cost_per_million=1.25,
                output_cost_per_million=10.0,
                cached_input_cost_per_million=None,
                source="llm-prices.com",
                snapshot_at=old_timestamp
            )
        }
        mock_db.get_latest_pricing.return_value = db_pricing

        manager = PricingManager(database=mock_db, database_stale_days=7)

        with patch("conduit.core.pricing_manager.logger") as mock_logger:
            pricing = await manager.get_pricing()

            # Should warn but still return pricing
            assert len(pricing) == 1
            mock_logger.warning.assert_called()
            assert "8 days old" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_get_pricing_database_fails_fallback_to_cache(self, tmp_path):
        """Test fallback to cache when database fails."""
        mock_db = AsyncMock()
        mock_db.pool = AsyncMock()
        mock_db.get_latest_pricing.side_effect = Exception("Database error")

        # Create cache file
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "gemini-2.5-flash": {
                    "input_cost_per_million": 0.3,
                    "output_cost_per_million": 2.5,
                    "cached_input_cost_per_million": 0.03,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            manager = PricingManager(database=mock_db)
            pricing = await manager.get_pricing()

            # Should have fallen back to cache
            assert "gemini-2.5-flash" in pricing
            assert pricing["gemini-2.5-flash"].input_cost_per_million == 0.3


class TestPricingManagerMemoryCache:
    """Tests for in-memory caching behavior."""

    @pytest.mark.asyncio
    async def test_in_memory_cache_hit(self, tmp_path):
        """Test that second call uses in-memory cache."""
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.1,
                    "output_cost_per_million": 4.4,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            manager = PricingManager(database=None)

            # First call loads from cache
            pricing1 = await manager.get_pricing()

            # Second call should use in-memory cache (same object instance)
            pricing2 = await manager.get_pricing()

            assert pricing1 is pricing2  # Same object

    @pytest.mark.asyncio
    async def test_in_memory_cache_expires_after_ttl(self, tmp_path):
        """Test that in-memory cache expires after cache_ttl_hours."""
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.1,
                    "output_cost_per_million": 4.4,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            manager = PricingManager(database=None, cache_ttl_hours=1)

            # First call loads from cache
            await manager.get_pricing()

            # Simulate time passing (> 1 hour)
            old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
            manager._cache_loaded_at = old_timestamp

            # Second call should reload (not use expired in-memory cache)
            with patch("conduit.core.pricing_manager.logger") as mock_logger:
                pricing2 = await manager.get_pricing()

                # Should log cache expiration
                assert any("expired" in str(call) for call in mock_logger.info.call_args_list)

    @pytest.mark.asyncio
    async def test_clear_cache_forces_reload(self, tmp_path):
        """Test clear_cache forces reload on next call."""
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.1,
                    "output_cost_per_million": 4.4,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            manager = PricingManager(database=None)

            # First call
            pricing1 = await manager.get_pricing()

            # Clear cache
            manager.clear_cache()

            # Next call should reload
            pricing2 = await manager.get_pricing()

            # Different instances (not same object from cache)
            assert pricing1 is not pricing2


class TestPricingManagerConfiguration:
    """Tests for PricingManager configuration options."""

    @pytest.mark.asyncio
    async def test_custom_cache_ttl(self, tmp_path):
        """Test custom cache_ttl_hours configuration."""
        # Cache is 2 hours old
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        cache_data = {
            "fetched_at": old_timestamp,
            "source_updated_at": "2025-11-24T00:00:00Z",
            "prices": {
                "o4-mini": {
                    "input_cost_per_million": 1.1,
                    "output_cost_per_million": 4.4,
                    "cached_input_cost_per_million": None,
                    "source": "llm-prices.com",
                    "snapshot_at": "2025-11-24T00:00:00Z"
                }
            }
        }

        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            # With cache_ttl_hours=1, this should trigger refresh
            manager = PricingManager(database=None, cache_ttl_hours=1)

            # Mock direct fetch
            fresh_data = {
                "updated_at": "2025-11-24T00:00:00Z",
                "prices": [{
                    "id": "o4-mini",
                    "input": 1.2,
                    "output": 4.5,
                    "input_cached": None
                }]
            }

            # Mock AsyncClient as async context manager
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = fresh_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                pricing = await manager.get_pricing()

                # Should have refreshed with new data
                assert pricing["o4-mini"].input_cost_per_million == 1.2

    @pytest.mark.asyncio
    async def test_custom_database_stale_days(self):
        """Test custom database_stale_days configuration."""
        mock_db = AsyncMock()
        mock_db.pool = AsyncMock()

        # Pricing is 3 days old
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=3)
        db_pricing = {
            "gpt-5": ModelPricing(
                model_id="gpt-5",
                input_cost_per_million=1.25,
                output_cost_per_million=10.0,
                cached_input_cost_per_million=None,
                source="llm-prices.com",
                snapshot_at=old_timestamp
            )
        }
        mock_db.get_latest_pricing.return_value = db_pricing

        # With database_stale_days=2, this should warn
        manager = PricingManager(database=mock_db, database_stale_days=2)

        with patch("conduit.core.pricing_manager.logger") as mock_logger:
            pricing = await manager.get_pricing()

            # Should warn about staleness
            mock_logger.warning.assert_called()
            assert "3 days old" in str(mock_logger.warning.call_args)


class TestPricingManagerErrorHandling:
    """Tests for PricingManager error handling."""

    @pytest.mark.asyncio
    async def test_all_sources_fail_raises_error(self):
        """Test that failure of all sources raises RuntimeError."""
        mock_db = AsyncMock()
        mock_db.pool = AsyncMock()
        mock_db.get_latest_pricing.side_effect = Exception("Database error")

        # No cache file
        with patch("conduit.core.pricing_manager.CACHE_FILE", Path("/nonexistent/pricing.json")):
            # Mock direct fetch failure
            with patch("httpx.AsyncClient.get", side_effect=Exception("Network error")):
                manager = PricingManager(database=mock_db)

                with pytest.raises(RuntimeError, match="Failed to load pricing from all sources"):
                    await manager.get_pricing()

    @pytest.mark.asyncio
    async def test_corrupted_cache_file_continues(self, tmp_path):
        """Test that corrupted cache file falls back to direct fetch."""
        cache_file = tmp_path / "pricing.json"
        cache_file.write_text("not valid json{{{")

        fresh_data = {
            "updated_at": "2025-11-24T00:00:00Z",
            "prices": [{
                "id": "o4-mini",
                "input": 1.1,
                "output": 4.4,
                "input_cached": None
            }]
        }

        with patch("conduit.core.pricing_manager.CACHE_FILE", cache_file):
            # Mock AsyncClient as async context manager
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = fresh_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                manager = PricingManager(database=None)
                pricing = await manager.get_pricing()

                # Should have successfully fetched despite corrupted cache
                assert "o4-mini" in pricing
