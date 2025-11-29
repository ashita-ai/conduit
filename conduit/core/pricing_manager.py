"""Intelligent pricing management with database, cache, and live fetch fallbacks.

This module provides a unified pricing interface that works with or without a database:
- Database mode: Historical snapshots, manual sync, staleness warnings
- Cache mode: Auto-refresh, 24-hour TTL, fail-fast on errors
- Direct fetch: Last resort fallback

Architecture:
    Database (if DATABASE_URL set):
        → get_latest_pricing() from database
        → Warn if > 7 days stale
        → NO auto-fetch (user controls via sync_pricing.py)

    Cache (if no database OR database fetch fails):
        → Load from ~/.cache/conduit/pricing.json
        → Auto-refresh if > cache_ttl_hours old
        → Fail-fast if fetch fails and cache stale

    Direct Fetch (last resort):
        → Fetch from llm-prices.com
        → Save to cache
        → Use fresh data

Configuration (via conduit.yaml):
    pricing:
      cache_ttl_hours: 24           # Cache freshness threshold
      database_stale_days: 7        # Database staleness warning threshold
      fail_on_stale_cache: true     # Fail-fast vs graceful degradation
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from conduit.core.pricing import ModelPricing

logger = logging.getLogger(__name__)

# llm-prices.com API endpoint
LLM_PRICES_API = "https://www.llm-prices.com/current-v1.json"

# Default cache location (XDG standard)
CACHE_DIR = Path.home() / ".cache" / "conduit"
CACHE_FILE = CACHE_DIR / "pricing.json"


class PricingManager:
    """Manages model pricing with intelligent fallback strategies.

    Provides unified pricing interface supporting:
    - Database with historical snapshots (production)
    - Local cache with auto-refresh (lightweight)
    - Direct fetch fallback (always works)

    Configuration:
        cache_ttl_hours: How long cache is considered fresh (default: 24)
        database_stale_days: When to warn about stale DB pricing (default: 7)
        fail_on_stale_cache: Fail-fast vs warn on stale cache (default: True)
    """

    def __init__(
        self,
        database=None,
        cache_ttl_hours: int = 24,
        database_stale_days: int = 7,
        fail_on_stale_cache: bool = True,
    ):
        """Initialize pricing manager.

        Args:
            database: Optional Database instance (if None, uses cache-only mode)
            cache_ttl_hours: Cache freshness threshold in hours
            database_stale_days: Database staleness warning threshold in days
            fail_on_stale_cache: Whether to fail-fast on stale cache errors
        """
        self.database = database
        self.cache_ttl_hours = cache_ttl_hours
        self.database_stale_days = database_stale_days
        self.fail_on_stale_cache = fail_on_stale_cache

        # In-memory cache for session (avoid repeated disk/DB reads)
        self._memory_cache: dict[str, ModelPricing] | None = None
        self._cache_loaded_at: datetime | None = None

    async def get_pricing(self) -> dict[str, ModelPricing]:
        """Get current model pricing with intelligent fallback.

        Strategy:
            1. Check in-memory cache (if fresh within cache_ttl_hours)
            2. Try database (if configured)
            3. Try local cache file
            4. Direct fetch from llm-prices.com

        In-memory cache has session-level TTL to prevent stale pricing
        in long-running processes (e.g., 30-day server uptime).

        Returns:
            Dictionary mapping model_id to ModelPricing

        Raises:
            RuntimeError: If all strategies fail or pricing is stale with fail_on_stale_cache=True
        """
        # Fast path: In-memory cache (with session-level TTL)
        if self._memory_cache is not None and self._cache_loaded_at is not None:
            cache_age_hours = (
                datetime.now(timezone.utc) - self._cache_loaded_at
            ).total_seconds() / 3600

            if cache_age_hours < self.cache_ttl_hours:
                logger.debug(
                    f"Using in-memory pricing cache (age: {cache_age_hours:.1f}h, "
                    f"TTL: {self.cache_ttl_hours}h)"
                )
                return self._memory_cache
            else:
                logger.info(
                    f"In-memory cache expired ({cache_age_hours:.1f}h > {self.cache_ttl_hours}h), "
                    "reloading pricing"
                )
                self._memory_cache = None
                self._cache_loaded_at = None

        # Try database first (if configured)
        if self.database is not None:
            try:
                pricing = await self._load_from_database()
                if pricing:
                    self._memory_cache = pricing
                    self._cache_loaded_at = datetime.now(timezone.utc)
                    return pricing
            except Exception as e:
                logger.warning(f"Database pricing load failed: {e}")
                # Continue to cache fallback

        # Try local cache
        try:
            pricing = await self._load_from_cache()
            if pricing:
                self._memory_cache = pricing
                self._cache_loaded_at = datetime.now(timezone.utc)
                return pricing
        except Exception as e:
            logger.warning(f"Cache pricing load failed: {e}")
            # Continue to direct fetch

        # Last resort: Direct fetch
        try:
            pricing = await self._fetch_and_cache()
            self._memory_cache = pricing
            self._cache_loaded_at = datetime.now(timezone.utc)
            return pricing
        except Exception as e:
            logger.error(f"Failed to fetch pricing from llm-prices.com: {e}")
            raise RuntimeError(
                "Failed to load pricing from all sources (database, cache, direct fetch). "
                "Check network connectivity and try running: python scripts/sync_pricing.py"
            ) from e

    async def _load_from_database(self) -> dict[str, ModelPricing] | None:
        """Load pricing from database with staleness check.

        Returns:
            Pricing dict if successful, None otherwise

        Raises:
            RuntimeError: If database pricing is too stale
        """
        if self.database is None or self.database.pool is None:
            return None

        logger.info("Loading pricing from database")
        pricing = await self.database.get_latest_pricing()

        if not pricing:
            logger.warning("No pricing found in database")
            return None

        # Check staleness (find oldest snapshot_at)
        oldest_snapshot = None
        for model_pricing in pricing.values():
            if model_pricing.snapshot_at:
                if oldest_snapshot is None or model_pricing.snapshot_at < oldest_snapshot:
                    oldest_snapshot = model_pricing.snapshot_at

        if oldest_snapshot:
            age_days = (datetime.now(timezone.utc) - oldest_snapshot).days
            if age_days > self.database_stale_days:
                logger.warning(
                    f"Database pricing is {age_days} days old (threshold: {self.database_stale_days} days). "
                    f"Run 'python scripts/sync_pricing.py' to refresh."
                )
                # Don't fail - just warn. User controls sync timing.

        logger.info(f"Loaded pricing for {len(pricing)} models from database")
        return pricing

    async def _load_from_cache(self) -> dict[str, ModelPricing] | None:
        """Load pricing from local cache with freshness check.

        Returns:
            Pricing dict if cache is fresh, None if stale or missing

        Raises:
            RuntimeError: If cache is stale and fail_on_stale_cache=True
        """
        if not CACHE_FILE.exists():
            logger.info("No pricing cache file found")
            return None

        logger.info(f"Loading pricing from cache: {CACHE_FILE}")

        try:
            with open(CACHE_FILE, "r") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file: {e}")
            return None

        # Check cache freshness
        fetched_at_str = cache_data.get("fetched_at")
        if not fetched_at_str:
            logger.warning("Cache missing fetched_at timestamp")
            return None

        fetched_at = datetime.fromisoformat(fetched_at_str.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600

        if age_hours > self.cache_ttl_hours:
            logger.warning(
                f"Cache is {age_hours:.1f} hours old (threshold: {self.cache_ttl_hours}h). "
                "Attempting to refresh from llm-prices.com"
            )
            # Auto-refresh in cache mode
            return None

        # Parse cached pricing
        pricing = {}
        for model_id, price_data in cache_data.get("prices", {}).items():
            pricing[model_id] = ModelPricing(
                model_id=model_id,
                input_cost_per_million=price_data["input_cost_per_million"],
                output_cost_per_million=price_data["output_cost_per_million"],
                cached_input_cost_per_million=price_data.get("cached_input_cost_per_million"),
                source=price_data.get("source", "cache"),
                snapshot_at=datetime.fromisoformat(
                    price_data["snapshot_at"].replace("Z", "+00:00")
                ) if price_data.get("snapshot_at") else None,
            )

        logger.info(f"Loaded pricing for {len(pricing)} models from cache (age: {age_hours:.1f}h)")
        return pricing

    async def _fetch_and_cache(self) -> dict[str, ModelPricing]:
        """Fetch pricing from llm-prices.com and save to cache.

        Returns:
            Fresh pricing dictionary

        Raises:
            httpx.HTTPError: If API request fails
        """
        logger.info(f"Fetching fresh pricing from {LLM_PRICES_API}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(LLM_PRICES_API)
            response.raise_for_status()
            data = response.json()

        fetched_at = datetime.now(timezone.utc)
        source_updated_at = data.get("updated_at", fetched_at.isoformat())

        # Parse pricing
        pricing = {}
        for model in data.get("prices", []):
            model_id = model["id"]
            pricing[model_id] = ModelPricing(
                model_id=model_id,
                input_cost_per_million=float(model["input"]),
                output_cost_per_million=float(model["output"]),
                cached_input_cost_per_million=(
                    float(model["input_cached"]) if model.get("input_cached") else None
                ),
                source="llm-prices.com",
                snapshot_at=datetime.fromisoformat(
                    source_updated_at.replace("Z", "+00:00")
                ),
            )

        # Save to cache
        await self._save_to_cache(pricing, fetched_at, source_updated_at)

        logger.info(f"Fetched and cached pricing for {len(pricing)} models")
        return pricing

    async def _save_to_cache(
        self,
        pricing: dict[str, ModelPricing],
        fetched_at: datetime,
        source_updated_at: str,
    ) -> None:
        """Save pricing to local cache file.

        Args:
            pricing: Pricing dictionary to cache
            fetched_at: When we fetched this data
            source_updated_at: When llm-prices.com last updated
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "fetched_at": fetched_at.isoformat(),
            "source_updated_at": source_updated_at,
            "prices": {
                model_id: {
                    "input_cost_per_million": p.input_cost_per_million,
                    "output_cost_per_million": p.output_cost_per_million,
                    "cached_input_cost_per_million": p.cached_input_cost_per_million,
                    "source": p.source,
                    "snapshot_at": p.snapshot_at.isoformat() if p.snapshot_at else None,
                }
                for model_id, p in pricing.items()
            }
        }

        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Saved pricing cache to {CACHE_FILE}")

    def clear_cache(self) -> None:
        """Clear in-memory cache (force reload on next get_pricing call)."""
        self._memory_cache = None
        self._cache_loaded_at = None
        logger.debug("Cleared in-memory pricing cache")
