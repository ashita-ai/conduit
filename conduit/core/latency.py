"""Latency tracking and prediction service for historical performance data.

This module provides a service for recording actual model latencies and
estimating future latencies based on historical patterns, replacing hardcoded
provider-based heuristics with real performance data.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from conduit.core.exceptions import DatabaseError
from conduit.core.models import QueryFeatures

logger = logging.getLogger(__name__)


class LatencyService:
    """Track and predict model latencies from historical data.

    This service records actual latency observations from LLM calls and provides
    estimates based on statistical analysis of historical performance patterns.
    It supports percentile-based estimation (p50, p95, p99) with optional
    feature-based segmentation for more accurate predictions.

    Example:
        >>> service = LatencyService(db_pool)
        >>> # Record actual latency after LLM call
        >>> await service.record_latency(
        ...     model_id="gpt-4o-mini",
        ...     latency=1.23,
        ...     query_features=features,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> # Get estimated latency for routing decision
        >>> estimate = await service.get_estimated_latency(
        ...     model_id="gpt-4o-mini",
        ...     query_features=features
        ... )
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        window_days: int = 7,
        percentile: float = 0.95,
        min_samples: int = 100,
    ):
        """Initialize latency service.

        Args:
            pool: PostgreSQL connection pool
            window_days: Number of days of history to consider (default: 7)
            percentile: Percentile for latency estimation (default: 0.95 for p95)
            min_samples: Minimum samples required before using historical data,
                otherwise falls back to heuristics (default: 100)
        """
        self.pool = pool
        self.window_days = window_days
        self.percentile = percentile
        self.min_samples = min_samples
        self._provider_baselines: dict[str, float] = {
            "openai": 1.5,
            "anthropic": 1.8,
            "google": 1.2,
            "groq": 0.5,
            "mistral": 1.3,
            "cohere": 1.4,
        }

    async def record_latency(
        self,
        model_id: str,
        latency: float,
        query_features: QueryFeatures | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record actual latency observation.

        Args:
            model_id: Model identifier (e.g., "gpt-4o-mini")
            latency: Actual latency in seconds
            query_features: Optional query features for context-aware tracking
            timestamp: Observation timestamp (defaults to now)

        Raises:
            DatabaseError: If recording fails
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Extract optional features for segmentation
        token_count = query_features.token_count if query_features else None
        complexity_score = (
            query_features.complexity_score if query_features else None
        )

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO model_latencies
                        (model_id, latency_seconds, token_count, complexity_score, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    model_id,
                    latency,
                    token_count,
                    complexity_score,
                    timestamp,
                )

            logger.debug(
                f"Recorded latency for {model_id}: {latency:.3f}s "
                f"(tokens={token_count}, complexity={complexity_score})"
            )

        except Exception as e:
            logger.error(f"Failed to record latency for {model_id}: {e}")
            raise DatabaseError(f"Failed to record latency: {e}") from e

    async def get_estimated_latency(
        self,
        model_id: str,
        query_features: QueryFeatures | None = None,
    ) -> float:
        """Get latency estimate based on historical data.

        Returns the configured percentile (default p95) of latencies from the
        last N days. If insufficient historical data exists, falls back to
        provider-based heuristics.

        Args:
            model_id: Model identifier
            query_features: Optional features for context-aware estimation
                (currently not used for segmentation, reserved for future)

        Returns:
            Estimated latency in seconds

        Estimation Strategy:
            1. Query historical data from last window_days
            2. If samples >= min_samples, return configured percentile
            3. Otherwise, fall back to provider baseline heuristics
        """
        try:
            # Query historical latencies within time window
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.window_days)

            async with self.pool.acquire() as conn:
                # Get count and percentile in one query for efficiency
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as sample_count,
                        PERCENTILE_CONT($1) WITHIN GROUP (ORDER BY latency_seconds) as percentile_latency
                    FROM model_latencies
                    WHERE model_id = $2
                      AND created_at >= $3
                    """,
                    self.percentile,
                    model_id,
                    cutoff,
                )

            sample_count = row["sample_count"] if row else 0
            percentile_latency = row["percentile_latency"] if row else None

            # Use historical data if we have enough samples
            if sample_count >= self.min_samples and percentile_latency is not None:
                logger.debug(
                    f"Using historical latency for {model_id}: "
                    f"{float(percentile_latency):.3f}s "
                    f"(p{int(self.percentile * 100)}, n={sample_count})"
                )
                return float(percentile_latency)

            # Fall back to heuristics
            logger.debug(
                f"Insufficient historical data for {model_id} "
                f"(n={sample_count} < {self.min_samples}), using heuristic"
            )
            return self._estimate_latency_heuristic(model_id)

        except Exception as e:
            logger.error(f"Failed to get latency estimate for {model_id}: {e}")
            # Fall back to heuristic on error
            return self._estimate_latency_heuristic(model_id)

    async def get_latency_stats(
        self, model_id: str
    ) -> dict[str, float | int]:
        """Get comprehensive latency statistics for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with statistics:
                - p50: Median latency (seconds)
                - p95: 95th percentile latency (seconds)
                - p99: 99th percentile latency (seconds)
                - mean: Average latency (seconds)
                - min: Minimum latency (seconds)
                - max: Maximum latency (seconds)
                - sample_count: Number of observations
                - window_days: Time window used

        Raises:
            DatabaseError: If query fails
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.window_days)

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as sample_count,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_seconds) as p50,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_seconds) as p95,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_seconds) as p99,
                        AVG(latency_seconds) as mean,
                        MIN(latency_seconds) as min,
                        MAX(latency_seconds) as max
                    FROM model_latencies
                    WHERE model_id = $1
                      AND created_at >= $2
                    """,
                    model_id,
                    cutoff,
                )

            if not row or row["sample_count"] == 0:
                return {
                    "sample_count": 0,
                    "window_days": self.window_days,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "mean": None,
                    "min": None,
                    "max": None,
                }

            return {
                "sample_count": int(row["sample_count"]),
                "window_days": self.window_days,
                "p50": float(row["p50"]) if row["p50"] is not None else None,
                "p95": float(row["p95"]) if row["p95"] is not None else None,
                "p99": float(row["p99"]) if row["p99"] is not None else None,
                "mean": float(row["mean"]) if row["mean"] is not None else None,
                "min": float(row["min"]) if row["min"] is not None else None,
                "max": float(row["max"]) if row["max"] is not None else None,
            }

        except Exception as e:
            logger.error(f"Failed to get latency stats for {model_id}: {e}")
            raise DatabaseError(f"Failed to get latency stats: {e}") from e

    def _estimate_latency_heuristic(self, model_id: str) -> float:
        """Estimate latency using provider-based heuristics (fallback only).

        This method is used as a fallback when insufficient historical data
        is available. It uses hardcoded provider baselines as rough estimates.

        Args:
            model_id: Model identifier

        Returns:
            Estimated latency in seconds
        """
        # Extract provider from model_id
        # Common patterns: "gpt-4o-mini" -> "openai", "claude-3-opus" -> "anthropic"
        provider = self._extract_provider(model_id)

        # Get baseline or use conservative default
        baseline = self._provider_baselines.get(provider, 2.0)

        # Apply cost-based multiplier for premium/fast models
        # Check for fast models first (more specific match)
        if any(x in model_id.lower() for x in ["mini", "haiku", "flash"]):
            baseline *= 0.7
        # Then check for premium models
        elif any(x in model_id.lower() for x in ["opus", "gpt-4o", "claude-3-5", "claude-3.5"]):
            baseline *= 1.3

        logger.debug(
            f"Heuristic latency estimate for {model_id}: {baseline:.3f}s "
            f"(provider={provider})"
        )

        return baseline

    def _extract_provider(self, model_id: str) -> str:
        """Extract provider name from model_id.

        Args:
            model_id: Model identifier

        Returns:
            Provider name (openai, anthropic, google, etc.)
        """
        model_lower = model_id.lower()

        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "llama" in model_lower and "groq" in model_lower:
            return "groq"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "command" in model_lower or "cohere" in model_lower:
            return "cohere"
        else:
            return "unknown"
