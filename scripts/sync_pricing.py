#!/usr/bin/env python3
"""Sync LLM pricing data from llm-prices.com to Supabase.

This script fetches current pricing for all LLM models from the community-maintained
llm-prices.com API and updates the model_prices table in Supabase.

Usage:
    python scripts/sync_pricing.py [--dry-run] [--verbose]

Features:
- Fetches pricing from https://www.llm-prices.com/current-v1.json
- Updates existing prices or inserts new models
- Preserves source attribution and snapshot timestamp
- Supports dry-run mode for validation
- Handles cached input pricing when available

Source: https://github.com/simonw/llm-prices
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import httpx
from supabase import acreate_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# llm-prices.com API endpoint
LLM_PRICES_API = "https://www.llm-prices.com/current-v1.json"


async def fetch_pricing_data() -> dict[str, Any]:
    """Fetch current pricing data from llm-prices.com API.

    Returns:
        Dictionary with 'updated_at' and 'prices' keys

    Raises:
        httpx.HTTPError: If API request fails
        json.JSONDecodeError: If response is not valid JSON
    """
    logger.info(f"Fetching pricing data from {LLM_PRICES_API}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(LLM_PRICES_API)
        response.raise_for_status()
        data = response.json()

    logger.info(
        f"Fetched {len(data['prices'])} models (updated: {data['updated_at']})"
    )
    return data


def map_model_to_pricing_record(model: dict[str, Any], snapshot_at: str) -> dict[str, Any]:
    """Map llm-prices.com model data to our model_prices schema.

    Args:
        model: Model data from API (id, vendor, name, input, output, input_cached)
        snapshot_at: ISO 8601 timestamp for this pricing snapshot

    Returns:
        Dictionary matching model_prices table schema
    """
    return {
        "model_id": model["id"],
        "input_cost_per_million": float(model["input"]),
        "output_cost_per_million": float(model["output"]),
        "cached_input_cost_per_million": (
            float(model["input_cached"]) if model.get("input_cached") else None
        ),
        "source": "llm-prices.com",
        "snapshot_at": snapshot_at,
    }


def deduplicate_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate models by model_id, keeping the last occurrence.

    Args:
        models: List of model pricing records

    Returns:
        Deduplicated list of models
    """
    seen = {}
    for model in models:
        seen[model["model_id"]] = model  # Last one wins
    return list(seen.values())


async def sync_pricing_to_database(
    pricing_data: dict[str, Any], dry_run: bool = False
) -> tuple[int, int]:
    """Sync pricing data to Supabase model_prices table.

    Args:
        pricing_data: Data from llm-prices.com API
        dry_run: If True, don't actually update database

    Returns:
        Tuple of (models_updated, models_inserted)

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_ANON_KEY not set
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment"
        )

    # Connect to Supabase
    client = await acreate_client(supabase_url, supabase_key)

    snapshot_at = pricing_data["updated_at"]
    models_raw = [
        map_model_to_pricing_record(model, snapshot_at)
        for model in pricing_data["prices"]
    ]

    # Deduplicate in case source data has duplicates
    models_to_sync = deduplicate_models(models_raw)
    if len(models_raw) != len(models_to_sync):
        logger.warning(
            f"Deduplicated {len(models_raw) - len(models_to_sync)} duplicate models"
        )

    logger.info(f"Syncing {len(models_to_sync)} models to database")

    if dry_run:
        logger.info("DRY RUN - Would update/insert the following models:")
        for model in models_to_sync[:10]:  # Show first 10
            logger.info(
                f"  {model['model_id']}: ${model['input_cost_per_million']}/M in, "
                f"${model['output_cost_per_million']}/M out"
            )
        if len(models_to_sync) > 10:
            logger.info(f"  ... and {len(models_to_sync) - 10} more")
        return (0, 0)

    # Use upsert to handle both inserts and updates in one operation
    # This is more efficient and handles conflicts automatically
    logger.info(f"Upserting {len(models_to_sync)} models (insert or update)")

    await client.table("model_prices").upsert(
        models_to_sync, on_conflict="model_id"
    ).execute()

    logger.info(f"Sync complete: {len(models_to_sync)} models synchronized")
    return (len(models_to_sync), 0)  # Can't distinguish inserts vs updates with upsert


async def main() -> int:
    """Main entry point for pricing sync script."""
    parser = argparse.ArgumentParser(
        description="Sync LLM pricing from llm-prices.com to Supabase"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without updating database",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Fetch pricing data
        pricing_data = await fetch_pricing_data()

        # Sync to database
        updated, inserted = await sync_pricing_to_database(
            pricing_data, dry_run=args.dry_run
        )

        if args.dry_run:
            logger.info("Dry run complete - no changes made")
        else:
            logger.info(f"Pricing sync successful: {inserted} new, {updated} updated")

        return 0

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch pricing data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
