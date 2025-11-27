#!/usr/bin/env python3
"""Truncate all database tables."""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()


async def truncate_all_tables():
    """Truncate all tables in the database."""
    # Use direct connection for administrative tasks
    database_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ No DATABASE_URL or DIRECT_URL found in environment")
        return

    try:
        conn = await asyncpg.connect(database_url)
        print("✅ Connected to database")

        # Truncate tables in reverse dependency order
        tables = [
            "feedback",
            "implicit_feedback",
            "evaluation_metrics",
            "responses",
            "routing_decisions",
            "queries",
            "model_states",
            "bandit_state",
            "model_prices",
        ]

        for table in tables:
            try:
                await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                print(f"✅ Truncated {table}")
            except Exception as e:
                print(f"⚠️  Skipped {table}: {e}")

        await conn.close()
        print("\n✅ Database truncation complete")

    except Exception as e:
        print(f"❌ Failed to truncate database: {e}")


if __name__ == "__main__":
    asyncio.run(truncate_all_tables())
