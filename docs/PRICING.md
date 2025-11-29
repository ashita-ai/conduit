# Pricing Architecture

**Last Updated**: 2025-11-29

Conduit's pricing infrastructure provides automatic model cost tracking with intelligent fallback strategies for maximum reliability.

## Overview

The pricing system uses a three-tier fallback architecture that works with or without a database:

```
Database → Local Cache → Direct Fetch from llm-prices.com
```

This ensures pricing is always available, whether you're running in production with full database infrastructure or developing locally without any setup.

## Architecture

### PricingManager

The `PricingManager` class (conduit/core/pricing_manager.py:53) orchestrates pricing retrieval with intelligent fallbacks:

**Three-Tier Strategy**:
1. **Database** (if configured): Load latest pricing snapshots with staleness warnings
2. **Local Cache** (~/.cache/conduit/pricing.json): Auto-refreshing file-based cache
3. **Direct Fetch** (llm-prices.com): Last resort fallback

**In-Memory Caching**: Session-level TTL prevents stale pricing in long-running processes (e.g., 30-day server uptime).

### Historical Pricing Support

Conduit stores historical pricing snapshots to explain routing decisions over time:

**Database Schema** (migrations/versions/34c6749675ef):
- `id`: Auto-incrementing primary key
- `model_id`: Model identifier (e.g., "gpt-5", "claude-sonnet-4.5")
- `input_cost_per_million`: Input cost per 1M tokens
- `output_cost_per_million`: Output cost per 1M tokens
- `cached_input_cost_per_million`: Optional cached input cost
- `source`: Pricing source (e.g., "llm-prices.com")
- `snapshot_at`: Timestamp of this pricing snapshot
- **UNIQUE(model_id, snapshot_at)**: Ensures one snapshot per model per timestamp

**Query Helper** (conduit/core/database.py:365):
```python
async def get_latest_pricing(self) -> dict[str, ModelPricing]:
    """Load latest pricing snapshot for each model using DISTINCT ON."""
```

## Configuration

Edit `conduit.yaml` to customize pricing behavior:

```yaml
pricing:
  # Cache freshness threshold (hours)
  # When using cache-only mode (no database), pricing is auto-refreshed if older than this
  cache_ttl_hours: 24

  # Database staleness warning threshold (days)
  # When using database mode, warn if latest pricing snapshot is older than this
  database_stale_days: 7

  # Fail-fast on stale cache errors
  # If true, fail immediately when cache is stale and llm-prices.com fetch fails
  # If false, use stale cache with warning (graceful degradation)
  fail_on_stale_cache: true
```

### Configuration Examples

**Production (Database-Backed)**:
```yaml
pricing:
  cache_ttl_hours: 24
  database_stale_days: 7
  fail_on_stale_cache: true
```
- Fresh pricing from database
- Weekly staleness warnings
- Fails fast on errors (prefer reliability)

**Development (Cache-Only)**:
```yaml
pricing:
  cache_ttl_hours: 168  # 1 week
  database_stale_days: 7
  fail_on_stale_cache: false
```
- Relaxed cache expiration for local dev
- Graceful degradation (continue with stale pricing)

**High-Frequency Trading**:
```yaml
pricing:
  cache_ttl_hours: 1
  database_stale_days: 1
  fail_on_stale_cache: true
```
- Hourly cache refresh
- Daily database staleness check
- Strict freshness requirements

## Usage

### Syncing Pricing to Database

Use `scripts/sync_pricing.py` to fetch latest pricing from llm-prices.com and store historical snapshots:

```bash
# Fetch and store latest pricing
python scripts/sync_pricing.py

# Force sync (bypass duplicate check)
python scripts/sync_pricing.py --force

# Dry run (preview changes without writing)
python scripts/sync_pricing.py --dry-run
```

**Behavior**:
- Checks if current snapshot date already exists (prevents duplicates)
- Inserts new pricing snapshots (preserves history)
- Never overwrites existing pricing records
- Use `--force` to insert duplicate snapshot (not recommended)

**Recommended Schedule**: Run daily or weekly to maintain fresh pricing.

### Programmatic Usage

```python
from conduit.core.pricing_manager import PricingManager
from conduit.core.database import Database

# With database (production)
database = Database(database_url="postgresql://...")
await database.connect()
pricing_manager = PricingManager(database=database)

# Without database (development)
pricing_manager = PricingManager(database=None)

# Get current pricing (handles all fallbacks automatically)
pricing = await pricing_manager.get_pricing()

# Access pricing for specific model
gpt5_pricing = pricing["gpt-5"]
print(f"Input: ${gpt5_pricing.input_cost_per_million}/M tokens")
print(f"Output: ${gpt5_pricing.output_cost_per_million}/M tokens")

# Clear in-memory cache (force reload on next call)
pricing_manager.clear_cache()
```

## Fallback Behavior

### Database Mode (DATABASE_URL set)

```
1. Try database.get_latest_pricing()
   ↓ (if fails)
2. Try local cache (~/.cache/conduit/pricing.json)
   ↓ (if fails)
3. Direct fetch from llm-prices.com
   ↓ (if fails)
4. RuntimeError: "Failed to load pricing from all sources"
```

**Database Staleness**:
- Warns if pricing > `database_stale_days` old
- Does NOT auto-fetch (user controls sync timing via `sync_pricing.py`)
- Continues with stale database pricing (no interruption)

### Cache-Only Mode (no DATABASE_URL)

```
1. Try local cache (~/.cache/conduit/pricing.json)
   ↓ (if stale or missing)
2. Direct fetch from llm-prices.com
   ↓ (save to cache)
3. Use fresh data
   ↓ (if fetch fails)
4. If fail_on_stale_cache=true: Fail
   If fail_on_stale_cache=false: Use stale cache with warning
```

**Cache Auto-Refresh**:
- Automatically refreshes when cache > `cache_ttl_hours` old
- Saves fetched pricing to ~/.cache/conduit/pricing.json
- Transparent to calling code

## Session-Level Caching

The `PricingManager` maintains an in-memory cache with TTL to prevent serving stale pricing in long-running processes (conduit/core/pricing_manager.py:110):

```python
# Check in-memory cache with session-level TTL
if self._memory_cache is not None and self._cache_loaded_at is not None:
    cache_age_hours = (
        datetime.now(timezone.utc) - self._cache_loaded_at
    ).total_seconds() / 3600

    if cache_age_hours < self.cache_ttl_hours:
        return self._memory_cache  # Fresh
    else:
        # Expired - reload from database/cache/fetch
        self._memory_cache = None
```

**Why This Matters**: Without session-level TTL, a server running for 30 days would use pricing from day 1 forever, never refreshing.

## Pricing Data Source

Conduit fetches pricing from **llm-prices.com** (https://www.llm-prices.com/current-v1.json), a community-maintained database of LLM pricing.

**Data Format**:
```json
{
  "updated_at": "2025-11-24T00:00:00Z",
  "prices": [
    {
      "id": "gpt-5",
      "input": 2.50,
      "output": 10.00,
      "input_cached": null
    }
  ]
}
```

**Model ID Mapping**: Conduit uses standardized model IDs (e.g., "gpt-5" instead of "gpt-4o"). See `conduit.yaml` litellm.model_mappings for provider-to-conduit ID translation.

## Error Handling

### All Sources Fail

```python
RuntimeError: "Failed to load pricing from all sources (database, cache, direct fetch).
              Check network connectivity and try running: python scripts/sync_pricing.py"
```

**Resolution**:
1. Check network connectivity
2. Run `python scripts/sync_pricing.py` to populate database
3. Verify DATABASE_URL if using database mode
4. Check llm-prices.com availability

### Corrupted Cache File

If `~/.cache/conduit/pricing.json` is corrupted:
- Logs warning: "Failed to read cache file: {error}"
- Falls back to direct fetch
- Overwrites corrupted cache with fresh data

### Database Connection Failure

If database.get_latest_pricing() fails:
- Logs warning: "Database pricing load failed: {error}"
- Falls back to local cache
- Continues gracefully (no service interruption)

## Testing

Comprehensive unit tests cover all pricing scenarios (tests/unit/test_pricing_manager.py):

**Test Coverage**:
- Cache-only mode (3 tests)
- Database fallback (3 tests)
- In-memory caching with TTL (3 tests)
- Custom configuration (2 tests)
- Error handling (2 tests)

**Run Tests**:
```bash
pytest tests/unit/test_pricing_manager.py -v
```

## Migration Guide

### Upgrading to Historical Pricing

If you have an existing Conduit deployment, run the migration:

```bash
# Apply database migration
alembic upgrade head
```

This migration (34c6749675ef):
1. Adds `id` column as auto-incrementing primary key
2. Changes primary key from `model_id` to `id`
3. Adds UNIQUE constraint on `(model_id, snapshot_at)`
4. Adds index on `model_id` for fast lookups

**Backward Compatibility**: Existing single-snapshot-per-model data continues to work. New syncs will append historical snapshots.

### Removing Hardcoded Pricing

If your code has hardcoded pricing (e.g., MODEL_PRIORS dictionaries):

**Before**:
```python
MODEL_PRIORS = {
    "gpt-5": {
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
    }
}
```

**After**:
```python
from conduit.core.pricing_manager import PricingManager

pricing_manager = PricingManager(database=database)
pricing = await pricing_manager.get_pricing()
gpt5 = pricing["gpt-5"]
# Use gpt5.input_cost_per_million and gpt5.output_cost_per_million
```

## Best Practices

1. **Production**: Use database mode with daily `sync_pricing.py` cron job
2. **Development**: Use cache-only mode (no DATABASE_URL)
3. **CI/CD**: Use cache-only mode with committed pricing.json fixture
4. **Testing**: Mock `PricingManager` or use temp cache files (see test examples)
5. **Monitoring**: Log pricing staleness warnings to detect sync failures

## Troubleshooting

### "No pricing found in database"

**Cause**: Database is empty or sync failed.

**Resolution**:
```bash
python scripts/sync_pricing.py
```

### "Cache is X hours old (threshold: 24h). Attempting to refresh"

**Cause**: Local cache is stale, auto-refreshing from llm-prices.com.

**Resolution**: Normal behavior. If refresh fails, check network or increase `cache_ttl_hours`.

### "Database pricing is X days old (threshold: 7 days)"

**Cause**: Database pricing hasn't been synced in over 7 days.

**Resolution**:
```bash
python scripts/sync_pricing.py
```

Set up automated sync (cron job or CI schedule).

### "In-memory cache expired (Xh > 24h), reloading pricing"

**Cause**: Long-running process exceeded cache TTL.

**Resolution**: Normal behavior. Adjust `cache_ttl_hours` if needed.

## Related Documentation

- [Architecture](ARCHITECTURE.md): Overall Conduit architecture
- [Bandit Algorithms](BANDIT_ALGORITHMS.md): How pricing is used in routing decisions
- [Configuration](../conduit.yaml): Pricing configuration options
- [Database](../conduit/core/database.py): Database interface and pricing queries
