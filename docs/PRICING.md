# Pricing Architecture

**Last Updated**: 2025-12-05

Conduit uses LiteLLM's bundled `model_cost` database for automatic, accurate model pricing with zero configuration.

## Overview

Pricing is handled automatically via LiteLLM's built-in pricing database, which includes:

- **500+ models** from all major providers (OpenAI, Anthropic, Google, etc.)
- **Exact model ID matching** (no normalization needed)
- **Cache pricing** (both creation and read costs)
- **Tiered pricing** (e.g., >200k token rates for Claude)
- **Provider metadata** (capabilities, limits, etc.)

Update pricing by running `uv update litellm`.

## Key Benefits

| Before (llm-prices.com) | After (LiteLLM) |
|-------------------------|-----------------|
| External API calls | Bundled with package |
| Needs model ID normalization | Direct model ID match |
| Cache file management | No file management |
| Database migrations | No database needed |
| Stale pricing concerns | Updates with pip |

## Usage

### Get Pricing for a Model

```python
from conduit.core.pricing import get_model_pricing

# Get pricing for a specific model
pricing = get_model_pricing("claude-sonnet-4-5-20250929")
if pricing:
    print(f"Input: ${pricing.input_cost_per_million}/1M tokens")
    print(f"Output: ${pricing.output_cost_per_million}/1M tokens")

    # Cache pricing (if supported)
    if pricing.cached_input_cost_per_million:
        print(f"Cache read: ${pricing.cached_input_cost_per_million}/1M tokens")
```

### Compute Cost for a Request

```python
from conduit.core.pricing import compute_cost

# Basic cost calculation
cost = compute_cost(
    input_tokens=1000,
    output_tokens=500,
    model_id="claude-sonnet-4-5-20250929",
)
print(f"Cost: ${cost:.6f}")

# With cache tokens
cost = compute_cost(
    input_tokens=500,
    output_tokens=500,
    model_id="claude-sonnet-4-5-20250929",
    cache_read_tokens=500,  # Tokens read from cache (cheaper)
    cache_creation_tokens=0,  # Tokens written to cache
)
```

### Get All Model Pricing

```python
from conduit.core.pricing import get_all_model_pricing

# Get pricing for all chat models
all_pricing = get_all_model_pricing()
print(f"Loaded pricing for {len(all_pricing)} models")

# Find cheapest model
cheapest = min(all_pricing.values(), key=lambda p: p.input_cost_per_million)
print(f"Cheapest: {cheapest.model_id} at ${cheapest.input_cost_per_million}/1M")
```

## ModelPricing Structure

```python
class ModelPricing(BaseModel):
    model_id: str                              # e.g., "claude-sonnet-4-5-20250929"
    input_cost_per_million: float              # e.g., 3.0
    output_cost_per_million: float             # e.g., 15.0
    cached_input_cost_per_million: float | None  # Cache read cost
    cache_creation_cost_per_million: float | None  # Cache write cost
    source: str | None                         # "litellm"
    snapshot_at: datetime | None               # When pricing was loaded

    # Computed properties (per-token costs)
    input_cost_per_token: float      # input_cost_per_million / 1,000,000
    output_cost_per_token: float     # output_cost_per_million / 1,000,000
    cached_input_cost_per_token: float | None
```

## Integration with ModelExecutor

The `ModelExecutor` automatically uses LiteLLM pricing for cost calculation:

```python
from conduit.engines.executor import ModelExecutor

executor = ModelExecutor()  # No pricing argument needed

# Cost is automatically calculated from LiteLLM's database
response = await executor.execute(
    model="claude-sonnet-4-5-20250929",
    prompt="Hello, world!",
    result_type=MyOutput,
    query_id="test-123",
)
print(f"Cost: ${response.cost:.6f}")
```

## Supported Models

LiteLLM includes pricing for all major providers:

- **OpenAI**: gpt-4o, gpt-4o-mini, o1, o1-mini, etc.
- **Anthropic**: claude-3.5-sonnet, claude-3-haiku, claude-opus-4-5, etc.
- **Google**: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, etc.
- **AWS Bedrock**: All Claude and other Bedrock models
- **Azure OpenAI**: All Azure-hosted models
- **Vertex AI**: Google models on Vertex
- **And many more**: Mistral, Cohere, Groq, Together, etc.

### Check Model Availability

```python
from conduit.core.pricing import get_model_pricing

# Check if a model has pricing
if get_model_pricing("my-custom-model") is None:
    print("No pricing found - cost will be $0")
```

## Updating Pricing

Pricing updates when you update LiteLLM:

```bash
# Update LiteLLM to get latest pricing
uv update litellm

# Or with pip
pip install --upgrade litellm
```

LiteLLM maintains pricing in their [model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) file, which is updated frequently.

## Historical Pricing Snapshots

For tracking pricing changes over time, you can snapshot LiteLLM pricing to a database:

```bash
# Preview what would be synced
python scripts/sync_pricing.py --dry-run

# Sync to database (requires SUPABASE_URL and SUPABASE_ANON_KEY)
python scripts/sync_pricing.py

# Force sync even if today's snapshot exists
python scripts/sync_pricing.py --force
```

The script:
- Extracts pricing from LiteLLM's bundled `model_cost` (no external API calls)
- Stores timestamped snapshots in the `model_prices` table
- Prevents duplicate snapshots for the same day (use `--force` to override)
- Works with any PostgreSQL database (Supabase, RDS, self-hosted)

**Use cases for historical pricing:**
- Track pricing changes over time
- Understand routing decisions with historical context
- Audit cost calculations retroactively
- Compare model economics across time periods

**Database Schema:**
```sql
CREATE TABLE model_prices (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    input_cost_per_million NUMERIC NOT NULL,
    output_cost_per_million NUMERIC NOT NULL,
    cached_input_cost_per_million NUMERIC,
    source TEXT,
    snapshot_at TIMESTAMPTZ,
    UNIQUE(model_id, snapshot_at)
);
```

## Error Handling

If a model has no pricing in LiteLLM's database:

```python
from conduit.core.pricing import get_model_pricing, compute_cost

# Returns None for unknown models
pricing = get_model_pricing("unknown-model-xyz")
assert pricing is None

# compute_cost returns 0 and logs a warning
cost = compute_cost(1000, 500, "unknown-model-xyz")
assert cost == 0.0  # Logged: "No pricing for unknown-model-xyz, returning 0 cost"
```

## Testing

```python
import pytest
from conduit.core.pricing import get_model_pricing, compute_cost

def test_get_known_model_pricing():
    """Test pricing retrieval for a known model."""
    pricing = get_model_pricing("gpt-4o-mini")
    assert pricing is not None
    assert pricing.input_cost_per_million > 0
    assert pricing.output_cost_per_million > 0

def test_compute_cost_with_cache():
    """Test cost computation with cache tokens."""
    cost = compute_cost(
        input_tokens=1000,
        output_tokens=500,
        model_id="claude-sonnet-4-5-20250929",
        cache_read_tokens=200,
    )
    assert cost > 0
```

## Related Documentation

- [Architecture](ARCHITECTURE.md): Overall Conduit architecture
- [Bandit Algorithms](BANDIT_ALGORITHMS.md): How pricing affects routing decisions
- [LiteLLM Documentation](https://docs.litellm.ai/): LiteLLM's model cost database
