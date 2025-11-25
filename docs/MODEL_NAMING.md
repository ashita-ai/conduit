# Model Naming Conventions

This document explains how model identifiers work in Conduit and how to ensure proper pricing lookups.

## Overview

Conduit uses model IDs in three contexts:
1. **Configuration** (config.py): Default models for routing
2. **PydanticAI Execution**: Model specification for LLM calls
3. **Pricing Database**: Cost calculation lookups

All three must use consistent model identifiers for proper operation.

## Model ID Format

Model IDs follow provider-specific conventions:

### OpenAI Models
- Format: `{model-name}` or `{model-name}-{version}`
- Examples:
  - `gpt-4o-mini` - GPT-4o Mini (current)
  - `gpt-4o` - GPT-4o (flagship)
  - `gpt-4o-2024-11-20` - Specific version snapshot
  - `o1-preview` - O1 preview model
  - `o3-mini` - O3 Mini model

### Anthropic Models
- Format: `claude-{version}-{variant}` or `claude-{variant}-{version}`
- Examples:
  - `claude-3.5-sonnet` - Claude 3.5 Sonnet (current popular)
  - `claude-opus-4` - Claude Opus 4 (premium)
  - `claude-sonnet-4.5` - Claude Sonnet 4.5 (newer)
  - `claude-3-haiku` - Claude 3 Haiku (fast/cheap)

### Google Models
- Format: `gemini-{version}-{variant}`
- Examples:
  - `gemini-2.0-flash-exp` - Gemini 2.0 Flash Experimental
  - `gemini-1.5-pro` - Gemini 1.5 Pro
  - `gemini-1.5-flash` - Gemini 1.5 Flash

### Other Providers
- **xAI**: `grok-{version}` (e.g., `grok-2-vision-1212`)
- **Amazon**: `amazon-nova-{size}` (e.g., `amazon-nova-pro`)
- **Mistral**: `mistral-{variant}` (e.g., `mistral-large-2411`)

## Configuration Best Practices

### Default Models Selection

Choose models that:
1. ✅ Exist in the pricing database (verify with `python scripts/sync_pricing.py --dry-run`)
2. ✅ Are current/popular versions (not deprecated)
3. ✅ Span different cost/quality tiers
4. ✅ Work with your available API keys

**Current Recommended Defaults** (config.py):
```python
default_models: list[str] = Field(
    default=[
        "gpt-4o-mini",      # OpenAI - cheap, fast, good quality
        "gpt-4o",           # OpenAI - flagship, balanced
        "claude-3.5-sonnet", # Anthropic - current popular
        "claude-opus-4",     # Anthropic - premium quality
    ],
    description="Available models for routing",
)
```

### Verifying Model IDs

Before adding a model to config.py, verify it exists in pricing:

```python
from conduit.core.database import Database

db = Database()
await db.connect()

pricing = await db.get_model_prices()
model_id = "claude-3.5-sonnet"

if model_id in pricing:
    print(f"✓ {model_id} has pricing data")
    print(f"  Input: ${pricing[model_id].input_cost_per_million}/M")
    print(f"  Output: ${pricing[model_id].output_cost_per_million}/M")
else:
    print(f"✗ {model_id} NOT in database - will use fallback pricing")
```

## PydanticAI Model Specification

PydanticAI supports multiple model ID formats:

### Basic Format
```python
Agent(model="gpt-4o-mini")  # Uses model ID directly
```

### Provider Prefixed Format
```python
Agent(model="openai:gpt-4o-mini")  # Explicit provider
Agent(model="anthropic:claude-3.5-sonnet")  # Anthropic with prefix
```

Conduit uses the basic format (no provider prefix) to keep model IDs consistent with the pricing database.

## Pricing Database Lookups

The pricing database uses exact model ID matching:

1. **Exact Match**: `gpt-4o-mini` → Database has `gpt-4o-mini` ✓
2. **No Match**: `claude-sonnet-4` → Database has `claude-3.5-sonnet` ✗ (uses fallback)
3. **Versioned Match**: `gpt-4o-2024-11-20` → Database has exact version ✓

### Fallback Pricing

If a model ID isn't found in the database, the executor uses hardcoded fallback pricing (see `conduit/engines/executor.py`). This is less accurate and may be outdated.

**To avoid fallbacks:**
- Keep config.py models aligned with database
- Run `python scripts/sync_pricing.py` weekly to update pricing
- Verify new models exist before adding to config

## Adding Custom Models

### Option 1: Use Pricing Sync (Recommended)

If the model is in llm-prices.com:
```bash
python scripts/sync_pricing.py  # Fetches latest pricing including new models
```

### Option 2: Manual Database Insert

If the model isn't in llm-prices.com:
```sql
INSERT INTO model_prices (
    model_id,
    input_cost_per_million,
    output_cost_per_million,
    cached_input_cost_per_million,
    source,
    snapshot_at
) VALUES (
    'custom-model-id',
    1.50,  -- Input cost per 1M tokens
    6.00,  -- Output cost per 1M tokens
    0.75,  -- Cached input cost (optional)
    'manual',
    NOW()
) ON CONFLICT (model_id) DO UPDATE SET
    input_cost_per_million = EXCLUDED.input_cost_per_million,
    output_cost_per_million = EXCLUDED.output_cost_per_million,
    cached_input_cost_per_million = EXCLUDED.cached_input_cost_per_million,
    snapshot_at = EXCLUDED.snapshot_at;
```

### Option 3: Update Fallback Pricing

For development/testing without database:
```python
# In conduit/engines/executor.py
fallback_pricing = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "your-custom-model": {"input": 2.00, "output": 8.00},  # Add here
}
```

## Troubleshooting

### Issue: "Using fallback pricing" warnings

**Cause**: Model ID in config doesn't match database

**Fix**:
1. Check database models: `SELECT model_id FROM model_prices;`
2. Update config.py to use exact model ID from database
3. OR sync pricing: `python scripts/sync_pricing.py`

### Issue: Cost calculations seem wrong

**Cause**: Using fallback pricing instead of database pricing

**Fix**:
1. Verify model ID exists in database (see verification code above)
2. Check logs for "Using fallback pricing" warnings
3. Update config or add model to database

### Issue: Model not found errors

**Cause**: Model ID doesn't exist in provider's API

**Fix**:
1. Check provider documentation for correct model ID
2. Verify API key has access to that model
3. Use a different model from the same provider

## Model Versioning Strategy

Providers use different versioning strategies:

### Pinned Versions (Recommended for Production)
- **Pros**: Predictable behavior, stable pricing, reproducible results
- **Cons**: Must manually update to newer versions
- **Example**: `gpt-4o-2024-11-20`, `claude-opus-4-20250514`

### Latest Versions (Good for Development)
- **Pros**: Automatically get improvements, latest features
- **Cons**: Behavior may change unexpectedly, pricing may shift
- **Example**: `gpt-4o`, `claude-3.5-sonnet`

### Conduit Recommendation
- **Development**: Use latest versions (`gpt-4o`, `claude-3.5-sonnet`)
- **Production**: Consider pinning to specific versions after testing
- **Pricing**: Sync weekly to capture version-specific pricing changes

## Related Documentation

- Pricing updates: [docs/PRICING_UPDATES.md](./PRICING_UPDATES.md)
- Architecture: [docs/ARCHITECTURE.md](./ARCHITECTURE.md)
- Configuration: [conduit/core/config.py](../conduit/core/config.py)
