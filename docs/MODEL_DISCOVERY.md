# Model Discovery and Dynamic Pricing

**Version**: 0.0.2-alpha
**Date**: 2025-11-20
**Status**: Production-ready

---

## Overview

Conduit uses a **dynamic model discovery** system that automatically fetches pricing and availability from [llm-prices.com](https://www.llm-prices.com), eliminating manual maintenance and ensuring access to the latest models and pricing.

### Key Benefits

- **Always Current**: Pricing auto-updates from community-maintained database
- **71+ Models**: Support for all models across PydanticAI-supported providers
- **Auto-Detection**: Automatically discovers which models YOU can use based on API keys
- **Zero Maintenance**: No hard-coded pricing tables to update
- **Graceful Degradation**: Falls back to static pricing if API unavailable

---

## Architecture

### Data Sources

```
llm-prices.com API
    ↓ (fetch with 24h cache)
PydanticAI Provider Filter
    ↓ (only supported providers)
ModelArm Registry (71 models)
    ↓ (auto-detect API keys)
Available Models (your subset)
```

### Components

**pricing_fetcher.py**: Dynamic pricing with caching
- Fetches from https://www.llm-prices.com/current-v1.json
- 24-hour in-memory cache (TTL)
- Fallback to static pricing on failure
- Quality estimation heuristics (llm-prices doesn't provide quality)

**registry.py**: Model registry management
- `create_model_registry()` - builds from llm-prices.com data
- `supported_models()` - all models Conduit can use
- `available_models()` - models YOU can use (auto-detects from .env)
- `get_available_providers()` - which providers have API keys

---

## API Reference

### supported_models()

Get all models Conduit supports (from llm-prices.com).

```python
from conduit.models import supported_models

# All 71+ models
all_models = supported_models()

# High-quality budget models
good_cheap = supported_models(min_quality=0.85, max_cost=0.001)

# OpenAI only
openai_models = supported_models(providers=["openai"])

# Multiple filters
filtered = supported_models(
    providers=["openai", "anthropic"],
    min_quality=0.90,
    max_cost=0.005
)
```

**Parameters**:
- `providers` (list[str], optional): Filter to specific providers
- `min_quality` (float, optional): Minimum expected quality (0-1 scale)
- `max_cost` (float, optional): Maximum average cost per token

**Returns**: `list[ModelArm]` with model metadata

**Example Output**:
```python
[
    ModelArm(
        model_id="openai:gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        cost_per_input_token=0.00015,   # $0.15/1M tokens
        cost_per_output_token=0.0006,   # $0.60/1M tokens
        expected_quality=0.85,
        metadata={
            "pricing_source": "llm-prices.com",
            "pricing_updated": "2025-11-20T12:00:00Z",
            "display_name": "GPT-4o Mini"
        }
    ),
    # ... 70+ more models
]
```

---

### available_models()

Get models YOU can actually use (based on API keys in .env).

```python
from conduit.models import available_models

# What can I use?
my_models = available_models()

# What high-quality models can I use?
my_good_models = available_models(min_quality=0.90)

# Custom .env location
models = available_models(dotenv_path="/path/to/.env")
```

**Parameters**:
- `dotenv_path` (str, default=".env"): Path to .env file
- `providers` (list[str], optional): Further filter providers
- `min_quality` (float, optional): Minimum expected quality
- `max_cost` (float, optional): Maximum average cost per token

**Returns**: `list[ModelArm]` for models with configured API keys

**Auto-Detection Logic**:
1. Loads .env file from specified path
2. Checks for API keys in environment:
   - `OPENAI_API_KEY` → openai models
   - `ANTHROPIC_API_KEY` → anthropic models
   - `GOOGLE_API_KEY` → google models
   - `MISTRAL_API_KEY` → mistral models
   - `AWS_ACCESS_KEY_ID` → amazon/bedrock models
3. Returns only models from providers with keys

---

### get_available_providers()

Get list of providers you have API keys for.

```python
from conduit.models import get_available_providers

providers = get_available_providers()
print(providers)
# ['openai', 'anthropic', 'google']
```

**Parameters**:
- `dotenv_path` (str, default=".env"): Path to .env file

**Returns**: `list[str]` of provider names with configured keys

---

## Provider Support

### PydanticAI Supported Providers

Conduit supports all providers that PydanticAI can execute:

| Provider | API Key | Models | Pricing Source |
|----------|---------|--------|----------------|
| openai | `OPENAI_API_KEY` | 25+ | llm-prices.com |
| anthropic | `ANTHROPIC_API_KEY` | 9+ | llm-prices.com |
| google | `GOOGLE_API_KEY` | 19+ | llm-prices.com |
| mistral | `MISTRAL_API_KEY` | 14+ | llm-prices.com |
| amazon | `AWS_ACCESS_KEY_ID` | 4+ | llm-prices.com |
| cohere* | `COHERE_API_KEY` | 2+ | Fallback pricing |
| groq* | `GROQ_API_KEY` | 3+ | Fallback pricing |
| huggingface* | `HUGGINGFACE_API_KEY` | varies | Fallback pricing |

\* These providers use fallback pricing until llm-prices.com adds them

### Filtering Logic

**Included Models**:
- Provider is in `PYDANTIC_AI_PROVIDERS` (can execute)
- AND pricing data exists (llm-prices.com OR fallback)

**Excluded Models**:
- Providers not supported by PydanticAI: deepseek, xai, minimax, moonshot-ai
- Even if pricing exists, we skip them since PydanticAI can't execute

---

## Pricing Updates

### Dynamic Pricing Flow

```python
# First call (or after 24h cache expiry)
registry = create_model_registry()
# → Fetches from llm-prices.com
# → Caches for 24 hours
# → Returns 71 models

# Subsequent calls (within 24h)
registry = create_model_registry()
# → Returns cached data (instant)
# → No API call

# Force refresh
registry = create_model_registry(use_cache=False)
# → Fetches fresh data
# → Updates cache
```

### Cache Configuration

```python
# In conduit/models/pricing_fetcher.py
CACHE_TTL_HOURS = 24
_pricing_cache: dict[str, Any] | None = None
_cache_expires_at: datetime | None = None
```

**Cache Expiry**: 24 hours from fetch
**Cache Storage**: In-memory (not Redis) for simplicity
**Cache Scope**: Per-process (each worker has own cache)

### Fallback Pricing

If llm-prices.com is unreachable:

```python
# Fallback pricing (minimal set)
FALLBACK_PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00, "quality": 0.95},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "quality": 0.85},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "quality": 0.96},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "quality": 0.97},
    },
    # Also includes cohere, groq (not in llm-prices.com)
}
```

---

## Quality Estimation

llm-prices.com provides pricing but NOT quality scores. Conduit estimates quality using heuristics:

```python
QUALITY_ESTIMATES = {
    # OpenAI
    "gpt-4o": 0.95,
    "gpt-4o-mini": 0.85,
    "gpt-4-turbo": 0.93,

    # Anthropic
    "claude-3.7": 0.97,
    "claude-3.5": 0.96,
    "claude-3-opus": 0.97,
    "claude-3-sonnet": 0.94,
    "claude-3-haiku": 0.80,

    # Google
    "gemini-2.0": 0.93,
    "gemini-1.5-pro": 0.92,
    "gemini-1.5-flash": 0.82,

    # ... 20+ model families

    "default": 0.75,  # Fallback for unknown models
}

def estimate_quality(model_id: str, model_name: str) -> float:
    """Match model ID/name against quality patterns."""
    for pattern, quality in QUALITY_ESTIMATES.items():
        if pattern in model_id.lower() or pattern in model_name.lower():
            return quality
    return QUALITY_ESTIMATES["default"]
```

**Quality Scale**:
- 0.95-1.0: Flagship models (GPT-4o, Claude Opus, Gemini Pro)
- 0.85-0.95: High-quality (GPT-4o-mini, Claude Sonnet)
- 0.75-0.85: Mid-tier (Claude Haiku, Gemini Flash)
- 0.60-0.75: Budget models

---

## Usage Examples

### Example 1: Model Discovery

```python
from conduit.models import supported_models, available_models, get_available_providers

# See what Conduit supports
all_models = supported_models()
print(f"Conduit supports {len(all_models)} models")

# See what you can use
providers = get_available_providers()
print(f"Your providers: {', '.join(providers)}")

my_models = available_models()
print(f"You can use {len(my_models)} models")

# Find best value models
budget_models = available_models(min_quality=0.85, max_cost=0.001)
for model in budget_models:
    avg_cost = (model.cost_per_input_token + model.cost_per_output_token) / 2
    print(f"{model.model_id}: {model.expected_quality:.0%} quality, ${avg_cost*1000:.4f}/1K")
```

Output:
```
Conduit supports 71 models
Your providers: openai, anthropic, google
You can use 34 models
openai:gpt-4o-mini: 85% quality, $0.3750/1K
anthropic:claude-3-haiku-20240307: 80% quality, $0.3125/1K
google:gemini-1.5-flash: 82% quality, $0.1000/1K
```

### Example 2: Router with Auto-Detection

```python
from conduit.engines.router import Router
from conduit.models import available_models

# Router automatically uses all your available models
router = Router()

# Or explicitly filter
my_good_models = available_models(min_quality=0.90)
router = Router(models=[m.model_id for m in my_good_models])
```

### Example 3: Provider-Specific Filtering

```python
from conduit.models import supported_models

# Compare providers
openai = supported_models(providers=["openai"])
anthropic = supported_models(providers=["anthropic"])

print(f"OpenAI: {len(openai)} models")
print(f"Anthropic: {len(anthropic)} models")

# Find cheapest high-quality model
cheap_good = supported_models(min_quality=0.90, max_cost=0.002)
if cheap_good:
    best = min(cheap_good, key=lambda m: m.cost_per_input_token)
    print(f"Best value: {best.model_id}")
```

---

## Configuration

### Environment Variables

Add API keys to `.env` to enable providers:

```bash
# OpenAI (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)
OPENAI_API_KEY=sk-...

# Anthropic (claude-3-5-sonnet, claude-3-opus, claude-3-haiku)
ANTHROPIC_API_KEY=sk-ant-...

# Google/Gemini (gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro)
GOOGLE_API_KEY=AIza...

# Groq (llama-3.1-70b, llama-3.1-8b, mixtral-8x7b)
GROQ_API_KEY=gsk_...

# Mistral (mistral-large, mistral-medium, mistral-small)
MISTRAL_API_KEY=...

# Cohere (command-r-plus, command-r)
COHERE_API_KEY=...

# AWS Bedrock (claude, llama, titan models via AWS)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# HuggingFace (hosted inference API)
HUGGINGFACE_API_KEY=hf_...
```

### Provider Detection Mapping

```python
# In conduit/models/registry.py
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "amazon": "AWS_ACCESS_KEY_ID",
    # Commented out until llm-prices.com adds them:
    # "groq": "GROQ_API_KEY",
    # "cohere": "COHERE_API_KEY",
    # "huggingface": "HUGGINGFACE_API_KEY",
}
```

**Note**: Cohere, Groq, and HuggingFace use fallback pricing until llm-prices.com adds them. To enable:

1. Add API key to `.env`
2. Models will use estimated pricing from `FALLBACK_PRICING`
3. When llm-prices.com adds them, uncomment in `PROVIDER_ENV_VARS`

---

## Implementation Details

### Model Conversion

llm-prices.com provides pricing per-million tokens. Conduit converts to per-1K:

```python
# From llm-prices.com
{
    "id": "gpt-4o-mini",
    "vendor": "openai",
    "name": "GPT-4o Mini",
    "input": 150.0,    # $150 per 1M tokens
    "output": 600.0    # $600 per 1M tokens
}

# Converted to ModelArm
ModelArm(
    model_id="openai:gpt-4o-mini",
    cost_per_input_token=0.00015,   # $0.15 per 1K tokens
    cost_per_output_token=0.0006,   # $0.60 per 1K tokens
    expected_quality=0.85,
    metadata={
        "pricing_source": "llm-prices.com",
        "quality_estimate_source": "conduit_heuristics"
    }
)
```

### Registry Creation

```python
def create_model_registry(use_cache: bool = True) -> list[ModelArm]:
    """Create comprehensive model registry from llm-prices.com."""

    # Fetch pricing (with 24h cache)
    try:
        pricing_data = fetch_pricing_sync() if use_cache else get_fallback_pricing()
    except Exception as e:
        logger.warning(f"Failed to fetch pricing: {e}")
        pricing_data = get_fallback_pricing()

    # Convert to ModelArm instances
    models = []
    skipped = 0
    for model_data in pricing_data["prices"]:
        provider = model_data["vendor"]

        # Skip providers not supported by PydanticAI
        if provider not in PYDANTIC_AI_PROVIDERS:
            skipped += 1
            continue

        # Convert pricing
        input_cost = float(model_data["input"]) / 1000
        output_cost = float(model_data["output"]) / 1000

        # Estimate quality
        quality = estimate_quality(model_data["id"], model_data["name"])

        arm = ModelArm(
            model_id=f"{provider}:{model_data['id']}",
            provider=provider,
            model_name=model_data["id"],
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            expected_quality=quality,
            metadata={
                "pricing_source": "llm-prices.com",
                "pricing_updated": pricing_data["updated_at"],
                "quality_estimate_source": "conduit_heuristics",
                "display_name": model_data["name"]
            }
        )
        models.append(arm)

    logger.info(f"Created registry with {len(models)} models (skipped {skipped})")
    return models
```

---

## Troubleshooting

### No Models Available

**Problem**: `available_models()` returns empty list

**Solutions**:
1. Check `.env` file exists and has API keys
2. Verify API key format (e.g., `sk-...` for OpenAI)
3. Check environment variable names match `PROVIDER_ENV_VARS`
4. Use `get_available_providers()` to debug detection

```python
from conduit.models import get_available_providers

providers = get_available_providers()
print(providers)  # Should show ['openai', 'anthropic', ...]

# If empty, check .env loading
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # Should show key
```

### Pricing Fetch Failed

**Problem**: Logs show "Failed to fetch pricing from llm-prices.com"

**Impact**: Uses fallback pricing (limited models)

**Solutions**:
1. Check internet connectivity
2. Verify llm-prices.com is accessible
3. Wait for cache expiry if transient failure
4. Fallback pricing still works (4-10 models per provider)

### Quality Estimates Inaccurate

**Problem**: Quality scores don't match real-world performance

**Solutions**:
1. Update `QUALITY_ESTIMATES` in `pricing_fetcher.py`
2. Use explicit feedback to let system learn true quality
3. Quality estimates are just priors - Thompson Sampling learns over time

---

## Future Enhancements

### Planned Improvements

1. **Quality from Benchmarks**: Fetch LMSYS Arena ratings for real quality scores
2. **Regional Pricing**: Support region-specific pricing variations
3. **Custom Pricing**: Override llm-prices.com with enterprise pricing
4. **Multi-Region**: Auto-detect best region for latency
5. **Provider Health**: Track provider availability and route around outages

### When llm-prices.com Adds Missing Providers

When llm-prices.com adds cohere/groq/huggingface:

1. Uncomment providers in `PROVIDER_ENV_VARS`
2. Remove from `FALLBACK_PRICING`
3. Models will use live pricing automatically

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [README.md](../README.md) - Quick start and examples
- [examples/01_quickstart/model_discovery.py](../examples/01_quickstart/model_discovery.py) - Working example

---

**Last Updated**: 2025-11-20
**Maintainer**: Conduit Team
**Source**: https://github.com/MisfitIdeas/conduit
