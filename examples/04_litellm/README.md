# LiteLLM Integration Examples

Conduit integrates seamlessly with [LiteLLM](https://github.com/BerriAI/litellm) to provide ML-powered routing across 100+ LLM providers with automatic cost optimization.

## Overview

These examples demonstrate how to use Conduit with LiteLLM's `Router` to enable:
- Automatic model selection based on query context
- Cost optimization (30-50% reduction) while maintaining quality
- Learning from every request without manual feedback
- Optional LLM-as-judge quality measurement with Arbiter

## Setup

### Installation

```bash
# Install Conduit with LiteLLM support
pip install conduit[litellm]

# Or using uv
uv pip install conduit[litellm]
```

### API Keys

Set environment variables for the providers you want to use:

```bash
# Required: At least one LLM provider
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Additional providers for multi-provider routing
export GOOGLE_API_KEY="..."  # or GEMINI_API_KEY
export GROQ_API_KEY="gsk-..."

# Optional: Redis for caching (examples work without it)
export REDIS_URL="redis://localhost:6379"
```

## Examples

### 1. basic_usage.py - Getting Started

**Purpose**: Simplest possible setup showing 3-step integration.

**What it demonstrates**:
- Creating LiteLLM Router with multiple models
- Enabling Conduit ML routing strategy
- Automatic feedback capture (no manual work required)

**When to use this pattern**:
- First time integrating Conduit
- Simple 2-3 model routing
- Default configuration is sufficient

**Run it**:
```bash
uv run python examples/04_litellm/basic_usage.py
```

**Expected output**:
```
================================================================================
Conduit + LiteLLM: Basic Usage Demo
================================================================================

Step 1: Creating LiteLLM router with 2 models...
Step 2: Enabling Conduit ML routing strategy...
Step 3: Making requests (Conduit learns from each one)...

Query 1/3: What is 2+2?...
  Model: gpt-4o-mini
  Cost: $0.000150
  Response: 2+2 equals 4...

✅ Conduit automatically learns from every request
✅ No manual feedback required (cost/latency tracked automatically)
✅ Quality estimated from response content
✅ Bandits improve routing over time (exploration → exploitation)
```

### 2. custom_config.py - Advanced Configuration

**Purpose**: Shows how to customize Conduit's behavior with advanced settings.

**What it demonstrates**:
- Two configuration approaches (auto-creation vs pre-configured Router)
- Redis caching integration (optional)
- Custom embedding models for domain-specific routing
- Cache statistics and monitoring

**When to use this pattern**:
- Production deployments with caching
- Domain-specific routing (e.g., medical, legal, technical)
- Need fine control over Router configuration

**Run it**:
```bash
# Without Redis (works fine, just no caching)
uv run python examples/04_litellm/custom_config.py

# With Redis (enable caching)
export REDIS_URL="redis://localhost:6379"
uv run python examples/04_litellm/custom_config.py
```

**Expected output**:
```
================================================================================
Conduit + LiteLLM: Custom Configuration Demo
================================================================================

=== Option 1: Configure via conduit_config ===
Passing configuration to auto-created Router...
  Cache enabled: False
  Embedding model: all-MiniLM-L6-v2 (384 dimensions)

=== Option 2: Pre-configured Conduit Router ===
Creating Conduit Router with explicit settings...
  Using pre-configured Router
  Models: 2 arms
  Feature dimensions: 384

Making test queries...
Query 1/3: What is machine learning?
  → Model: gpt-4o-mini
  → Cost: $0.000200

Configuration Options Summary:
Cache Settings:
  cache_enabled: Enable Redis caching (default: False)
  redis_url: Redis connection URL

Embedding Models:
  all-MiniLM-L6-v2: Fast, 384 dims (default)
  all-mpnet-base-v2: Better quality, 768 dims
  paraphrase-multilingual: Multi-language, 384 dims
```

### 3. multi_provider.py - Multi-Provider Routing

**Purpose**: Demonstrates Conduit's strength: learning optimal routing across many providers.

**What it demonstrates**:
- Dynamic model list based on available API keys
- Routing across OpenAI, Anthropic, Google, Groq (5+ models)
- Learning which provider is best for different query types
- Cost/quality/latency trade-offs per provider

**When to use this pattern**:
- Want cost optimization across multiple providers
- Different providers excel at different tasks
- Need fallback/redundancy across providers
- High query volume justifies multi-provider setup

**Run it**:
```bash
# Requires at least 2 providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# Optional: Add more providers for better routing
export GROQ_API_KEY="gsk-..."
export GOOGLE_API_KEY="..."

uv run python examples/04_litellm/multi_provider.py
```

**Expected output**:
```
================================================================================
Conduit Multi-Provider Routing Demo
================================================================================
Available providers: OpenAI, Anthropic, Groq, Google

Configured 5 models across 4 providers

Enabling Conduit ML routing...

Testing with 8 diverse queries...

Query 1/8: What is 2+2?...
  → Provider: Groq
  → Model: groq/llama-3.1-8b-instant
  → Cost: $0.000050
  → Response: 2+2 equals 4...

Query 2/8: Write a Python function to sort a list...
  → Provider: OpenAI
  → Model: gpt-4o-mini
  → Cost: $0.000180
  → Response: def sort_list(items): return sorted(items)...

================================================================================
Routing Distribution:
================================================================================
  Anthropic: 1/8 queries (12.5%)
  Google: 1/8 queries (12.5%)
  Groq: 4/8 queries (50.0%)
  OpenAI: 2/8 queries (25.0%)

  Total cost: $0.000680

================================================================================
Key Insights:
================================================================================
✅ Conduit routes across multiple providers automatically
✅ Learns cost/quality/latency trade-offs per provider
✅ Adapts routing based on query context
✅ No manual configuration needed - ML does the work

With more queries, Conduit will:
  - Identify which providers excel at code vs creative tasks
  - Learn speed/cost trade-offs per provider
  - Optimize routing for your specific workload
```

### 4. arbiter_quality_measurement.py - LLM-as-Judge Evaluation

**Purpose**: Shows how to enable Arbiter LLM-as-judge for quality assessment.

**What it demonstrates**:
- ArbiterEvaluator integration for semantic + factuality scoring
- Sampling strategy (10% evaluation rate) to control costs
- Budget controls ($10/day limit)
- Fire-and-forget async evaluation (non-blocking)
- Storing evaluation feedback for bandit learning

**When to use this pattern**:
- Need objective quality measurement beyond implicit signals
- Want to track quality metrics over time
- Building quality assurance pipeline
- Validating routing decisions

**Run it**:
```bash
export OPENAI_API_KEY="sk-..."  # For LLM routing
# Arbiter will use same key for evaluation

uv run python examples/04_litellm/arbiter_quality_measurement.py
```

**Expected output**:
```
================================================================================
Conduit + LiteLLM: Arbiter Quality Measurement Demo
================================================================================

Setting up database...
Creating router with Arbiter evaluator...
  Sample rate: 10% (evaluate 1 in 10 queries)
  Daily budget: $10.00
  Evaluation model: gpt-4o-mini

Testing with 3 queries...

Query 1/3: What is 2+2?
  Model used: gpt-4o-mini
  Response: 2+2 equals 4...
  Cost: $0.000150
  ✓ Arbiter evaluation queued (background)

Query 2/3: Explain quantum computing
  Model used: gpt-4o-mini
  Response: Quantum computing uses quantum mechanics...
  Cost: $0.000280
  ○ Skipped evaluation (sampling: 10%)

Query 3/3: Write a Python function
  Model used: claude-3-haiku-20240307
  Response: def example(): pass...
  Cost: $0.000120
  ✓ Arbiter evaluation queued (background)

Waiting for evaluations to complete...

================================================================================
Evaluation Results:
================================================================================
Query: "What is 2+2?"
  Quality scores: semantic=0.95, factuality=1.00
  Evaluation cost: $0.000080

Query: "Write a Python function"
  Quality scores: semantic=0.88, factuality=0.92
  Evaluation cost: $0.000075

Total evaluation cost: $0.000155
Total routing cost: $0.000550

✅ Fire-and-forget evaluation (non-blocking)
✅ Configurable sampling rate (10% = low cost)
✅ Automatic feedback storage for bandit learning
✅ Budget controls prevent overspending
```

## Integration Patterns

### Minimal Setup (2 models, no config)

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

router = Router(model_list=[...])
strategy = ConduitRoutingStrategy()
ConduitRoutingStrategy.setup_strategy(router, strategy)
```

### With Arbiter Evaluation

```python
from conduit.evaluation import ArbiterEvaluator
from conduit_litellm import ConduitRoutingStrategy

evaluator = ArbiterEvaluator(db=database, sample_rate=0.1)
strategy = ConduitRoutingStrategy(evaluator=evaluator)
ConduitRoutingStrategy.setup_strategy(router, strategy)
```

### With Custom Configuration

```python
from conduit.engines.router import Router as ConduitRouter
from conduit_litellm import ConduitRoutingStrategy

# Pre-configure Conduit router
conduit_router = ConduitRouter(
    models=["gpt-4o-mini", "claude-3-haiku"],
    embedding_model="all-MiniLM-L6-v2",
    cache_enabled=True,
)

strategy = ConduitRoutingStrategy(conduit_router=conduit_router)
ConduitRoutingStrategy.setup_strategy(litellm_router, strategy)
```

## Common Issues

### Issue: "Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
**Solution**: Export at least one LLM provider API key before running examples.

### Issue: "Need at least 2 providers" (multi_provider.py)
**Solution**: Export API keys for at least 2 different providers (OpenAI + Anthropic recommended).

### Issue: Redis connection errors
**Solution**: Examples work without Redis (caching disabled). To enable caching, start Redis locally:
```bash
redis-server
export REDIS_URL="redis://localhost:6379"
```

### Issue: Import errors "No module named 'conduit_litellm'"
**Solution**: Install with LiteLLM support: `pip install conduit[litellm]`

## Next Steps

1. **Start simple**: Run `basic_usage.py` to verify setup
2. **Add caching**: Set up Redis and run `custom_config.py`
3. **Multi-provider**: Configure multiple API keys and run `multi_provider.py`
4. **Quality measurement**: Add Arbiter evaluation with `arbiter_quality_measurement.py`

## Related Documentation

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Conduit Core Documentation](../../README.md)
- [Bandit Algorithms](../../docs/bandits.md)
- [LiteLLM Plugin README](../../conduit_litellm/README.md)
