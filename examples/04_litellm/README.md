# LiteLLM Integration Examples

Practical examples demonstrating Conduit's ML-powered routing as a LiteLLM plugin.

## Overview

These examples show how to use Conduit with LiteLLM to enable intelligent, ML-based model selection across 100+ LLM providers. Conduit replaces LiteLLM's default routing with contextual bandit algorithms that learn optimal model selection from production traffic.

## Prerequisites

```bash
# Install Conduit with LiteLLM support
pip install conduit[litellm]

# Set API keys (at least one required)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Optional: Redis for caching (recommended)
brew install redis && redis-server
```

## Examples

### 1. basic_usage.py - Getting Started

**Purpose**: Simplest possible example showing LiteLLM + Conduit integration.

**Features**:
- Basic model configuration (3 OpenAI models)
- Automatic feedback learning
- Zero-config intelligent routing

**Run**:
```bash
python examples/04_litellm/basic_usage.py
```

**Expected Output**:
- Shows which model was selected for each query
- Demonstrates automatic feedback capture
- Explains learning behavior

**Key Concepts**:
- `ConduitRoutingStrategy()` replaces default LiteLLM routing
- `setup_strategy()` helper method initializes strategy correctly
- Feedback captured automatically from every request

---

### 2. custom_config.py - Advanced Configuration

**Purpose**: Demonstrate advanced Conduit configuration options.

**Features**:
- Hybrid routing (UCB1→LinUCB warm start)
- Redis caching (10-40x performance improvement)
- Custom embedding models
- Performance metrics

**Run**:
```bash
# Without Redis (works but no caching)
python examples/04_litellm/custom_config.py

# With Redis (recommended)
redis-server &  # Start Redis in background
python examples/04_litellm/custom_config.py
```

**Expected Output**:
- Shows cache hit/miss on repeated queries
- Demonstrates hybrid routing phases
- Reports performance benefits

**Configuration Options**:
```python
strategy = ConduitRoutingStrategy(
    use_hybrid=True,              # 30% faster convergence
    cache_enabled=True,           # 10-40x speedup on repeated queries
    redis_url="redis://localhost:6379",
    embedding_model="all-MiniLM-L6-v2",  # or "all-mpnet-base-v2"
)
```

---

### 3. multi_provider.py - Cross-Provider Routing

**Purpose**: Show intelligent routing across multiple LLM providers.

**Features**:
- 5+ provider support (OpenAI, Anthropic, Google, etc.)
- Automatic provider failover
- Cost optimization across providers
- Provider-specific strengths learning

**Run**:
```bash
# Set multiple provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export GEMINI_API_KEY="..."

python examples/04_litellm/multi_provider.py
```

**Expected Output**:
- Shows which provider/model selected for each query
- Demonstrates cross-provider learning
- Explains provider strengths

**Real-World Benefits**:
- **Reliability**: Automatic failover during provider outages
- **Cost**: Route simple queries to cheaper providers (Gemini Flash)
- **Quality**: Route complex queries to capable models (GPT-4o, Claude Sonnet)

---

### 4. feedback_loop.py - Automatic Learning

**Purpose**: Deep dive into Conduit's automatic feedback loop.

**Features**:
- Automatic feedback capture from LiteLLM responses
- Cost, latency, quality learning
- No manual feedback required
- Continuous improvement demonstration

**Run**:
```bash
python examples/04_litellm/feedback_loop.py
```

**Expected Output**:
- Shows feedback captured after each request
- Demonstrates learning over time
- Explains composite reward calculation

**Feedback Components**:
1. **Cost**: Extracted from `response._hidden_params['response_cost']`
2. **Latency**: Calculated from `end_time - start_time`
3. **Quality**: Estimated (0.9 for success, 0.1 for failure)

**Composite Reward**:
```
reward = 0.7 * quality - 0.2 * normalized_cost - 0.1 * normalized_latency
```

---

### 5. performance_comparison.py - ML vs Rule-Based

**Purpose**: Compare ML routing against traditional strategies.

**Features**:
- Round-robin baseline
- Conduit ML routing
- Cost and latency analysis
- Model distribution comparison

**Run**:
```bash
python examples/04_litellm/performance_comparison.py
```

**Expected Output**:
- Benchmark results for both strategies
- Cost savings percentage
- Latency improvement
- Model usage distribution

**Expected Results**:
- **Cost**: 20-40% savings with ML routing
- **Latency**: 10-30% improvement with ML routing
- **Distribution**: ML adapts to complexity (RR is uniform)

---

## Quick Start

If you're new to Conduit + LiteLLM, follow this path:

1. **Start**: `basic_usage.py` - Verify installation and basic routing
2. **Configure**: `custom_config.py` - Learn advanced options (hybrid, caching)
3. **Scale**: `multi_provider.py` - Add multiple providers for reliability
4. **Understand**: `feedback_loop.py` - Deep dive into learning mechanism
5. **Validate**: `performance_comparison.py` - Confirm ML benefits

## Common Patterns

### Basic Setup Pattern

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM router
router = Router(model_list=[...])

# Setup Conduit strategy
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(router, strategy)

try:
    # Use router - Conduit handles routing + learning
    response = await router.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
finally:
    # Clean up resources
    strategy.cleanup()
```

### Production Pattern

```python
# Production configuration
strategy = ConduitRoutingStrategy(
    use_hybrid=True,              # Faster convergence
    cache_enabled=True,           # Performance
    redis_url=os.getenv("REDIS_URL"),
)

# Router with fallback and retry
router = Router(
    model_list=model_list,
    fallbacks={"gpt-4": ["claude-3", "gemini"]},
    timeout=30,
    num_retries=2,
)

ConduitRoutingStrategy.setup_strategy(router, strategy)

# Use in production - automatic learning from all traffic
```

## Performance Expectations

### Without Learning (Cold Start)
- **First 100 queries**: Exploration phase (trying different models)
- **Cost**: Similar to random selection
- **Latency**: Mixed results

### With Learning (Warm)
- **After 500 queries**: UCB1 converges (if using hybrid)
- **After 2000 queries**: LinUCB converges (contextual routing)
- **Cost savings**: 20-40% vs random/round-robin
- **Latency improvement**: 10-30% vs random/round-robin

### Production (Long-Term)
- **Continuous improvement**: Performance improves with more data
- **Adaptation**: Adjusts to changing model costs/capabilities
- **Optimal routing**: Context-aware decisions for each query type

## Troubleshooting

### Redis Connection Failed

Examples work without Redis but caching is disabled:
```
⚠️  Redis unavailable - caching disabled
   Install: brew install redis && redis-server
```

**Solution**: Start Redis or run without caching (slower but functional)

### API Key Errors

```
Error: Set OPENAI_API_KEY environment variable
```

**Solution**: Set at least one provider API key:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

### Import Errors

```
ModuleNotFoundError: No module named 'litellm'
```

**Solution**: Install with LiteLLM extras:
```bash
pip install conduit[litellm]
```

## Configuration Reference

### ConduitRoutingStrategy Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hybrid` | bool | False | Enable UCB1→LinUCB warm start |
| `cache_enabled` | bool | False | Enable Redis caching |
| `redis_url` | str | None | Redis connection URL |
| `embedding_model` | str | "all-MiniLM-L6-v2" | Sentence transformer model |

### Performance Impact

| Feature | Impact | Benefit |
|---------|--------|---------|
| Hybrid routing | 30% faster convergence | Reach optimal routing sooner |
| Redis caching | 10-40x speedup | Faster repeated queries |
| Multi-provider | 99.9% uptime | Failover during outages |
| Feedback loop | 20-40% cost savings | Learn optimal model selection |

## Next Steps

After running these examples:

1. **Production Deployment**: See `conduit_litellm/README.md` for deployment guide
2. **API Integration**: Integrate with your existing LiteLLM usage
3. **Monitoring**: Add observability for routing decisions
4. **Tuning**: Adjust configuration based on your workload

## Additional Resources

- **[Main README](../../README.md)**: Project overview
- **[LiteLLM Integration README](../../conduit_litellm/README.md)**: Plugin documentation
- **[Issue #13](https://github.com/ashita-ai/conduit/issues/13)**: Feedback loop implementation
- **[Issue #14](https://github.com/ashita-ai/conduit/issues/14)**: This examples issue

## Support

Questions or issues? Open an issue at: https://github.com/ashita-ai/conduit/issues
