# Conduit LiteLLM Integration

ML-powered routing strategy for LiteLLM's 100+ providers with automatic feedback loop learning.

## Installation

```bash
pip install conduit[litellm]
```

## Usage

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM router with your models
router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o-mini"},
        },
        {
            "model_name": "claude-3",
            "litellm_params": {"model": "claude-3-haiku"},
        }
    ]
)

# Setup Conduit routing strategy (feedback loop auto-enabled)
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(router, strategy)

try:
    # LiteLLM uses ML routing + learns from every request
    response = await router.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    # Feedback captured automatically: cost, latency, quality
finally:
    strategy.cleanup()  # Clean up resources
```

## Features

- **ML-Powered Selection**: Contextual bandits (LinUCB, Thompson Sampling) learn optimal routing
- **Automatic Learning**: Feedback loop captures cost/latency from every request (Issue #13 complete)
- **100+ Providers**: Works with all LiteLLM-supported providers
- **Hybrid Routing**: UCB1→LinUCB warm start for faster convergence
- **Resource Management**: Cleanup method prevents memory leaks
- **Async & Sync**: Supports both contexts (Issue #31 fixed)

## Configuration Options

```python
strategy = ConduitRoutingStrategy(
    use_hybrid=True,              # Enable UCB1→LinUCB warm start
    cache_enabled=True,           # Enable Redis caching
    redis_url="redis://localhost:6379"
)
```

## Quality Measurement with Arbiter (Issue #52)

Enable LLM-as-judge quality assessment using Arbiter evaluator (optional):

```python
from conduit.evaluation import ArbiterEvaluator
from conduit_litellm import ConduitRoutingStrategy

# Initialize Arbiter evaluator with your database
evaluator = ArbiterEvaluator(
    db=database,
    sample_rate=0.1  # Evaluate 10% of queries to control costs
)

# Pass evaluator to routing strategy
strategy = ConduitRoutingStrategy(
    use_hybrid=True,
    evaluator=evaluator  # Enable LLM-as-judge quality measurement
)
ConduitRoutingStrategy.setup_strategy(router, strategy)

# Quality scores now feed back to bandit learning
# - Semantic similarity (reference-free)
# - Factuality assessment
# - Fire-and-forget (doesn't block routing)
```

**Features**:
- Non-blocking fire-and-forget evaluation
- Configurable sampling rate to control API costs
- Automatic cost tracking via Arbiter framework
- Stores feedback for bandit learning
- Backward compatible (evaluator is optional)

## Testing

**Note**: LiteLLM is an optional dependency. Tests require `pip install conduit[litellm]`.

```bash
# Run LiteLLM integration tests
pytest tests/unit/test_litellm_strategy.py -v
```

## Issue #31 Fix

The sync `get_available_deployment()` method now correctly handles async contexts by running the async version in a separate thread when an event loop is already running. This prevents the `RuntimeError: This event loop is already running` error.

**Before** (Issue #31):
```python
# Raised RuntimeError in async contexts
return loop.run_until_complete(self.async_get_available_deployment(...))
```

**After** (Fixed):
```python
# Runs in separate thread when event loop exists
with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(asyncio.run, self.async_get_available_deployment(...))
    return future.result()
```

## Related Issues

- #9 - LiteLLM integration (parent issue)
- #13 - **Complete**: Feedback collection and learning
- #14 - LiteLLM integration examples
- #15 - LiteLLM plugin usage documentation
- #31 - **Fixed**: RuntimeError in async contexts
- #52 - **Complete**: Arbiter LLM-as-judge integration
