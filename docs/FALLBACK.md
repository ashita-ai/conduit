# Fallback Chains

**Last Updated**: 2025-11-29

Automatic model fallback when primary model fails. Essential for production resilience.

## Overview

When a selected model fails (rate limit, timeout, error), Conduit automatically tries the next-best model in the fallback chain instead of surfacing errors to users.

```
Primary Model → Fails → Fallback 1 → Fails → Fallback 2 → Success!
                  ↓                     ↓
            Penalized            Penalized
```

## Quick Start

```python
from conduit.engines.router import Router
from conduit.engines.executor import ModelExecutor, AllModelsFailedError
from conduit.core.models import Query
from pydantic import BaseModel

class MyOutput(BaseModel):
    answer: str

async def main():
    router = Router()
    executor = ModelExecutor()

    query = Query(text="What is 2+2?")
    decision = await router.route(query)

    # decision.fallback_chain contains ordered fallback models
    print(f"Primary: {decision.selected_model}")
    print(f"Fallbacks: {decision.fallback_chain}")

    try:
        # Automatic fallback on failure
        result = await executor.execute_with_fallback(
            decision=decision,
            prompt=query.text,
            result_type=MyOutput,
        )

        if result.was_fallback:
            print(f"Used fallback: {result.model_used}")
            print(f"Failed models: {result.failed_models}")

        # Proper feedback attribution
        await router.update_with_fallback_attribution(
            execution_result=result,
            quality_score=0.95,  # Your quality assessment
            features=decision.features,
        )

    except AllModelsFailedError as e:
        print(f"All {len(e.errors)} models failed")
        for model, error in e.errors:
            print(f"  {model}: {error}")
```

## How It Works

### 1. Fallback Chain Generation

When routing a query, Conduit generates an ordered fallback chain based on expected quality:

```python
decision = await router.route(query)

# Example output:
# decision.selected_model = "gpt-4o"
# decision.fallback_chain = ["gpt-4o-mini", "claude-3-haiku", "gemini-flash"]
```

The fallback chain is sorted by `expected_quality` (descending), excluding the selected model.

### 2. Execution with Fallback

The `execute_with_fallback()` method tries models in order:

```python
result = await executor.execute_with_fallback(
    decision=decision,
    prompt="Your prompt",
    result_type=YourOutputModel,
    timeout=60.0,        # Per-model timeout
    max_fallbacks=3,     # Max fallback attempts (default: 3)
)
```

**ExecutionResult fields:**
- `response`: The successful Response object
- `model_used`: Which model produced the response
- `was_fallback`: True if a fallback model was used
- `original_model`: The originally selected model
- `failed_models`: List of models that failed before success

### 3. Feedback Attribution

Proper feedback ensures the bandit learns from failures:

```python
await router.update_with_fallback_attribution(
    execution_result=result,
    quality_score=0.95,
    features=decision.features,
)
```

This method:
1. **Penalizes failed models** (quality=0.0) to reduce future selection
2. **Rewards successful model** with actual quality score

This is critical for learning - without proper attribution, the router won't learn which models are unreliable.

## Error Handling

### AllModelsFailedError Exception

Raised when all models (primary + fallbacks) fail:

```python
from conduit.engines.executor import AllModelsFailedError

try:
    result = await executor.execute_with_fallback(...)
except AllModelsFailedError as e:
    # e.errors is a list of (model_id, exception) tuples
    for model, error in e.errors:
        print(f"{model} failed: {error}")

    # Handle gracefully - e.g., return cached response, queue for retry
```

### Errors That Trigger Fallback

- `ExecutionError`: Model returned error or invalid response
- `asyncio.TimeoutError`: Model exceeded timeout
- Rate limit errors (wrapped in ExecutionError)
- API errors (wrapped in ExecutionError)

## Configuration

### Per-Query Configuration

```python
result = await executor.execute_with_fallback(
    decision=decision,
    prompt=prompt,
    result_type=OutputModel,
    timeout=30.0,       # 30 second timeout per model
    max_fallbacks=2,    # Only try 2 fallbacks (3 total models)
)
```

### Router-Level Configuration

The fallback chain length is determined by available models in the router:

```python
router = Router(
    models=["gpt-4o", "gpt-4o-mini", "claude-3-haiku", "gemini-flash"],
)
# Fallback chain will have up to 3 models (excluding selected)
```

## Best Practices

### 1. Always Use Feedback Attribution

```python
# GOOD: Proper attribution
await router.update_with_fallback_attribution(result, quality, features)

# BAD: Only updating successful model (failed models not penalized)
await router.update(result.model_used, ...)
```

### 2. Set Appropriate Timeouts

```python
# Fast models: shorter timeout
result = await executor.execute_with_fallback(
    ...,
    timeout=30.0,  # 30 seconds for simple queries
)

# Complex queries: longer timeout
result = await executor.execute_with_fallback(
    ...,
    timeout=120.0,  # 2 minutes for complex reasoning
)
```

### 3. Handle AllModelsFailedError Gracefully

```python
try:
    result = await executor.execute_with_fallback(...)
except AllModelsFailedError:
    # Options:
    # 1. Return cached response if available
    # 2. Queue for retry with backoff
    # 3. Return degraded response
    # 4. Alert operations team
    pass
```

### 4. Monitor Fallback Rate

Track how often fallbacks are used to identify unreliable models:

```python
if result.was_fallback:
    metrics.increment("fallback_used", tags={
        "original": result.original_model,
        "used": result.model_used,
    })
```

## API Reference

### RoutingDecision

```python
@dataclass
class RoutingDecision:
    selected_model: str           # Primary model to try
    fallback_chain: list[str]     # Ordered fallback models
    confidence: float             # Routing confidence
    features: QueryFeatures       # Query features for feedback
    # ... other fields
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    response: Response            # Successful response
    model_used: str               # Model that succeeded
    was_fallback: bool = False    # True if fallback was used
    original_model: str = ""      # Originally selected model
    failed_models: list[str]      # Models that failed
```

### AllModelsFailedError

```python
class AllModelsFailedError(ExecutionError):
    """Raised when all models (primary + fallbacks) fail."""
    errors: list[tuple[str, Exception]]  # (model_id, error) pairs
```

### Methods

```python
# ModelExecutor
async def execute_with_fallback(
    decision: RoutingDecision,
    prompt: str,
    result_type: type[BaseModel],
    timeout: float = 60.0,
    max_fallbacks: int = 3,
) -> ExecutionResult

# Router
async def update_with_fallback_attribution(
    execution_result: ExecutionResult,
    quality_score: float,
    features: QueryFeatures,
) -> None
```

## Integration with Circuit Breaker

Conduit's fallback system works alongside the existing circuit breaker in `conduit/cache/circuit_breaker.py`. When a model repeatedly fails, the circuit breaker can temporarily exclude it from selection, while the fallback system handles individual request failures.

Future enhancement: Integrate circuit breaker state with fallback chain generation to automatically deprioritize models with open circuits.
