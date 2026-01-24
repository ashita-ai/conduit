# Feedback Integration Guide

This guide explains how to integrate user feedback into Conduit's bandit learning system for production applications.

## Overview

Conduit's feedback system enables continuous learning from user interactions. Unlike static routers, Conduit adapts based on:

- **Explicit feedback**: Thumbs up/down, ratings, task success indicators
- **Implicit feedback**: Errors, latency, retries, regeneration

## Quick Start

```python
from conduit import Router, Query
from conduit.feedback import FeedbackCollector, FeedbackEvent

# Setup
router = Router()
collector = FeedbackCollector(router)

# Route a query
decision = await router.route(Query(text="What is Python?"))

# Track for delayed feedback
await collector.track(decision, cost=0.001, latency=0.5)

# Later: Record user feedback
await collector.record(FeedbackEvent(
    query_id=decision.query_id,
    signal_type="thumbs",
    payload={"value": "up"}
))
```

## Feedback Patterns

### Pattern 1: Delayed Feedback (Recommended)

Use when feedback arrives seconds/minutes after the response:

```python
# 1. Route and execute
decision = await router.route(query)
response = await execute_llm(decision.selected_model, query.text)

# 2. Track the query (stores context for later)
await collector.track(
    decision,
    cost=response.cost,
    latency=response.latency,
)

# 3. Later: User provides feedback
await collector.record(FeedbackEvent(
    query_id=decision.query_id,
    signal_type="thumbs",
    payload={"value": "up"}
))
```

### Pattern 2: Immediate Feedback

Use when feedback is available immediately (e.g., automated evaluation):

```python
# Route, execute, and update in one flow
decision = await router.route(query)
response = await execute_llm(decision.selected_model, query.text)
quality_score = evaluate_response(response)  # Your evaluation

await router.update(
    model_id=decision.selected_model,
    cost=response.cost,
    quality_score=quality_score,
    latency=response.latency,
    features=decision.features,
)
```

## Built-in Adapters

Adapters convert feedback signals to bandit rewards:

| Adapter | Signal Type | Payload | Reward |
|---------|-------------|---------|--------|
| `ThumbsAdapter` | `thumbs` | `{"value": "up"\|"down"}` | 1.0 / 0.0 |
| `RatingAdapter` | `rating` | `{"rating": 1-5}` | normalized |
| `TaskSuccessAdapter` | `task_success` | `{"success": bool}` | 1.0 / 0.0 |
| `RegenerationAdapter` | `regeneration` | `{"regenerated": bool}` | 0.0 / 1.0 |
| `QualityScoreAdapter` | `quality_score` | `{"score": 0.0-1.0}` | score |

### Using Adapters

```python
from conduit.feedback import (
    FeedbackCollector,
    ThumbsAdapter,
    RatingAdapter,
    TaskSuccessAdapter,
)

collector = FeedbackCollector(router)

# Register additional adapters (ThumbsAdapter is registered by default)
collector.register_adapter(RatingAdapter())
collector.register_adapter(TaskSuccessAdapter())

# Record different signal types
await collector.record(FeedbackEvent(
    query_id="...",
    signal_type="rating",
    payload={"rating": 5}
))
```

### Custom Adapters

Create adapters for application-specific signals:

```python
from conduit.feedback import FeedbackAdapter, FeedbackEvent, RewardMapping

class ConversionAdapter(FeedbackAdapter):
    """Reward based on whether user converted (purchased, signed up, etc.)."""

    @property
    def signal_type(self) -> str:
        return "conversion"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        converted = event.payload.get("converted", False)
        return RewardMapping(
            reward=1.0 if converted else 0.3,  # Partial credit for engagement
            confidence=1.0,
        )

# Register and use
collector.register_adapter(ConversionAdapter())
```

## Feedback Storage

For production, use persistent storage to survive restarts:

### In-Memory (Development)

```python
from conduit.feedback import FeedbackCollector, InMemoryFeedbackStore

store = InMemoryFeedbackStore()
collector = FeedbackCollector(router, store=store)
```

### Redis (Production)

```python
from conduit.feedback import FeedbackCollector, RedisFeedbackStore
from redis.asyncio import Redis

redis = Redis.from_url("redis://localhost:6379")
store = RedisFeedbackStore(redis, ttl_seconds=3600)
collector = FeedbackCollector(router, store=store)
```

### PostgreSQL (Production)

```python
from conduit.feedback import FeedbackCollector, PostgresFeedbackStore

store = PostgresFeedbackStore(database_url="postgresql://...")
await store.initialize()  # Creates tables if needed
collector = FeedbackCollector(router, store=store)
```

## FastAPI Integration

Example webhook endpoint for receiving feedback:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from conduit import Router
from conduit.feedback import FeedbackCollector, FeedbackEvent

app = FastAPI()
router = Router()
collector = FeedbackCollector(router)

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID from routing response")
    signal_type: str = Field(default="thumbs", description="Feedback type")
    payload: dict = Field(default_factory=dict, description="Signal data")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    event = FeedbackEvent(
        query_id=request.query_id,
        signal_type=request.signal_type,
        payload=request.payload,
    )

    try:
        result = await collector.record(event)
        return {"status": "recorded", "reward": result.reward}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

## Session Feedback

For multi-turn conversations, propagate session-level feedback:

```python
from conduit.feedback import SessionFeedback

# At session end, record overall satisfaction
session_feedback = SessionFeedback(
    session_id="session_abc123",
    signal_type="rating",
    payload={"rating": 4},
    propagation_weight=0.5,  # 50% weight vs individual query feedback
)

await collector.record_session_feedback(session_feedback)
```

## Idempotency

The feedback system is idempotent. Recording the same feedback multiple times has the same effect as recording once:

```python
# These are deduplicated automatically:
await collector.record(FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"}))
await collector.record(FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"}))
# ^ Only counted once

# Different signal types are recorded separately:
await collector.record(FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"}))
await collector.record(FeedbackEvent(query_id="q1", signal_type="rating", payload={"rating": 5}))
# ^ Both recorded
```

For explicit deduplication keys:

```python
await collector.record(FeedbackEvent(
    query_id="q1",
    signal_type="thumbs",
    payload={"value": "up"},
    idempotency_key="client-request-uuid-12345",  # Your deduplication key
))
```

## Confidence Weighting

Adapters specify confidence levels for their signals:

- **1.0 (high)**: Explicit user action (thumbs, rating)
- **0.8 (medium)**: Strong implicit signal (regeneration, task failure)
- **0.5 (low)**: Weak signal (engagement time)

Low-confidence signals have reduced impact on bandit updates.

## Monitoring

Check feedback system health:

```python
# Pending queries count
pending = await collector.get_pending_count()

# Recent feedback events
history = await collector.get_recent_feedback(limit=100)

# Adapter statistics
for adapter in collector.adapters.values():
    print(f"{adapter.signal_type}: registered")
```

## Examples

See these example files for complete implementations:

- `examples/production_feedback.py` - Full production feedback patterns
- `examples/feedback_loop.py` - Implicit feedback with caching
- `examples/integrations/fastapi_service.py` - FastAPI integration

## Troubleshooting

### "Unknown query_id" Error

The query was not tracked before feedback arrived:

```python
# Always track after routing
decision = await router.route(query)
await collector.track(decision, cost=0.001, latency=0.5)  # Don't forget this!
```

### Feedback Not Affecting Routing

1. Verify the collector has a router reference
2. Check that adapters are registered for your signal types
3. Ensure TTL hasn't expired (default 1 hour for pending queries)

### High Memory Usage

Use Redis or PostgreSQL storage instead of in-memory for production.
