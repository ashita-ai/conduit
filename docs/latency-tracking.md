# Latency Tracking Service

## Overview

The LatencyService provides historical latency tracking and prediction for LLM models, replacing hardcoded provider-based heuristics with actual performance data from production usage.

## Features

- **Historical Tracking**: Records actual latency observations from LLM calls
- **Statistical Estimation**: Provides p50, p95, p99 percentile-based latency estimates
- **Context-Aware**: Tracks token count and complexity for future segmentation
- **Intelligent Fallback**: Uses provider-based heuristics when insufficient data
- **Configurable**: Adjustable time windows, percentiles, and minimum sample thresholds

## Database Schema

```sql
CREATE TABLE model_latencies (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    latency_seconds FLOAT NOT NULL,
    token_count INT,
    complexity_score FLOAT,
    region VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_latencies_model_time 
    ON model_latencies(model_id, created_at DESC);
```

## Configuration

Add to your `.env` file or environment:

```bash
# Enable latency tracking (default: True)
LATENCY_TRACKING_ENABLED=true

# Days of history to consider (default: 7)
LATENCY_ESTIMATION_WINDOW_DAYS=7

# Percentile for estimates (default: 0.95 for p95)
LATENCY_PERCENTILE=0.95

# Minimum samples before using historical data (default: 100)
LATENCY_MIN_SAMPLES=100
```

## Usage

### Automatic Recording (Default)

When `LATENCY_TRACKING_ENABLED=true`, the ModelExecutor automatically records latencies after each LLM call:

```python
from conduit.core import Database
from conduit.engines.executor import ModelExecutor

# Initialize with database
database = Database()
await database.connect()

executor = ModelExecutor(
    database=database,
    latency_tracking_enabled=True,  # Default
)

# Execute query - latency is recorded automatically
response = await executor.execute(
    model="gpt-4o-mini",
    prompt="What is 2+2?",
    result_type=MyResponse,
    query_id="query-123",
)
```

### Manual Usage

You can also use the LatencyService directly:

```python
from conduit.core import Database, LatencyService

database = Database()
await database.connect()

service = LatencyService(
    pool=database.pool,
    window_days=7,
    percentile=0.95,
    min_samples=100,
)

# Record actual latency
await service.record_latency(
    model_id="gpt-4o-mini",
    latency=1.23,
    query_features=features,  # Optional
)

# Get estimated latency for routing decisions
estimate = await service.get_estimated_latency("gpt-4o-mini")
print(f"Estimated p95 latency: {estimate:.2f}s")

# Get comprehensive statistics
stats = await service.get_latency_stats("gpt-4o-mini")
print(f"p50: {stats['p50']:.2f}s")
print(f"p95: {stats['p95']:.2f}s")
print(f"p99: {stats['p99']:.2f}s")
print(f"Samples: {stats['sample_count']}")
```

## Estimation Strategy

1. **Sufficient Data**: When >= `min_samples` exist within the time window, returns the configured percentile (default p95)
2. **Insufficient Data**: Falls back to provider-based heuristics:
   - OpenAI: 1.5s baseline
   - Anthropic: 1.8s baseline
   - Google: 1.2s baseline
   - Groq: 0.5s baseline
   - With multipliers for premium (1.3x) and mini (0.7x) models

## Migration

Run the Alembic migration to create the table:

```bash
alembic upgrade head
```

## Testing

The LatencyService includes comprehensive tests covering:
- Latency recording with and without features
- Estimation with sufficient/insufficient data
- Statistics retrieval
- Heuristic fallback
- Provider extraction
- Error handling

Run tests:
```bash
pytest tests/unit/test_latency_service.py -v
```

## Integration Points

1. **ModelExecutor**: Records latencies after LLM calls (automatic when enabled)
2. **Router**: Can use LatencyService for max_latency constraint filtering (future)
3. **API**: Exposes latency statistics via endpoints (future)
4. **Monitoring**: Surfaces latency trends in metrics (future)

## Future Enhancements

- [ ] Context-aware segmentation by token count and complexity
- [ ] Regional latency tracking and estimation
- [ ] Time-of-day pattern detection
- [ ] Anomaly detection and alerting
- [ ] Router integration for max_latency constraints
