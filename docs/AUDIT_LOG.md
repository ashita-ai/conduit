# Decision Audit Log

The decision audit log captures detailed context for every routing decision made by Conduit. This enables:

- **Debugging**: Why did Conduit select model X for query Y?
- **Compliance**: Regulatory audit of AI decision-making (EU AI Act, SOC 2)
- **Analysis**: Post-mortem investigation of routing behavior
- **Optimization**: Understanding model selection patterns over time

## Quick Start

### Enable Audit Logging

```python
from conduit.core.database import Database
from conduit.engines.router import Router
from conduit.observability.audit import PostgresAuditStore

# Initialize database
db = Database()
await db.connect()

# Create audit store
audit_store = PostgresAuditStore(pool=db.pool, retention_days=90)

# Create router with audit logging enabled
router = Router(
    models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
    audit_store=audit_store,  # Enable audit logging
)

# Route queries (audit entries are created automatically)
decision = await router.route(query)
```

### Query Audit Log via API

```bash
# List recent audit entries
curl "http://localhost:8000/v1/audit?limit=10"

# Get specific decision
curl "http://localhost:8000/v1/audit/decision-uuid-here"

# Filter by model
curl "http://localhost:8000/v1/audit?selected_model=gpt-4o-mini"

# Filter by time range
curl "http://localhost:8000/v1/audit?start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z"
```

## Database Schema

The audit log is stored in the `decision_audit` table:

```sql
CREATE TABLE decision_audit (
    id BIGSERIAL PRIMARY KEY,

    -- Foreign keys
    decision_id TEXT NOT NULL,
    query_id TEXT NOT NULL,

    -- Decision snapshot
    selected_model TEXT NOT NULL,
    fallback_chain TEXT[] DEFAULT '{}',
    confidence NUMERIC(4,3) NOT NULL,

    -- Algorithm context
    algorithm_phase TEXT NOT NULL,  -- 'thompson_sampling', 'linucb', etc.
    query_count INTEGER NOT NULL,   -- Total queries at decision time

    -- Scores at decision time
    arm_scores JSONB NOT NULL,      -- Score breakdown for each model

    -- Feature vector (for contextual algorithms)
    feature_vector JSONB,           -- 386-dim vector

    -- Constraints applied
    constraints_applied JSONB DEFAULT '{}',

    -- Reasoning
    reasoning TEXT,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes

The table includes indexes for common query patterns:

- `idx_decision_audit_decision_id` - Lookup by decision ID
- `idx_decision_audit_query_id` - Lookup by query ID
- `idx_decision_audit_selected_model` - Filter by model
- `idx_decision_audit_algorithm_phase` - Filter by algorithm
- `idx_decision_audit_created_at` - Time range queries
- `idx_decision_audit_model_time` - Composite for model + time queries

## Audit Entry Structure

Each audit entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Auto-generated primary key |
| `decision_id` | str | UUID of the routing decision |
| `query_id` | str | UUID of the original query |
| `selected_model` | str | Model ID that was selected |
| `fallback_chain` | list[str] | Ordered fallback models |
| `confidence` | float | Decision confidence (0-1) |
| `algorithm_phase` | str | Algorithm used (e.g., "thompson_sampling", "linucb") |
| `query_count` | int | Total queries processed by router |
| `arm_scores` | dict | Score breakdown for each model |
| `feature_vector` | list[float] | 386-dim feature vector (contextual algorithms) |
| `constraints_applied` | dict | Constraints that affected the decision |
| `reasoning` | str | Human-readable explanation |
| `created_at` | datetime | When the decision was made |

### Score Structure

The `arm_scores` field contains algorithm-specific score breakdowns:

**Thompson Sampling:**
```json
{
  "gpt-4o-mini": {
    "alpha": 10.5,
    "beta": 2.3,
    "mean": 0.82,
    "variance": 0.01,
    "total": 0.82
  }
}
```

**LinUCB:**
```json
{
  "gpt-4o-mini": {
    "mean": 0.72,
    "uncertainty": 0.15,
    "total": 0.87
  }
}
```

## API Endpoints

### GET /v1/audit

Query audit log entries with optional filters.

**Query Parameters:**
- `decision_id` (str): Filter by specific decision
- `query_id` (str): Filter by specific query
- `selected_model` (str): Filter by model ID
- `algorithm_phase` (str): Filter by algorithm phase
- `start_time` (str): ISO timestamp, entries after this time
- `end_time` (str): ISO timestamp, entries before this time
- `limit` (int): Maximum entries to return (default: 100, max: 1000)
- `offset` (int): Skip this many entries (pagination)

**Response:**
```json
{
  "entries": [
    {
      "id": 1234,
      "decision_id": "550e8400-e29b-41d4-a716-446655440000",
      "query_id": "661f5501-f39c-52e5-b827-557766551111",
      "selected_model": "gpt-4o-mini",
      "fallback_chain": ["gpt-4o"],
      "confidence": 0.92,
      "algorithm_phase": "thompson_sampling",
      "query_count": 1500,
      "arm_scores": {"gpt-4o-mini": {"mean": 0.85, "total": 0.85}},
      "feature_vector": null,
      "constraints_applied": {},
      "reasoning": "Selected for cost efficiency",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### GET /v1/audit/{decision_id}

Get audit entry for a specific routing decision.

**Response:**
```json
{
  "id": 1234,
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_id": "661f5501-f39c-52e5-b827-557766551111",
  "selected_model": "gpt-4o-mini",
  ...
}
```

## Retention Policy

Configure how long audit entries are retained:

```python
# Keep entries for 90 days (default)
audit_store = PostgresAuditStore(pool=db.pool, retention_days=90)

# Keep entries forever
audit_store = PostgresAuditStore(pool=db.pool, retention_days=0)

# Apply retention policy manually
deleted = await audit_store.apply_retention_policy()
print(f"Deleted {deleted} old entries")
```

For automated cleanup, schedule `apply_retention_policy()` via cron or similar:

```python
# Example: daily cleanup task
import asyncio

async def daily_cleanup():
    while True:
        deleted = await audit_store.apply_retention_policy()
        logger.info(f"Retention cleanup: deleted {deleted} entries")
        await asyncio.sleep(86400)  # 24 hours
```

## Development and Testing

### In-Memory Store

For testing without a database:

```python
from conduit.observability.audit import InMemoryAuditStore

# Use in-memory store for tests
audit_store = InMemoryAuditStore(max_entries=10000)

router = Router(
    models=["gpt-4o-mini"],
    audit_store=audit_store,
)
```

### Running Migrations

Apply the audit log migration:

```bash
# Apply all migrations
alembic upgrade head

# Apply only the audit log migration
alembic upgrade 16742edc4c01
```

### Manual Audit Entry Creation

```python
from conduit.observability.audit import AuditEntry, create_audit_entry

# Create from routing decision
entry = create_audit_entry(
    decision=decision,
    algorithm_phase="linucb",
    query_count=5000,
    arm_scores=bandit.compute_scores(features),
    constraints_applied={"max_cost": 0.01},
)

# Log manually
await audit_store.log_decision(entry)
```

## Compliance Use Cases

### EU AI Act

Article 14 of the EU AI Act requires high-risk AI systems to maintain logs of:
- Input data characteristics
- Decision rationale
- System state at decision time

The audit log captures all of this:
- `feature_vector` contains input characteristics
- `reasoning` explains decision rationale
- `arm_scores`, `algorithm_phase`, `query_count` capture system state

### SOC 2

For SOC 2 compliance, enable comprehensive logging:

```python
# Configure audit with maximum retention
audit_store = PostgresAuditStore(
    pool=db.pool,
    retention_days=365 * 7,  # 7 years for SOC 2
)
```

Export audit data for compliance reports:

```python
# Query all decisions in time range
entries = await audit_store.query(AuditQuery(
    start_time=start_of_quarter,
    end_time=end_of_quarter,
    limit=10000,
))

# Export to CSV/JSON for compliance report
export_audit_report(entries, format="csv")
```

## Performance Considerations

### Write Performance

Audit logging adds minimal overhead:
- Async database writes do not block routing
- Failures are logged but do not affect routing decisions
- Batching can be implemented if needed for high-throughput scenarios

### Query Performance

Use indexes for efficient queries:
- Filter by `selected_model` for model-specific analysis
- Filter by `created_at` for time-range reports
- Use `limit` and `offset` for pagination

### Storage

Estimate storage requirements:
- Each entry is approximately 2-5 KB (varies with feature_vector size)
- 100,000 entries/day = 200-500 MB/day
- Implement retention policy to manage storage

## Troubleshooting

### Audit logging not working

1. Check that `audit_store` was passed to Router:
   ```python
   router = Router(audit_store=audit_store)  # Required
   ```

2. Verify database connection:
   ```python
   # Test database connectivity
   await db.pool.execute("SELECT 1")
   ```

3. Check migration was applied:
   ```bash
   alembic current  # Should show audit migration
   ```

### Missing entries

Check for errors in logs:
```python
import logging
logging.getLogger("conduit.engines.router").setLevel(logging.DEBUG)
```

Audit errors are logged but do not affect routing.

### Slow queries

Add appropriate indexes or use pagination:
```python
# Use smaller page sizes
entries = await audit_store.query(AuditQuery(limit=100, offset=0))
```
