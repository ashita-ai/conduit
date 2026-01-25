# Operations Guide

This guide covers operational aspects of running Conduit in production, including deployment, graceful shutdown, monitoring, and troubleshooting.

## Graceful Shutdown

Conduit supports graceful shutdown to ensure no data loss or broken connections during deployments. This is critical for:

- Kubernetes rolling updates
- Docker container restarts
- Manual server restarts

### How It Works

When Conduit receives a shutdown signal (SIGTERM or SIGINT), it executes the following sequence:

1. **Stop accepting new requests** - The server immediately stops accepting new connections
2. **Drain in-flight requests** - Wait up to 30 seconds for active requests to complete
3. **Persist bandit state** - Save learned model weights to PostgreSQL
4. **Close database connections** - Gracefully close the connection pool
5. **Log completion** - Record shutdown timing for debugging

### Shutdown Timeline

```
Signal Received
     |
     v
+--------------------+
| Phase 1: Drain     | <-- Up to 30s for in-flight requests
| (in-flight reqs)   |
+--------------------+
     |
     v
+--------------------+
| Phase 2: Persist   | <-- Save bandit state to PostgreSQL
| (router state)     |
+--------------------+
     |
     v
+--------------------+
| Phase 3: Close     | <-- Up to 10s for pool close
| (db connections)   |
+--------------------+
     |
     v
   Complete
```

### Configuration

The shutdown timeout can be configured when creating the lifecycle manager:

```python
from conduit.core.lifecycle import LifecycleManager

manager = LifecycleManager(
    router=router,
    database=database,
    shutdown_timeout=30.0,  # Seconds to wait for request drain
)
```

### Kubernetes Integration

For Kubernetes deployments, configure the termination grace period to allow enough time for graceful shutdown:

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 45  # Should be > shutdown_timeout + buffer
      containers:
        - name: conduit
          # ...
```

**Recommended values:**
- `terminationGracePeriodSeconds`: 45 seconds (allows 30s drain + 10s db close + 5s buffer)
- Conduit `shutdown_timeout`: 30 seconds (default)

### Docker Integration

For Docker containers, use the `--stop-timeout` flag or configure in docker-compose:

```yaml
services:
  conduit:
    image: conduit:latest
    stop_grace_period: 45s
```

### Monitoring Shutdown

Conduit logs shutdown progress at INFO level. Look for these messages:

```
INFO - Starting graceful shutdown sequence
INFO - Phase 1/3: Draining N requests
INFO - All N requests drained in X.Xs
INFO - Phase 2/3: Persisting bandit state
INFO - Bandit state persisted successfully
INFO - Phase 3/3: Closing database connections
INFO - PostgreSQL connection pool closed gracefully
INFO - Graceful shutdown completed in X.Xs
```

If shutdown times out or has errors:

```
WARNING - Timeout waiting for requests, N still in flight
ERROR - Failed to persist router state: <error>
WARNING - Pool close timed out after 10s, terminating connections
WARNING - Shutdown completed with N errors in X.Xs
```

### State Persistence

Bandit state is automatically persisted during shutdown. This ensures:

- No loss of learned model preferences
- Restarts resume from previous learning state
- Multi-replica deployments share state via PostgreSQL

**State persistence order is critical:**
1. Bandit state saved BEFORE database pool closes
2. If state save fails, error is logged but shutdown continues
3. Database connections are closed last

### Troubleshooting Shutdown Issues

#### Requests Not Draining

If requests don't complete within the timeout:

1. Check for slow downstream dependencies (LLM APIs)
2. Verify request timeout settings
3. Consider increasing `shutdown_timeout` for slow workloads

#### State Persistence Failures

If bandit state fails to persist:

1. Check PostgreSQL connectivity
2. Verify database permissions
3. Check disk space on PostgreSQL server
4. Review error logs for specific failure reason

#### Connection Pool Timeout

If the database pool times out during close:

1. Check for long-running transactions
2. Verify no connection leaks in application code
3. Consider reducing pool size for faster drain

### Manual Shutdown

For programmatic shutdown (e.g., in tests or custom applications):

```python
from conduit.core.lifecycle import LifecycleManager

# Create manager
manager = LifecycleManager(router=router, database=database)

# Install signal handlers (optional)
manager.install_signal_handlers()

# ... application runs ...

# Trigger shutdown manually
state = await manager.shutdown()

print(f"Shutdown completed in {state.duration_seconds:.1f}s")
if state.errors:
    print(f"Errors: {state.errors}")
```

### FastAPI Integration

The default FastAPI application automatically integrates graceful shutdown:

```python
from conduit.api.app import create_app, get_lifecycle_manager

app = create_app()

# The lifecycle manager is automatically configured and available:
manager = get_lifecycle_manager()
```

## Health Checks

### Liveness Probe

The `/health` endpoint returns basic service status:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Readiness Probe

For Kubernetes readiness probes, use the same `/health` endpoint. During shutdown, the endpoint will return unhealthy status once request draining begins.

## Metrics

Conduit exposes metrics via OpenTelemetry. Key metrics for operations:

- `conduit_requests_total` - Total request count
- `conduit_requests_active` - Current in-flight requests
- `conduit_routing_latency_seconds` - Routing decision latency
- `conduit_model_selection` - Model selection counts by model

## Logging

### Log Levels

Configure via `LOG_LEVEL` environment variable:

- `DEBUG` - Detailed routing decisions, state changes
- `INFO` - Startup, shutdown, significant events (recommended for production)
- `WARNING` - Recoverable errors, timeouts
- `ERROR` - Failures requiring attention

### Structured Logging

Conduit uses structured logging compatible with JSON formatters. Key fields:

- `router_id` - Router instance identifier
- `model_id` - Selected model
- `latency_ms` - Operation latency
- `phase` - Shutdown phase (during shutdown)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string (optional) | None |
| `LOG_LEVEL` | Logging level | INFO |
| `CONDUIT_ROUTER_ID` | Router instance ID for state persistence | Auto-generated |

## Deployment Checklist

Before deploying to production:

- [ ] Configure `DATABASE_URL` with production PostgreSQL
- [ ] Set appropriate `terminationGracePeriodSeconds` (Kubernetes)
- [ ] Configure health check probes
- [ ] Verify log aggregation is working
- [ ] Test shutdown behavior in staging
- [ ] Configure monitoring/alerting on error logs
- [ ] Set `CONDUIT_ROUTER_ID` for multi-replica deployments
