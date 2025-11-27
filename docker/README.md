# Conduit Docker Setup

Run Conduit + LiteLLM Router with Open WebUI for visual demonstration of ML-powered routing.

## What This Demonstrates

- **ML-Powered Routing**: Conduit learns which model deployment to use for each query
- **Open WebUI Integration**: Visual chat interface showing Conduit's routing decisions
- **Multiple Deployments**: Same model (e.g., gpt-4o-mini) with multiple deployments for Conduit to route between
- **Automatic Learning**: Every request teaches Conduit about cost/quality/latency trade-offs

## Architecture

```
┌─────────────┐
│ Open WebUI  │  (Port 3000)
│ Chat UI     │
└──────┬──────┘
       │ OpenAI-compatible API
       ▼
┌─────────────────────┐
│   Conduit API       │  (Port 8000)
│ - LiteLLM Router    │
│ - Conduit Strategy  │
└──────┬──────────────┘
       │
       ├─► OpenAI (gpt-4o-mini-1, gpt-4o-mini-2)
       ├─► Anthropic (claude-3-5-haiku-1, claude-3-5-haiku-2)
       └─► Groq (llama-3.1-8b-1, llama-3.1-8b-2)
```

## Quick Start

### 1. Set Environment Variables

```bash
# Required: At least one LLM provider
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Additional providers
export GROQ_API_KEY="gsk-..."
export GOOGLE_API_KEY="..."
```

### 2. Start Services

```bash
# Build and start all services
docker compose -f docker-compose.openwebui.yml up --build

# Or run in background
docker compose -f docker-compose.openwebui.yml up -d --build
```

### 3. Access Open WebUI

Open browser to http://localhost:3000

**First Time Setup**:
1. Create an account (stored locally)
2. You'll see models like:
   - `gpt-4o-mini-1` and `gpt-4o-mini-2`
   - `claude-3-5-haiku-1` and `claude-3-5-haiku-2`
   - `llama-3.1-8b-1` and `llama-3.1-8b-2`

### 4. Test the Router

```bash
# Check services are healthy
docker ps

# Run test script
chmod +x docker/test_router.sh
./docker/test_router.sh
```

## How Conduit Routing Works

### Multiple Deployments per Model

The router is configured with **multiple deployments of the same model**:

```python
model_list = [
    {'model_name': 'gpt-4o-mini', 'model_info': {'id': 'gpt-4o-mini-1'}, ...},
    {'model_name': 'gpt-4o-mini', 'model_info': {'id': 'gpt-4o-mini-2'}, ...},
]
```

This gives Conduit options to route between and learn from.

### Learning Process

1. **Request comes in**: User sends "Explain quantum computing"
2. **Conduit selects deployment**: Uses LinUCB to pick gpt-4o-mini-1 or gpt-4o-mini-2
3. **LiteLLM executes**: Routes to OpenAI with selected deployment
4. **Feedback captured**: Cost, latency, quality recorded automatically
5. **Bandit updated**: Conduit learns which deployment performed better

### Routing Strategy

- **Early queries (0-2000)**: UCB1 bandit explores deployments
- **Later queries (2000+)**: LinUCB uses query context (embeddings) for smarter routing
- **Continuous learning**: Every request improves routing decisions

## Monitoring

### View Logs

```bash
# Router logs (see Conduit decisions)
docker logs conduit-litellm-router -f

# Open WebUI logs
docker logs open-webui -f

# Redis logs
docker logs conduit-redis -f
```

### Check Health

```bash
# Router health
curl http://localhost:8000/health

# Router info
curl http://localhost:8000/

# List models
curl http://localhost:8000/v1/models
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### List Models (OpenAI-compatible)
```bash
GET http://localhost:8000/v1/models
```

### Chat Completions (OpenAI-compatible)
```bash
POST http://localhost:8000/v1/chat/completions
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7,
  "max_tokens": 100
}
```

## Configuration

### Environment Variables

**Router (`conduit-litellm-router`)**:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google/Gemini API key
- `GROQ_API_KEY`: Groq API key
- `REDIS_URL`: Redis connection (default: redis://redis:6379)
- `USE_HYBRID_ROUTING`: Always true (hybrid routing is default)
- `LOG_LEVEL`: Logging level (default: INFO)

**Open WebUI (`open-webui`)**:
- `OPENAI_API_BASE_URL`: Points to Conduit router
- `OPENAI_API_KEY`: Any value (not used)
- `WEBUI_NAME`: Display name in UI

### Volumes

- `redis_data`: Redis persistence
- `open-webui-data`: Open WebUI user data and settings

## Troubleshooting

### Router shows "unhealthy"

Check logs for errors:
```bash
docker logs conduit-litellm-router
```

Common issues:
- Missing API keys → Set environment variables
- Port 8000 in use → Stop conflicting service
- Import errors → Rebuild with `--build` flag

### Open WebUI can't connect

1. Check router is healthy: `docker ps`
2. Test router directly: `curl http://localhost:8000/health`
3. Restart Open WebUI: `docker restart open-webui`

### No models showing in UI

1. Check model list: `curl http://localhost:8000/v1/models`
2. Verify API keys are set correctly
3. Check router logs for initialization errors

## Stopping Services

```bash
# Stop all services
docker compose -f docker-compose.openwebui.yml down

# Stop and remove volumes (clears data)
docker compose -f docker-compose.openwebui.yml down -v
```

## Files

- `docker-compose.openwebui.yml`: Docker Compose configuration
- `Dockerfile.router`: Router container image
- `litellm_router_api.py`: FastAPI server with Conduit integration
- `test_router.sh`: Test script for API endpoints

## Next Steps

1. **Try different queries** in Open WebUI and observe routing decisions in logs
2. **Add more providers** by setting additional API keys
3. **Monitor learning** by watching logs as Conduit improves routing
4. **Integrate with your app** using the OpenAI-compatible API at localhost:8000
