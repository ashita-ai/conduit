# Conduit Examples

Demonstrations of Conduit's intelligent LLM routing capabilities.

## Getting Started

```bash
# Install Conduit
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."

# Optional: Install Redis for caching
brew install redis && redis-server
```

## Examples (4 Core Files)

| File | Description | Run Command |
|------|-------------|-------------|
| **quickstart.py** | Basic routing in 10 lines | `uv run python examples/quickstart.py` |
| **routing_options.py** | Constraints, preferences, algorithms | `uv run python examples/routing_options.py` |
| **feedback_loop.py** | Caching, learning, state persistence | `uv run python examples/feedback_loop.py` |
| **litellm_integration.py** | LiteLLM multi-provider routing | `uv run python examples/litellm_integration.py` |

### quickstart.py - Basic Routing

The simplest Conduit usage in 10 lines:

```python
from conduit.core.models import Query
from conduit.engines.router import Router

router = Router()
query = Query(text="What is 2+2?")
decision = await router.route(query)
print(f"Model: {decision.selected_model}, Confidence: {decision.confidence:.0%}")
```

### routing_options.py - Advanced Routing

Demonstrates all routing configuration:

- **Constraints**: `max_cost`, `max_latency`, `min_quality`, `preferred_provider`
- **Preferences**: `optimize_for` (balanced, quality, cost, speed)
- **Algorithms**: `thompson_sampling`, `linucb`, `ucb1`, `epsilon_greedy`
- **Model Selection**: Custom model lists, provider preferences

### feedback_loop.py - Learning & Persistence

Shows how Conduit improves over time:

1. **Caching** - 10-40x speedup with Redis
2. **Learning** - Bandit algorithms optimize from feedback
3. **Persistence** - Save/restore state across restarts

### litellm_integration.py - LiteLLM Integration

Use Conduit as an intelligent routing strategy for LiteLLM:

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

model_list = [
    {"model_name": "llm", "litellm_params": {"model": "gpt-4o-mini"}, ...},
    {"model_name": "llm", "litellm_params": {"model": "gpt-4o"}, ...},
]

router = Router(model_list=model_list)
strategy = ConduitRoutingStrategy()
ConduitRoutingStrategy.setup_strategy(router, strategy)

# Conduit automatically routes to optimal model
response = await router.acompletion(model="llm", messages=[...])
```

## Framework Integrations

Located in `integrations/` with separate dependencies:

| Framework | Description | Run |
|-----------|-------------|-----|
| **LangChain** | Use Conduit as LangChain LLM - automatic routing in chains | `uv run python examples/integrations/langchain_integration.py` |
| **LlamaIndex** | Use Conduit in RAG pipelines - intelligent model selection | `uv run python examples/integrations/llamaindex_integration.py` |
| **FastAPI** | Production REST API with routing, feedback, health checks | `uv run python examples/integrations/fastapi_service.py --demo` |
| **Gradio** | Interactive web UI for testing routing decisions | `uv run python examples/integrations/gradio_demo.py` |

**FastAPI server mode:**
```bash
uv run uvicorn examples.integrations.fastapi_service:app --reload
# Then open http://127.0.0.1:8000/docs for API documentation
```

## Additional Examples

All examples are now in the root `examples/` directory:

- **hello_world.py** - Minimal "hello world" example
- **simple_router.py** - Basic router setup
- **basic_routing.py** - Routing with different models
- **with_constraints.py** - Cost and quality constraints
- **hybrid_routing.py** - Hybrid bandit algorithms
- **context_specific_priors.py** - Domain-specific priors
- **caching.py** - Redis caching setup
- **pca_comparison.py** - Feature dimensionality reduction
- **state_persistence.py** - Save/load bandit state
- **basic_usage.py** - LiteLLM basic usage
- **custom_config.py** - Custom LiteLLM configuration
- **learning_demo.py** - Learning from feedback
- **multi_provider.py** - Multi-provider routing
- **arbiter_quality_measurement.py** - LLM-as-judge quality
- **explicit_preferences.py** - User preference handling

## Key Concepts

### Query -> Routing Decision -> Feedback Loop

```
Query("Explain ML") -> Router -> RoutingDecision(model="gpt-4o", confidence=0.85)
                         ^                           |
                    Feedback <- BanditFeedback(quality=0.92, cost=0.01)
```

### Algorithms

- **thompson_sampling** (default): Bayesian bandit, best cold-start quality
- **linucb**: Contextual linear bandit, uses query features
- **ucb1**: Upper confidence bound, simple exploration
- **hybrid**: UCB1 -> LinUCB transition (30% faster convergence)

### Configuration

```python
# Constraints
QueryConstraints(max_cost=0.01, min_quality=0.9)

# Preferences
UserPreferences(optimize_for="quality")  # balanced, quality, cost, speed

# Router
Router(algorithm="linucb", models=["gpt-4o-mini", "gpt-4o"])
```

## Redis Setup (Optional)

```bash
# macOS
brew install redis && brew services start redis

# Linux
sudo apt-get install redis-server && sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

## CLI Alternative

```bash
conduit run --query "What is 2+2?"
conduit run --query "Write a poem" --max-cost 0.001
conduit demo  # Run 10 sample queries
```

## Additional Resources

- [Main README](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/](../docs/) - Design documentation
