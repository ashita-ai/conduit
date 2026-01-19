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

## Core Examples (6 Files)

| File | Description | API Required |
|------|-------------|--------------|
| **hello_world.py** | Minimal 5-line example | Yes |
| **routing_options.py** | Constraints, preferences, algorithms | Yes |
| **feedback_loop.py** | Caching, learning, state persistence | Yes |
| **production_feedback.py** | User feedback integration patterns | Yes |
| **litellm_integration.py** | LiteLLM multi-provider routing | Yes |
| **hybrid_routing.py** | UCB1 to LinUCB phase transition | No |

### hello_world.py - Minimal Example

The simplest Conduit usage in 5 lines:

```python
from conduit.core.models import Query
from conduit.engines.router import Router

router = Router()
decision = await router.route(Query(text="What is 2+2?"))
print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")
```

### routing_options.py - All Routing Options

Comprehensive reference for routing configuration:

- **Constraints**: `max_cost`, `max_latency`, `min_quality`, `preferred_provider`
- **Preferences**: `optimize_for` (balanced, quality, cost, speed)
- **Algorithms**: `thompson_sampling`, `linucb`, `ucb1`, `epsilon_greedy`
- **Model Selection**: Custom model lists, provider preferences

### feedback_loop.py - Learning & Persistence

Shows how Conduit improves over time:

1. **Caching** - 10-40x speedup with Redis
2. **Learning** - Bandit algorithms optimize from feedback
3. **Persistence** - Save/restore state across restarts

### production_feedback.py - User Feedback Integration

Production patterns for collecting user feedback:

- Delayed feedback (track now, record later)
- Immediate feedback (automated checks)
- Multiple signal types (thumbs, ratings, task success)
- Custom adapter registration
- Batch feedback processing

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

response = await router.acompletion(model="llm", messages=[...])
```

### hybrid_routing.py - Algorithm Visualization

Demonstrates hybrid routing without API calls:

- Phase 1 (UCB1): Fast exploration, no embeddings
- Phase 2 (LinUCB): Contextual optimization with features
- Knowledge transfer between phases

## Framework Integrations

Located in `integrations/` with separate dependencies:

| Framework | Description | Run |
|-----------|-------------|-----|
| **LangChain** | Use Conduit as LangChain LLM | `uv run python examples/integrations/langchain_integration.py` |
| **LlamaIndex** | Use Conduit in RAG pipelines | `uv run python examples/integrations/llamaindex_integration.py` |
| **FastAPI** | Production REST API | `uv run python examples/integrations/fastapi_service.py --demo` |
| **Gradio** | Interactive web UI | `uv run python examples/integrations/gradio_demo.py` |

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
- **hybrid**: UCB1 to LinUCB transition (30% faster convergence)

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

## Additional Resources

- [Main README](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/](../docs/) - Design documentation
