# Conduit Router

**Smart LLM routing that learns to cut costs without sacrificing quality.**

Conduit is an ML-powered routing system that learns which LLM to use for each type of query. Like a chess engine that improves with every game, Conduit gets better at picking the right model as it sees more queries.

## Quick Start

```python
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model}")

asyncio.run(main())
```

## Why Conduit?

| Feature | Conduit | Static Routers | Manual Selection |
|---------|---------|----------------|------------------|
| **Cost Optimization** | Learns cheapest model per query type | Fixed rules, no learning | Expensive models everywhere |
| **Quality** | Balances cost vs quality via feedback | Depends on rules | High but wasteful |
| **Adapts to Changes** | Learns when models improve/degrade | Manual rule updates needed | Manual switching |
| **Setup Time** | 5 lines of code | Complex rule configuration | Simple but inefficient |

## Documentation

```{toctree}
:maxdepth: 2
:caption: Core Documentation

ARCHITECTURE
BANDIT_ALGORITHMS
HYBRID_ROUTING_ALGORITHMS
```

```{toctree}
:maxdepth: 2
:caption: Configuration

EMBEDDING_PROVIDERS
FALLBACK
LITELLM_INTEGRATION
```

```{toctree}
:maxdepth: 2
:caption: Reference

GLOSSARY
SCOPE
TROUBLESHOOTING
```

## Key Features

- **Learning System**: Gets smarter with every query, learns your workload patterns automatically
- **Multi-Objective**: Balance cost, quality, and speed based on what matters to you
- **100+ LLM Providers**: Direct support for 8 providers, extended via LiteLLM integration
- **Smart Caching**: Faster embedding lookups on repeated queries (optional Redis)
- **Per-Query Control**: Override optimization per query (quality-first, cost-first, speed-first)
- **Zero Config**: Auto-detects available models from your API keys

## Installation

```bash
# Clone and install
git clone https://github.com/ashita-ai/conduit.git
cd conduit && pip install -e .

# Set API keys (at least one required)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

# Run example
python examples/hello_world.py
```

## How It Works

Conduit uses Thompson Sampling by default, a Bayesian bandit algorithm that:

1. **Models uncertainty** for each LLM using Beta distributions
2. **Samples** from each model's distribution to select
3. **Updates beliefs** after every query based on success/failure

The reward formula balances quality, cost, and speed:
```
reward = 0.7 * quality + 0.2 * cost_efficiency + 0.1 * speed
```

## Links

- [GitHub Repository](https://github.com/ashita-ai/conduit)
- [Issue Tracker](https://github.com/ashita-ai/conduit/issues)
- [Examples](https://github.com/ashita-ai/conduit/tree/main/examples)

## License

MIT License - see [LICENSE](https://github.com/ashita-ai/conduit/blob/main/LICENSE) file.
