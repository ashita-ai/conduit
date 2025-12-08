# Conduit

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

## Documentation

### Core
- [Architecture](ARCHITECTURE.md) - System design and components
- [Bandit Algorithms](BANDIT_ALGORITHMS.md) - ML algorithm details
- [Hybrid Routing](HYBRID_ROUTING_ALGORITHMS.md) - Combining exploration and exploitation

### Configuration
- [Embedding Providers](EMBEDDING_PROVIDERS.md) - Configure embeddings
- [Fallback](FALLBACK.md) - Fallback chain behavior
- [LiteLLM Integration](LITELLM_INTEGRATION.md) - 100+ provider support
- [Pricing](PRICING.md) - Cost tracking

### Reference
- [Glossary](GLOSSARY.md) - Terminology
- [Scope](SCOPE.md) - What Conduit does and doesn't do
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

## Key Features

- **Learning System**: Gets smarter with every query
- **Multi-Objective**: Balance cost, quality, and speed
- **100+ LLM Providers**: Via LiteLLM integration
- **Smart Caching**: Optional Redis for faster lookups
- **Per-Query Control**: Override optimization per query
- **Zero Config**: Auto-detects models from API keys

## How It Works

Conduit uses Thompson Sampling by default, a Bayesian bandit algorithm that:

1. **Models uncertainty** for each LLM using Beta distributions
2. **Samples** from each model's distribution to select
3. **Updates beliefs** after every query based on feedback

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
