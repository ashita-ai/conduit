# Conduit

![Conduit banner](assets/conduit.jpeg)

[![CI](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml/badge.svg)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Smart LLM routing that learns to cut costs without sacrificing quality.**

Conduit picks the right model for each query using contextual bandits. It learns from feedback, adapts to your workload, and gets smarter over time.

## Quick Start

```python
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")

asyncio.run(main())
```

## Installation

```bash
pip install conduit-router

# Set at least one API key
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

# Run an example
python examples/quickstart.py
```

**Optional**: Redis for caching, PostgreSQL for history persistence.

## Why Conduit?

You have multiple LLM options with 100x price differences. Most teams either overspend on expensive models or sacrifice quality with cheap ones.

Conduit solves this by learning which model works best for each query type:
- Simple questions route to fast, cheap models
- Complex tasks route to capable, expensive models
- The system improves automatically from feedback

**How it works**: Thompson Sampling explores models initially, then exploits what works. Reward = 70% quality + 20% cost efficiency + 10% speed.

## Features

- **Learns continuously** from every query
- **100+ LLM providers** via LiteLLM integration
- **Per-query control**: optimize for quality, cost, or speed
- **Zero config**: auto-detects models from API keys
- **Production-ready**: 91% test coverage, thread-safe

## User Preferences

```python
from conduit.core import Query, UserPreferences

# Optimize for cost on simple queries
query = Query(text="What is 2+2?", preferences=UserPreferences(optimize_for="cost"))

# Presets: balanced (default), quality, cost, speed
```

## Examples

```
examples/
├── quickstart.py              # Basic routing
├── routing_options.py         # Constraints and preferences
├── feedback_loop.py           # Learning from feedback
├── litellm_integration.py     # 100+ provider support
└── integrations/              # LangChain, LlamaIndex, FastAPI, Gradio
```

## When to Use Conduit

**Good fit**:
- $500+/month LLM spend
- 1000+ queries/day with varying complexity
- 3+ models in your stack

**Not a fit**:
- Single model only
- <100 queries/day
- Need streaming responses (not yet supported)

## Documentation

| Doc | Description |
|-----|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design |
| [Algorithms](docs/BANDIT_ALGORITHMS.md) | ML algorithm details |
| [LiteLLM](docs/LITELLM_INTEGRATION.md) | Extended provider support |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |
| [Development](AGENTS.md) | Contributing guide |

## Development

```bash
pytest --cov=conduit   # Tests (must be >80% coverage)
mypy conduit/          # Type checking
ruff check conduit/    # Linting
```

## Status

**Version 0.1.0** (Pre-1.0) - Core routing stable, APIs may change.

## License

MIT
