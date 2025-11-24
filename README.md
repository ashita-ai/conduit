# Conduit

ML-powered LLM routing system that learns optimal model selection for cost, latency, and quality optimization.

## Overview

Conduit uses contextual bandits (Thompson Sampling) to intelligently route queries to the optimal LLM model based on learned patterns from usage data. Unlike static rule-based routers, Conduit continuously improves routing decisions through feedback loops.

## Key Features

- **ML-Driven Routing**: Learns from usage patterns vs static IF/ELSE rules
- **Multi-Objective Optimization**: Balance cost, latency, and quality constraints
- **Flexible Provider Support**:
  - Direct: 8 providers via PydanticAI - OpenAI, Anthropic, Google, Groq, Mistral, Cohere, AWS Bedrock, HuggingFace (structured outputs, type safety)
  - Extended: 100+ providers via LiteLLM integration (see `conduit_litellm/` and `docs/LITELLM_INTEGRATION.md`)
- **Dual Feedback Loop**: Explicit (user ratings) + Implicit (errors, latency, retries)
- **Redis Caching**: 10-40x performance improvement on repeated queries
- **Graceful Degradation**: Core routing works without Redis
- **9 Bandit Algorithms**: Contextual Thompson Sampling, LinUCB, Thompson Sampling, UCB1, Epsilon-Greedy, + 4 baselines
- **Multi-Objective Rewards**: Composite rewards (70% quality + 20% cost + 10% latency)
- **Non-Stationarity Handling**: Sliding window adaptation for changing model quality/costs
- **Dynamic Pricing**: 71+ models with auto-updated pricing from llm-prices.com (24h cache)
- **Model Discovery**: Auto-detects available models based on your API keys (zero configuration)

## Quick Start

```python
# Minimal example - just 5 lines!
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")

asyncio.run(main())
```

**See `examples/` for complete usage:**
- **01_quickstart/**: hello_world.py (5 lines), simple_router.py
- **02_routing/**: basic_routing.py, hybrid_routing.py, with_constraints.py
- **03_optimization/**: caching.py, explicit_feedback.py

## Installation & Setup

### Prerequisites

- Python 3.10+ (3.13 recommended)
- **LLM API Keys** (at least one): OpenAI, Anthropic, Google, Groq, Mistral, Cohere, AWS Bedrock, HuggingFace
- Redis (optional - caching)
- PostgreSQL (optional - history persistence)

### Installation

```bash
git clone https://github.com/MisfitIdeas/conduit.git
cd conduit

python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

# Development tools
pip install -e ".[dev]"
```

### Configuration

Create `.env`:
```bash
# LLM Provider (at least one)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Optional
DATABASE_URL=postgresql://postgres:password@localhost:5432/conduit
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

Database setup:
```bash
./migrate.sh  # or: psql $DATABASE_URL < migrations/001_initial_schema.sql
```

## Tech Stack

- Python 3.10+, PydanticAI 1.14+, FastAPI
- PostgreSQL (history), Redis (caching)
- numpy, sentence-transformers (embeddings)

## Development

```bash
pytest --cov=conduit  # Tests
mypy conduit/         # Type checking
ruff check conduit/   # Linting
black conduit/        # Formatting
```

## Architecture

```
Query → Embedding → ML Routing Engine → LLM Provider → Response
   ↓                                                        ↓
   └─────────────────── Feedback Loop ─────────────────────┘
```

**Routing Process**:
1. Analyze query (embedding, features)
2. ML model predicts optimal route
3. Execute via PydanticAI (direct) or LiteLLM (100+ providers)
4. Collect feedback
5. Update routing model

**Bandit Algorithms** (`conduit.engines.bandits`):
- **Contextual Thompson Sampling**: Bayesian linear regression with multivariate normal posterior (NEW - Phase 3)
- **LinUCB**: Ridge regression with upper confidence bounds (contextual, optimal for LLM routing)
- **Thompson Sampling**: Bayesian probability matching with Beta distributions (default)
- **UCB1**: Upper Confidence Bound (optimistic exploration, fast convergence)
- **Epsilon-Greedy**: Simple exploration-exploitation (baseline)
- **Baselines**: Random, Oracle, AlwaysBest, AlwaysCheapest (for comparison)

All algorithms support:
- **Contextual features**: 387 dimensions (384 embedding + 3 metadata)
- **Multi-objective rewards**: Quality (70%) + Cost (20%) + Latency (10%)
- **Non-stationarity**: Sliding window adaptation (configurable window_size)

## Documentation

- **Examples**: See `examples/` for usage patterns and working code
- **Architecture**: See `docs/ARCHITECTURE.md` for system design
- **Bandit Algorithms**: See `docs/BANDIT_ALGORITHMS.md` for algorithm details
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md` for debugging common issues
- **LiteLLM Integration**: See `docs/LITELLM_INTEGRATION.md` for integration strategies
- **Development**: See `AGENTS.md` for development guidelines
- **Strategic Decisions**: See `notes/2025-11-18_business_panel_analysis.md`

## Status

**Version**: 0.0.5-alpha (Phase 3 Complete)
**Test Coverage**: 87% (64/73 bandit tests passing)

### Recent Additions
- ✅ Arbiter LLM-as-Judge (automatic quality evaluation)
- ✅ LiteLLM Feedback Loop (zero-config learning, Issue #13)
- ✅ Hybrid Routing (UCB1→LinUCB, 30% faster convergence)
- ✅ PCA Reduction (387→67 dims, 75% sample reduction)
- ✅ Multi-Objective Rewards (70% quality + 20% cost + 10% latency)
- ✅ Non-Stationarity Handling (sliding window adaptation)
- ✅ Contextual Thompson Sampling (Bayesian linear regression)
- ✅ Implicit Feedback (error/latency/retry detection)
- ✅ Dynamic Pricing (71+ models from llm-prices.com)

## License

MIT License - see LICENSE file for details
