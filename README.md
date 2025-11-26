# Conduit Router

[![CI](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml/badge.svg)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cut your LLM costs 30-50% without sacrificing quality.**

ML-powered routing system that learns which model to use for each type of query. Unlike static rule-based routers, Conduit continuously improves its decisions through feedback loops.

## Why Conduit?

| Feature | Conduit | Static Routers | Manual Selection |
|---------|---------|----------------|------------------|
| **Cost Optimization** | ✅ Learns cheapest model per query type | ❌ Fixed rules, no learning | ❌ Expensive models everywhere |
| **Quality Maintained** | ✅ 95%+ quality threshold | ⚠️ Depends on rules | ✅ High but wasteful |
| **Adapts to Changes** | ✅ Learns when models improve/degrade | ❌ Manual rule updates needed | ❌ Manual switching |
| **Setup Time** | ✅ 5 lines of code | ⚠️ Complex rule configuration | ✅ Simple but inefficient |
| **Typical Savings** | ✅ 30-50% cost reduction | ⚠️ 10-20% (if well-configured) | ❌ Baseline (most expensive) |

## Quick Start

```python
# Just 5 lines to start saving money
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")

asyncio.run(main())
```

**See results immediately**: Conduit starts optimizing from query 1, with full adaptive routing by query 50.

## How It Works

```
Query → Feature Analysis → ML Selection → LLM Provider → Response
   ↓                                                         ↓
   └──────────────── Continuous Learning ───────────────────┘
```

1. **Analyze**: Extract query features (complexity, domain, embeddings)
2. **Select**: ML algorithm picks optimal model (balances cost, quality, speed)
3. **Execute**: Route to selected model via PydanticAI or LiteLLM
4. **Learn**: Collect feedback and improve future decisions

## Key Features

- **ML-Driven Selection**: Contextual bandits (LinUCB, Thompson Sampling) learn from usage patterns
- **Multi-Objective Optimization**: Balance cost, quality, and latency based on your priorities
- **100+ LLM Providers**: Direct support for 8 providers, extended via LiteLLM integration
- **Smart Caching**: 10-40x faster on repeated queries (optional Redis)
- **User Preferences**: Control optimization per query (balanced, quality-first, cost-first, speed-first)
- **Zero Configuration**: Auto-detects available models from API keys

## Installation

### Quick Install

```bash
# Clone and install
git clone https://github.com/ashita-ai/conduit.git
cd conduit && source .venv/bin/activate || python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API keys (at least one required)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

# Run example
python examples/01_quickstart/hello_world.py
```

### Full Setup

**Prerequisites**:
- Python 3.10+ (3.13 recommended)
- At least one LLM API key (OpenAI, Anthropic, Google, Groq, Mistral, Cohere, AWS Bedrock, HuggingFace)
- Optional: Redis (caching), PostgreSQL (history)

**Configuration** (`.env`):
```bash
# LLM Providers (at least one)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Embedding (optional - defaults to free HuggingFace API)
EMBEDDING_PROVIDER=huggingface  # Options: huggingface, openai, cohere, sentence-transformers

# Optional
DATABASE_URL=postgresql://postgres:password@localhost:5432/conduit
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

**Database setup** (optional):
```bash
./migrate.sh  # or: psql $DATABASE_URL < migrations/001_initial_schema.sql
```

## User Preferences

Control optimization per query with 4 presets:

```python
from conduit.core import Query, UserPreferences

# Optimize for minimum cost
query = Query(
    text="Simple calculation",
    preferences=UserPreferences(optimize_for="cost")
)

# Presets:
# - balanced: Default (70% quality, 20% cost, 10% latency)
# - quality:  Maximize quality (80% quality, 10% cost, 10% latency)
# - cost:     Minimize cost (40% quality, 50% cost, 10% latency)
# - speed:    Minimize latency (40% quality, 10% cost, 50% latency)
```

Customize weights in `conduit.yaml` - see `examples/05_personalization/explicit_preferences.py`.

## Examples

```
examples/
├── 01_quickstart/       # hello_world.py (5 lines), simple_router.py
├── 02_routing/          # basic_routing.py, hybrid_routing.py, with_constraints.py
├── 03_optimization/     # caching.py, explicit_feedback.py
├── 04_litellm/          # LiteLLM integration (100+ providers)
└── 05_personalization/  # User preferences for optimization control
```

## Documentation

- **Architecture**: `docs/ARCHITECTURE.md` - System design and components
- **Algorithms**: `docs/BANDIT_ALGORITHMS.md` - ML algorithm details
- **Embeddings**: `docs/EMBEDDING_PROVIDERS.md` - Embedding configuration
- **Troubleshooting**: `docs/TROUBLESHOOTING.md` - Common issues and solutions
- **LiteLLM**: `docs/LITELLM_INTEGRATION.md` - Extended provider support
- **Development**: `AGENTS.md` - Development guidelines and contribution guide
- **Strategic Decisions**: `notes/2025-11-18_business_panel_analysis.md`

## Comparison with Alternatives

| Solution | Best For | Learning | Provider Support | Setup Complexity |
|----------|----------|----------|------------------|------------------|
| **Conduit** | Cost optimization with quality guarantees | ✅ Continuous ML | 100+ via LiteLLM | Low (5 lines) |
| **Martian** | Simple routing rules | ❌ Static rules | Limited | Low |
| **Portkey** | Enterprise features (observability) | ❌ Static rules | 100+ | Medium |
| **LiteLLM** | Provider abstraction only | ❌ No routing | 100+ | Low |
| **Manual** | Full control, low volume | ❌ No learning | Any | None |

**When to choose Conduit**:
- You want to reduce LLM costs without manual optimization
- You have queries that vary in complexity
- You want routing decisions to improve over time
- You need quality guarantees (not just cheapest model)

**When to choose alternatives**:
- You need enterprise features like teams, SSO, compliance (Portkey)
- You just need provider abstraction without routing (LiteLLM)
- You have fixed routing rules that don't need learning (Martian)

## Tech Stack

- **Core**: Python 3.10+, PydanticAI 1.14+, FastAPI
- **ML**: NumPy 2.0+ (LinUCB matrices), contextual bandits
- **Storage**: PostgreSQL (history), Redis (caching)
- **Embeddings**: HuggingFace API (free default), OpenAI, Cohere, sentence-transformers

## Status

**Version**: 0.1.0 (Pre-1.0)
**Test Status**: 100% passing (565/565 tests), 81% coverage
**CI/CD**: GitHub Actions with automated testing

### Recent Additions
- ✅ Lightweight API-based embeddings (HuggingFace default, no heavy dependencies)
- ✅ Arbiter LLM-as-Judge (automatic quality evaluation)
- ✅ LiteLLM feedback loop (zero-config learning)
- ✅ Hybrid routing (30% faster convergence)
- ✅ Multi-objective rewards (quality + cost + latency)
- ✅ User preferences (per-query optimization control)
- ✅ Dynamic pricing (71+ models auto-updated)

## Frequently Asked Questions

### How much can I actually save?

**Typical savings: 30-50%** based on routing expensive queries (GPT-4, Claude Opus) to cheaper models (GPT-4o-mini, Claude Haiku) when appropriate. Actual savings depend on your query distribution.

**Example**: If 60% of your queries are simple (routed to cheap models) and 40% are complex (routed to expensive models), you save ~45% compared to using expensive models everywhere.

### How does Conduit maintain quality?

Conduit learns a **quality threshold per query type** through continuous feedback. It won't route to cheaper models if quality drops below your threshold (default: 95% of expensive model quality).

The system uses:
- Explicit feedback (user ratings, task success)
- Implicit feedback (errors, retries, latency) weighted at 30%
- Composite rewards balancing quality (70%), cost (20%), latency (10%)

### How long until I see savings?

- **Immediate**: Conduit starts routing from query 1 using UCB1 (simple, fast)
- **50 queries**: Switches to LinUCB (contextual, better decisions)
- **200 queries**: Fully adapted to your query distribution
- **Continuous**: Adapts to model updates, pricing changes, new models

### What if I don't have Redis or PostgreSQL?

Conduit works without them - they're optional optimizations:
- **No Redis**: Slower embeddings (200ms vs 5ms), but routing still works
- **No PostgreSQL**: No history persistence, but in-memory routing works fine

For production, Redis highly recommended (10-40x faster).

### Can I use my own quality evaluation?

Yes. Provide explicit feedback:

```python
from conduit.engines.bandits.base import BanditFeedback

feedback = BanditFeedback(
    model_id="gpt-4o-mini",
    cost=response.cost,
    quality_score=0.95,  # Your evaluation (0.0-1.0)
    latency=response.latency
)
await router.hybrid_router.give_feedback(feedback, features)
```

Or use the built-in Arbiter (LLM-as-judge) for automatic evaluation.

### Is this production-ready?

**Pre-1.0 status**: Core routing is stable and tested (565/565 tests passing), but:
- APIs may change before 1.0
- Some features are experimental (PCA reduction, Arbiter evaluation)
- Production use recommended with monitoring and fallbacks

### How does this compare to prompt optimization?

Complementary approaches:
- **Prompt optimization**: Get better results from the same model
- **Conduit**: Route queries to the right model for the task

Use both. Conduit's savings are independent of prompt quality.

### What about streaming responses?

Not currently supported. Conduit focuses on request/response routing. Streaming support planned for post-1.0.

## Development

```bash
# Tests
pytest --cov=conduit  # Run with coverage (must be >80%)

# Code quality (must pass before commit)
mypy conduit/         # Type checking
ruff check conduit/   # Linting
black conduit/        # Formatting

# Install git hooks (runs tests before push)
bash scripts/install-hooks.sh
```

## Security

**Automated Dependency Scanning**: GitHub Dependabot monitors dependencies weekly for security vulnerabilities. See `.github/dependabot.yml` and repository Security tab.

**Never commit credentials**: Use environment variables in `.env` (gitignored). See `AGENTS.md` for security practices.

## License

MIT License - see LICENSE file
