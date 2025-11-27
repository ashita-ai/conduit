# Conduit

[![CI](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml/badge.svg)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://github.com/ashita-ai/conduit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Smart LLM routing that learns to cut costs without sacrificing quality.**

Smart routing that learns which LLM to use for each type of query. Like a chess engine that improves with every game, Conduit gets better at picking the right model as it sees more queries. Unlike static routers with fixed rules, Conduit adapts to your workload automatically.

## The Problem

You have 5+ LLM options (GPT-4, Claude Opus, Gemini, etc.) with 100x price differences. Most teams either:
- **Overspend**: Route everything to expensive models "to be safe"
- **Underspend**: Route everything to cheap models and get poor quality
- **Manual rules**: Write brittle if/else logic that breaks when models update

There's a better way.

## Why Conduit?

| Feature | Conduit | Static Routers | Manual Selection |
|---------|---------|----------------|------------------|
| **Cost Optimization** | ✅ Learns cheapest model per query type | ❌ Fixed rules, no learning | ❌ Expensive models everywhere |
| **Quality** | ✅ Balances cost vs quality via feedback | ⚠️ Depends on rules | ✅ High but wasteful |
| **Adapts to Changes** | ✅ Learns when models improve/degrade | ❌ Manual rule updates needed | ❌ Manual switching |
| **Setup Time** | ✅ 5 lines of code | ⚠️ Complex rule configuration | ✅ Simple but inefficient |
| **Learning** | ✅ Continuous adaptation | ❌ No learning | ❌ No learning |

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

**Starts learning immediately**: Conduit begins routing from query 1, with full contextual learning after 2,000 queries.

## How It Works

Think of Conduit as a recommendation system for LLMs. It learns from every query which models work best for which tasks.

```
Query → Analyze → Smart Selection → LLM Provider → Response
   ↓                                                    ↓
   └──────────── Learn & Improve ─────────────────────┘
```

1. **Analyze**: Understand query complexity, domain, and context
2. **Smart Selection**: Pick optimal model balancing cost, quality, and speed
3. **Execute**: Route to selected model via PydanticAI or LiteLLM
4. **Learn**: Track what worked and improve future routing decisions

**Under the hood**: Uses contextual bandits (LinUCB, Thompson Sampling) for multi-armed optimization. If you don't know what that means, don't worry - it just works.

## Key Features

- **Learning System**: Gets smarter with every query, learns your workload patterns automatically
- **Multi-Objective**: Balance cost, quality, and speed based on what matters to you
- **100+ LLM Providers**: Direct support for 8 providers, extended via LiteLLM integration
- **Smart Caching**: Faster embedding lookups on repeated queries (optional Redis)
- **Per-Query Control**: Override optimization per query (quality-first, cost-first, speed-first)
- **Zero Config**: Auto-detects available models from your API keys

## Installation

### Quick Install

```bash
# Clone and install
git clone https://github.com/ashita-ai/conduit.git
cd conduit && source .venv/bin/activate || python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API keys (at least one LLM provider required)
export OPENAI_API_KEY=sk-...      # For LLMs AND embeddings (recommended)
export ANTHROPIC_API_KEY=sk-...   # Alternative LLM provider

# Optional: Install local embeddings (if not using OpenAI embeddings)
# pip install fastembed  # Lightweight option (~100MB)
# OR
# pip install sentence-transformers  # Full PyTorch (~2GB)

# Run example
python examples/01_quickstart/hello_world.py
```

**Note**: If you have `OPENAI_API_KEY` set, embeddings work automatically (uses same key for both LLMs and embeddings). No additional setup needed.

### Full Setup

**Prerequisites**:
- Python 3.10+ (3.13 recommended)
- At least one LLM API key (OpenAI, Anthropic, Google, Groq, Mistral, Cohere, AWS Bedrock, HuggingFace)
- **Embedding provider** (choose one):
  - **OpenAI API key** (recommended - reuses your LLM key, no additional setup)
  - **Cohere API key** (alternative API option)
  - **Local embeddings** (`pip install fastembed` or `pip install sentence-transformers`)
- Optional: Redis (caching), PostgreSQL (history)

**Configuration** (`.env`):
```bash
# LLM Providers (at least one required)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Embeddings - Auto-detects in this order:
# 1. OpenAI (if OPENAI_API_KEY set) - RECOMMENDED, reuses LLM key
# 2. Cohere (if COHERE_API_KEY set)
# 3. FastEmbed (if installed via: pip install fastembed)
# 4. sentence-transformers (if installed via: pip install sentence-transformers)
EMBEDDING_PROVIDER=auto  # or: openai, cohere, fastembed, sentence-transformers

# If using Cohere for embeddings (separate from LLM provider)
# COHERE_API_KEY=your_cohere_key

# Optional
DATABASE_URL=postgresql://postgres:password@localhost:5432/conduit
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

**Embedding Provider Notes**:
- **Recommended**: Use OpenAI embeddings (no extra setup if you already have `OPENAI_API_KEY`)
- **No API key options**: Install `fastembed` (~100MB) or `sentence-transformers` (~2GB)
- **PCA models**: Automatically generated per provider on first use (~5 seconds), then cached

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

## Comparison with Routing Alternatives

| Solution | Best For | Learning | Provider Support | Setup Complexity |
|----------|----------|----------|------------------|------------------|
| **Conduit** | ML-based cost optimization | ✅ Continuous ML | 100+ via LiteLLM/PydanticAI | Low (5 lines) |
| **RouteLLM** | Research/custom ML routing | ✅ Requires training data | Custom integration | High |
| **Martian** | Simple static rules | ❌ Fixed rules | Limited | Low |
| **Manual if/else** | Full control, low volume | ❌ No learning | Any | None |

**Not Competitors, We Integrate**:
- **LiteLLM**: Provider abstraction layer. Conduit uses LiteLLM to access 100+ providers, adds ML routing on top.
- **Portkey**: Enterprise gateway (teams, SSO, observability). Could run Conduit behind Portkey for ML routing + enterprise features.
- **LangChain**: LLM orchestration framework. Conduit integrates as a routing component (see `examples/06_integrations/`).

Think of it this way: LiteLLM/Portkey are the roads, Conduit is the GPS that picks the best route.

**When Conduit shines**:
- You're spending $500+/month on LLM APIs (enough volume to benefit)
- You have 1000+ queries/day with varying complexity
- You use 3+ different models across your workload
- You want automatic optimization without writing routing rules

**When NOT to use Conduit** (be honest with yourself):
- **Single model**: If you only use one model, you don't need routing
- **Low volume**: <100 queries/day? Simple round-robin is fine
- **Fixed patterns**: If 100% of queries need the same model, no router helps
- **Need streaming**: Conduit doesn't support streaming responses yet (post-1.0)
- **Enterprise compliance**: Need SOC2, teams, SSO? Use Portkey instead

**When to choose routing alternatives**:
- You have research/custom ML models and training data (→ RouteLLM)
- You have simple static rules that never change (→ Martian)
- You want to write all routing logic yourself (→ Manual if/else)

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

**Savings depend on your query mix**. If you route expensive queries (GPT-4, Claude Opus) to cheaper models (GPT-4o-mini, Claude Haiku) when quality allows, you can reduce costs significantly. Actual savings depend on your query distribution and quality requirements.

**Example**: If 60% of your queries can use cheap models without quality loss and 40% require expensive models, you'll save compared to using expensive models everywhere. Exact percentages require benchmarking on your workload.

### How does Conduit balance cost and quality?

Conduit learns which models work best for different query types through continuous feedback. The system optimizes a multi-objective reward balancing quality, cost, and latency.

**Feedback system**:
- Explicit feedback (user ratings, task success)
- Implicit feedback (errors, retries, latency) weighted at 30%
- Composite rewards balancing quality (70%), cost (20%), latency (10%) by default

You can adjust these weights per query using UserPreferences to prioritize quality over cost or vice versa.

### How long until I see savings?

- **Immediate**: Conduit starts routing from query 1 using UCB1 (non-contextual, fast exploration)
- **2,000 queries**: Switches to LinUCB (contextual, query-aware decisions)
- **Continuous**: Adapts to model updates, pricing changes, new models as they're added

### What if I don't have Redis or PostgreSQL?

Conduit works without them - they're optional optimizations:
- **No Redis**: Repeated queries require re-computing embeddings (slower), but routing still works
- **No PostgreSQL**: No history persistence, but in-memory routing works fine

For production, Redis recommended for embedding caching.

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
