# Conduit

ML-powered LLM routing system that learns optimal model selection for cost, latency, and quality optimization.

## Overview

Conduit uses contextual bandits (Thompson Sampling) to intelligently route queries to the optimal LLM model based on learned patterns from usage data. Unlike static rule-based routers, Conduit continuously improves routing decisions through feedback loops.

## Key Features

- **ML-Driven Routing**: Learns from usage patterns vs static IF/ELSE rules
- **Multi-Objective Optimization**: Balance cost, latency, and quality constraints
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq via PydanticAI
- **Dual Feedback Loop**: Explicit (user ratings) + Implicit (errors, latency, retries)
- **Redis Caching**: 10-40x performance improvement on repeated queries
- **Graceful Degradation**: Core routing works without Redis
- **Thompson Sampling**: Bayesian bandit algorithm for exploration/exploitation

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
- **02_routing/**: basic_routing.py, with_constraints.py
- **03_optimization/**: caching.py, explicit_feedback.py, implicit_feedback.py, combined_feedback.py
- **04_production/**: (coming soon - FastAPI, batch processing, monitoring)

## Installation & Setup

### Prerequisites

- Python 3.10+ (3.13 recommended)
- LLM API keys (OpenAI, Anthropic, Google, or Groq) - at least one required
- Redis instance (optional - for caching and retry detection)
- Supabase account (optional - for query history persistence)

### Step 1: Clone and Install

```bash
git clone https://github.com/MisfitIdeas/conduit.git
cd conduit

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install development tools
pip install mypy black ruff pytest pytest-asyncio pytest-cov psycopg2-binary
```

**Note:** Some ML dependencies (scipy, scikit-learn) require Fortran compilers. If installation fails, you can still use the core functionality.

### Step 2: Environment Configuration

Create `.env` file:
```bash
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here

# Supabase (required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
DATABASE_URL=postgresql://postgres.your-project:[password]@aws-0-region.pooler.supabase.com:6543/postgres

# Redis (optional - Phase 2+)
REDIS_URL=redis://localhost:6379

# Application
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Step 3: Database Setup

**Option A: Using Alembic (recommended)**
```bash
# Ensure DATABASE_URL in .env points to Supabase pooler connection
./migrate.sh
```

**Option B: Manual SQL (if pooler not available)**
```bash
# Copy SQL content and paste into Supabase SQL Editor
cat migrations/001_initial_schema.sql
```

See `migrations/DEPLOYMENT.md` for detailed migration instructions.

## Tech Stack

- **Python 3.10+**
- **PydanticAI 1.14+** (unified LLM interface)
- **FastAPI** (REST API)
- **PostgreSQL** (routing history)
- **scikit-learn** (ML algorithms)
- **sentence-transformers** (query embeddings)

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=conduit

# Type checking
mypy conduit/

# Linting
ruff check conduit/

# Formatting
black conduit/
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
3. Execute via PydanticAI
4. Collect feedback
5. Update routing model

## Documentation

- **Examples**: See `examples/` for usage patterns and working code
- **Architecture**: See `docs/ARCHITECTURE.md` for system design
- **Development**: See `AGENTS.md` for development guidelines
- **Strategic Decisions**: See `notes/2025-11-18_business_panel_analysis.md`

## Current Status

**Phase**: 2 Complete (Implicit Feedback + Examples)
**Version**: 0.0.2-alpha
**Last Updated**: 2025-11-19

### Phase 2 Completed (2025-11-19)
- ✅ **Implicit Feedback System** - "Observability Trinity"
  - Error detection (model failures, empty responses, error patterns)
  - Latency tracking (user patience tolerance categorization)
  - Retry detection (semantic similarity with 5-min history window)
  - 76 comprehensive tests (98-100% coverage)

- ✅ **Redis Caching** - 10-40x performance improvement
  - QueryFeatures caching with circuit breaker
  - Cache hit/miss statistics
  - Graceful degradation without Redis

- ✅ **Examples Suite** - Progressive learning path
  - 7 examples from hello_world.py (5 lines) to combined_feedback.py
  - Organized into 4 folders: quickstart, routing, optimization, production
  - All tested and working with graceful Redis degradation

- ✅ **Database Migration** - Schema for implicit feedback storage

### Phase 1 Completed (2025-11-18)
- ✅ Core routing engine (ML-powered model selection)
- ✅ Query analysis (embeddings, complexity, domain classification)
- ✅ Thompson Sampling bandit algorithm
- ✅ Database schema (PostgreSQL/Supabase)
- ✅ Type safety (mypy strict mode passes)
- ✅ Comprehensive tests (87% overall coverage)

### Phase 3 Priorities (Next)
- ⏳ Document success metrics and quality baselines
- ⏳ Create demo showing 30% cost reduction on real workload
- ⏳ Production API examples (FastAPI endpoint, batch processing)
- ⏳ Monitoring and observability tooling
- ⏳ API layer testing (currently 0% coverage)

### Test Coverage (2025-11-19)
- **Overall**: 87% ✅ (exceeds 80% target)
- **Core Engine**: 96-100% (models, analyzer, bandit, router, executor)
- **Feedback System**: 98-100% (signals, history, integration - 76 tests)
- **Database**: 84% (integration tests complete)
- **API Layer**: 0% (untested - Phase 3 priority)
- **CLI**: 0% (untested)

## License

MIT License - see LICENSE file for details
