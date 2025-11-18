# AGENTS.md - Project Context

**Purpose**: Quick reference for working on Conduit
**Last Updated**: 2025-11-18
**Status**: Phase 1 complete, strategic direction defined

---

## Quick Orientation

**Conduit**: ML-powered LLM routing system for cost/latency/quality optimization
**Stack**: Python 3.10+, PydanticAI, FastAPI, Contextual Bandit (Thompson Sampling)
**Goal**: Learn optimal model selection from usage patterns vs static rules

**Positioning**: Intelligent LLM infrastructure (not cost tool)
**Value Prop**: Learn which LLM to use for YOUR workload, reducing costs 30-50% while maintaining quality
**Competitive Moat**: Self-improving Thompson Sampling algorithm creates data network effect

### Directory Structure

```
conduit/
├── conduit/
│   ├── core/               # Models, config, exceptions
│   ├── engines/            # Routing engine, ML models
│   ├── resilience/         # Circuit breaker, retry logic
│   ├── cli/                # Command-line interface
│   └── utils/              # Helper functions
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
├── examples/               # Usage examples
├── docs/                   # Design documentation
└── notes/                  # Strategic decisions & analysis (dated)
```

---

## Strategic Decisions (2025-11-18)

**Full analysis**: See `notes/2025-11-18_business_panel_analysis.md`

### Core Principles
1. **Positioning**: Infrastructure (essential) not tool (optional)
2. **Quality Guarantees**: Probabilistic (95%+) not deterministic (100%)
3. **Feedback Design**: Dual system (explicit + implicit signals)
4. **Pricing Model**: Usage-based (per-call) not SaaS (per-seat)

### Success Metrics (To Be Documented)
- **Technical**: Model parameters converge within 1,000 queries
- **Customer**: >30% cost savings within first month
- **Quality**: 95% of queries meet or exceed baseline
- **System**: p99 latency < 200ms for routing decisions

### Phase 2 Priorities
1. Implement implicit feedback system (retry behavior, latency, errors)
2. Add query result caching (Redis) and batch processing
3. Document success metrics and quality baselines
4. Create demo showing 30% cost reduction on real workload

### Communication Guidelines
**Say**: "Saves 30-50% costs", "95% quality guarantee", "Gets smarter with use"
**Avoid**: "Thompson Sampling", "contextual bandits", "100% optimal"

---

## Critical Rules

### 1. PydanticAI for LLM Integration
All LLM calls use PydanticAI for provider-agnostic access and structured outputs.

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class RoutingDecision(BaseModel):
    model: str
    confidence: float
    reasoning: str

router = Agent(
    model="openai:gpt-4o-mini",
    result_type=RoutingDecision
)
```

### 2. Provider-Agnostic Design
Must work with ANY LLM provider (OpenAI, Anthropic, Google, Groq).

### 3. Type Safety (Strict Mypy)
All functions require type hints, no `Any` without justification.

### 4. No Placeholders/TODOs
Production-grade code only. Complete implementations or nothing.

### 5. Complete Features Only
If you start, you finish:
- Implementation complete
- Tests (>80% coverage)
- Docstrings
- Example code
- Exported in `__init__.py`

---

## Development Workflow

### Before Starting
1. Check `git status` and `git branch`
2. Create feature branch: `git checkout -b feature/my-feature`

### During Development
1. Write tests as you code (not after)
2. Run `pytest` frequently
3. Follow type hints strictly

### Before Committing
```bash
pytest --cov=conduit     # Tests pass with coverage
mypy conduit/            # Type checking clean
ruff check conduit/      # Linting clean
black conduit/           # Formatting applied
```

---

## Tech Stack

### Core Dependencies
- **Python**: 3.10+ (modern type hints, async/await)
- **PydanticAI**: 1.14+ (unified LLM interface with structured outputs)
- **Pydantic**: 2.12+ (data validation)
- **FastAPI**: 0.115+ (REST API)
- **PostgreSQL**: Database for routing history
- **Redis**: Caching and rate limiting

### ML Stack
- **scikit-learn**: Contextual bandit implementation
- **sentence-transformers**: Query embeddings
- **numpy**: Numerical operations

### Development Tools
- **pytest**: 9.0+ (testing)
- **pytest-asyncio**: Async test support
- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking (strict mode)

---

## Key Patterns

### Routing Decision Flow
1. **Query Analysis**: Embed query, extract features
2. **ML Selection**: Contextual bandit predicts optimal model
3. **Execution**: Call selected model via PydanticAI
4. **Feedback Loop**: Update ML model with results
   - **Explicit**: User ratings (quality_score, met_expectations)
   - **Implicit**: System signals (retry, latency, errors) [Phase 2]

### PydanticAI Integration Benefits
- Unified interface across all providers (no provider-specific code)
- Automatic interaction tracking (cost, latency, tokens)
- Type-safe structured outputs
- Built-in retry logic and error handling

---

## Code Quality Standards

### Docstrings
```python
async def route_query(query: str) -> RoutingResult:
    """Route query to optimal LLM model based on learned patterns.

    Args:
        query: User query to route

    Returns:
        RoutingResult with model selection and metadata

    Raises:
        RoutingError: If routing fails

    Example:
        >>> result = await route_query("What is 2+2?")
        >>> print(result.model)
        "gpt-4o-mini"
    """
```

### Formatting
- **black**: Line length 88
- **ruff**: Follow pyproject.toml config
- **mypy**: Strict mode

---

## Quick Reference

### Key Concepts
- **Contextual Bandit**: ML algorithm for exploration/exploitation
- **Thompson Sampling**: Bayesian approach to model selection
- **Query Embedding**: Semantic representation for routing features
- **Dual Feedback Loop**: Explicit (user ratings) + Implicit (system signals)
- **Data Moat**: Learning algorithm improves with usage, creating competitive barrier
- **Probabilistic Guarantees**: 95%+ quality confidence, not 100% promises

### Make Targets (To Be Created)
```bash
make test          # Run tests with coverage
make type-check    # Run mypy
make lint          # Run ruff
make format        # Run black
make all           # All quality checks
```

---

**Last Updated**: 2025-11-18
