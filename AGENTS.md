# AGENTS.md - Project Context

**Purpose**: Quick reference for working on Conduit
**Last Updated**: 2025-11-20
**Status**: Phase 2 complete - Implicit feedback system + examples shipped

---

## Quick Orientation

**Conduit**: ML-powered LLM routing system for cost/latency/quality optimization
**Stack**: Python 3.10+, PydanticAI, FastAPI, Contextual Bandit (Thompson Sampling)
**Goal**: Learn optimal model selection from usage patterns vs static rules

**⚠️ ALPHA SOFTWARE**: Breaking changes happen. No backwards compatibility guarantees. Move fast, fix things.

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

### Current Test Coverage (2025-11-20)
- **Overall**: 87% coverage ✅ (exceeds 80% Phase 1 target)
- **Core Engine**: 96-100% (models, analyzer, bandit, router, executor)
- **Feedback System**: 98-100% (signals, history, integration, detector - 76 comprehensive tests)
- **Bandit Algorithms**: 52/61 tests passing (85% pass rate, up from 52%)
  - Epsilon-Greedy: 14/14 passing (100%) ✅
  - Thompson Sampling: 6/7 passing (86%)
  - UCB1: 6/11 passing (55%)
  - Baselines: 15/18 passing (83%)
- **Database**: 84% (integration tests complete, edge cases covered)
- **CLI**: 98% coverage ✅ (20 comprehensive tests)
- **API Layer**: 0% (routes, middleware - not yet tested)
- **Status**: Phase 2 complete + test suite significantly improved

### Phase 2 Completion (2025-11-19)
**Completed Features**: ✅ All implicit feedback components shipped

1. ✅ **Implicit Feedback System** - "Observability Trinity" (Errors + Latency + Retries)
   - QueryHistoryTracker with Redis (5-min TTL, cosine similarity)
   - ImplicitFeedbackDetector (orchestrates all signal detection)
   - FeedbackIntegrator (weighted rewards: 70% explicit, 30% implicit)
   - SignalDetector (error patterns, latency tolerance, retry detection)
   - 76 comprehensive tests (100% coverage for core components)

2. ✅ **Redis Caching** - 10-40x performance improvement
   - QueryFeatures caching with circuit breaker
   - Cache hit/miss statistics
   - Graceful degradation without Redis

3. ✅ **Examples Suite** - Progressive learning path (8 examples)
   - 01_quickstart/: hello_world.py (5 lines), simple_router.py, model_discovery.py
   - 02_routing/: basic_routing.py, with_constraints.py
   - 03_optimization/: caching.py, explicit_feedback.py, implicit_feedback.py, combined_feedback.py
   - 04_production/: (planned - fastapi, batch, monitoring)

4. ✅ **Database Migration** - Schema for implicit feedback storage

### Phase 3 Priorities (Next)
**Prerequisites**: ✅ Implicit feedback system complete, examples validated

1. Document success metrics and quality baselines
2. Create demo showing 30% cost reduction on real workload
3. Production API examples (FastAPI endpoint, batch processing)
4. Monitoring and observability tooling

### Recent Updates (2025-11-20)

**Session 1: Test Suite Improvements**
- ✅ **Test Suite Improvements**: Added 33 new tests (13 detector + 20 CLI)
- ✅ **Detector Tests**: 100% coverage with normalized embedding fixes
- ✅ **CLI Tests**: 98% coverage testing all commands (serve, run, demo, version)
- ✅ **Bandit Test Fixes**: Improved from 52% to 85% pass rate
  - Fixed arm_pulls counting logic across all algorithms
  - Fixed field name mismatches (counts→arm_pulls, values→mean_reward)
  - Added Pydantic validation for BanditFeedback
  - Epsilon-Greedy now 100% passing with reproducible random seed
- ✅ **Overall Coverage**: 87% (up from ~52%), exceeds Phase 1 target

**Session 2: Dynamic Pricing & Model Discovery**
- ✅ **Dynamic Pricing**: Fetch from llm-prices.com at runtime (24h cache, graceful fallback)
- ✅ **Model Discovery API**: `supported_models()`, `available_models()`, `get_available_providers()`
- ✅ **Auto-Detection**: Automatically detects which models YOU can use based on API keys in .env
- ✅ **Provider Filtering**: Only expose 71 models where PydanticAI support AND pricing exist
- ✅ **Model Name Fixes**: Fixed Anthropic names (claude-opus-4 → claude-3-opus-20240229)
- ✅ **Documentation**: Created comprehensive MODEL_DISCOVERY.md
- ✅ **Examples**: Created model_discovery.py example (8th example)

**Previous Session (2025-11-19)**:
- ✅ Implemented complete implicit feedback system (QueryHistoryTracker, Detector, Integrator)
- ✅ Added 76 comprehensive tests for feedback components (98-100% coverage)
- ✅ Created Redis caching with circuit breaker pattern
- ✅ Reorganized examples into 4-folder progressive structure
- ✅ Created combined_feedback.py showing explicit + implicit together
- ✅ All examples tested and working (graceful Redis degradation)

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
  - **API Change**: Use `Agent(...).run(..., deps=...)` instead of `Agent(..., deps=...)`
- **Pydantic**: 2.12+ (data validation)
- **FastAPI**: 0.115+ (REST API)
- **PostgreSQL**: Database for routing history (via Supabase)
- **Redis**: Caching and rate limiting [Phase 2]

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
   - **Explicit**: User ratings (quality_score, user_rating, met_expectations)
   - **Implicit**: System signals (errors, latency, retries) - weighted 30%
   - **Integration**: FeedbackIntegrator combines both (70% explicit, 30% implicit)

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
- **Thompson Sampling**: Bayesian approach to model selection (Beta distributions)
- **Query Embedding**: Semantic representation for routing features (sentence-transformers)
- **Dual Feedback Loop**: Explicit (user ratings) + Implicit (system signals)
- **Observability Trinity**: Error detection + Latency tracking + Retry detection
- **Weighted Feedback**: 70% explicit (user ratings) + 30% implicit (behavioral signals)
- **Data Moat**: Learning algorithm improves with usage, creating competitive barrier
- **Probabilistic Guarantees**: 95%+ quality confidence, not 100% promises
- **Graceful Degradation**: Core routing works without Redis (caching/retry disabled)
- **Dynamic Pricing**: Auto-fetch 71+ models from llm-prices.com (24h cache, fallback)
- **Model Discovery**: Auto-detects available models based on API keys in .env
- **Provider Filtering**: Only expose models where PydanticAI support AND pricing exist

### Make Targets (To Be Created)
```bash
make test          # Run tests with coverage
make type-check    # Run mypy
make lint          # Run ruff
make format        # Run black
make all           # All quality checks
```

---

## Working with AI Agents

### Task Management
**TodoWrite enforcement (MANDATORY)**: For ANY task with 3+ distinct steps, use TodoWrite to track progress - even if the user doesn't request it explicitly. This ensures nothing gets forgotten and provides visibility into progress for everyone working on the project.

**Plan before executing**: For complex tasks, create a plan first. Understand requirements, identify dependencies, then execute systematically.

### Output Quality
**Full data display**: Show complete data structures, not summaries or truncations. Examples should display real, useful output (not "[truncated]" or "...").

**Debugging context**: When showing debug output, include enough detail to actually debug - full prompts, complete responses, actual data structures. Truncating output defeats the purpose.

**Verify usefulness**: Before showing output, verify it's actually helpful for the user's goal. Test that examples demonstrate real functionality, not abstractions.

### Audience & Context Recognition
**Auto-detect technical audiences**: Code examples, technical docs, developer presentations → eliminate ALL marketing language automatically. Engineering contexts get technical tone (no superlatives like "blazingly fast", "magnificent", "revolutionary").

**Recognize audience immediately**: Engineers get technical tone, no marketing language. Business audiences get value/ROI focus. Academic audiences get methodology and rigor. Adapt tone and content immediately based on context.

**Separate material types**: Code examples stay clean (no narratives or marketing). Presentation materials (openers, talking points) live in separate files. Documentation explains architecture and usage patterns.

### Quality & Testing
**Test output quality, not just functionality**: Run code AND verify the output is actually useful. Truncated or abstracted output defeats the purpose of examples. Show real data structures, not summaries.

**Verify before committing**: Run tests and verify examples work before showing output. Test both functionality and usefulness.

**Connect work to strategy**: Explicitly reference project milestones, coverage targets, and strategic priorities when completing work. Celebrate milestones when achieved.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix → commit. Don't perfect upfront. Progressive refinement beats upfront perfection.

**Proactive problem solving**: Use tools like Glob to check file existence before execution. Anticipate common issues and handle them gracefully.

**Parallel execution**: Batch independent operations (multiple reads, parallel test execution) to improve efficiency.

### Communication & Feedback
**Direct feedback enables fast iteration**: Clear, immediate feedback on what's wrong enables rapid course correction. Specific, actionable requests work better than vague suggestions.

**Match user communication style**: Some users prefer speed over process formality, results over explanations. Adapt communication style accordingly while maintaining quality standards.

### Git & Commit Hygiene
**Commit hygiene**: Each meaningful change gets its own commit with clear message (what + why). This makes progress tracking and rollback easier.

**Clean git workflow**: Always check `git status` and `git branch` before operations. Use feature branches for all changes.

---

**Last Updated**: 2025-11-20
