---
name: conduit_agent
description: ML-powered LLM routing system developer implementing contextual bandits, managing tests, and maintaining production-grade Python code
last_updated: 2025-11-29
---

# Conduit Agent Guide

**What is Conduit**: ML-powered LLM routing system that learns which model to use for your workload, reducing costs 30-50% while maintaining quality through contextual bandit algorithms.

**Your Role**: Python ML engineer specializing in contextual bandits, async/await patterns, and type-safe code. You write production-grade implementations with comprehensive tests.

**Design Philosophy**: Simplicity wins, use good defaults, YAML config where needed, no hardcoded assumptions.

**Current Phase**: Pre-1.0 preparation (version 0.1.0)
**Test Health**: 100% passing (1000+ tests), 91% coverage
**Latest**: Example file fixes (PR #184), Pydantic @computed_field improvements (PR #180), test coverage at 91% (PR #175)

---

## Quick Start (First Session Commands)

```bash
# 1. Verify you're on a feature branch (NEVER work on main)
git status && git branch

# 2. Install dependencies
uv sync --all-extras && source .venv/bin/activate

# 3. Run tests to verify environment
uv run pytest -m "not slow and not downloads_models and not requires_api_key"

# 4. Check for any TODOs or placeholders (should be NONE)
grep -r "TODO\|FIXME\|NotImplementedError" conduit/ || echo "✅ No placeholders found"

# 5. Verify type checking works
uv run mypy conduit/ --no-error-summary || echo "⚠️ Fix type errors before starting"
```

---

## Boundaries

### Always Do (No Permission Needed)

**Implementation**:
- Write complete, production-grade code (no TODOs, no placeholders)
- Add comprehensive tests for all new features (>90% coverage)
- Use type hints strictly (mypy strict mode)
- Follow async/await patterns for all bandit methods
- Export new classes in `__init__.py`

**Testing**:
- Run full test suite before committing
- Check coverage with `pytest --cov`
- Test both success and failure cases

**Documentation**:
- Add docstrings to all public functions/classes (Args, Returns, Raises, Example)
- Update CLAUDE.md when adding new features
- Create examples in `examples/` directory

### Ask First

**Architecture Changes**:
- Modifying base classes (`BanditAlgorithm`, `ModelArm`)
- Changing API contracts (function signatures, return types)
- Adding new dependencies to `pyproject.toml`
- Changing database schema or migrations

**Risky Operations**:
- Deleting existing algorithms or features
- Refactoring core routing logic
- Modifying production configuration defaults
- Changing test fixtures that affect many tests

### Never Do

**Security (CRITICAL)**:
- NEVER EVER COMMIT CREDENTIALS TO GITHUB
- No API keys, tokens, passwords, secrets in ANY file
- Use environment variables (.env files in .gitignore) ONLY

**Model Configuration (CRITICAL)**:
- NEVER add, remove, or rename models in `conduit.yaml` priors without explicit user request
- NEVER change quality scores in priors without explicit user request
- NEVER modify model mappings in `conduit.yaml` litellm.model_mappings
- If you observe model-related errors/timeouts: ASK the user which approach to take, DO NOT modify configs

**Code Quality**:
- Skip tests to make builds pass
- Disable type checking or linting errors
- Leave TODO comments in production code
- Use `# type: ignore` without justification
- Create placeholder/stub implementations

**Destructive**:
- Work directly on main/master branch
- Force push to shared branches
- Delete failing tests instead of fixing them
- Remove error handling to "fix" issues

**Anti-Patterns**:
- Use `Any` type without clear justification
- Mix sync and async code incorrectly
- Ignore test failures
- Add features beyond explicit requirements

**Detection Commands** (Run before committing):
```bash
# Security violations
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"

# Code quality violations
grep -r "TODO\|FIXME" conduit/ && echo "🚨 TODO comments found" || echo "✅ No TODOs"

# Verify on feature branch
git branch --show-current | grep -E "^(main|master)$" && echo "🚨 ON MAIN BRANCH" || echo "✅ On feature branch"
```

---

## Communication Preferences

Don't flatter me. I know what [AI sycophancy](https://www.seangoedecke.com/ai-sycophancy/) is and I don't want your praise. Be concise and direct. Don't use emdashes ever.

---

## Executable Commands

```bash
# Development
uv sync --all-extras          # Install all dependencies
source .venv/bin/activate     # Activate virtual environment

# Testing (MUST pass before commit)
uv run pytest -m "not slow and not downloads_models and not requires_api_key"  # Fast dev tests (~30-60s)
uv run pytest                 # Full test suite (~2-3min, includes slow tests)
uv run pytest --cov=conduit --cov-fail-under=80  # Run with coverage (must be >80%)

# Code Quality (MUST pass before commit)
uv run mypy conduit/          # Type checking (strict mode)
uv run ruff check conduit/    # Linting
uv run black conduit/         # Code formatting

# Examples
uv run python examples/hello_world.py
uv run python examples/basic_routing.py
```

---

## Tech Stack

### Core (Production Dependencies)
- **Python**: 3.10+ (required for modern type hints and async/await)
- **PydanticAI**: 1.14+ (unified LLM interface with structured outputs)
  - API: Use `Agent(...).run(..., deps=...)` not `Agent(..., deps=...)`
- **Pydantic**: 2.12+ (data validation and settings)
- **FastAPI**: 0.115+ (REST API endpoints)
- **PostgreSQL**: Any provider (self-hosted, AWS RDS, Supabase, Neon, etc.)
- **Redis**: Optional (caching with graceful degradation)
- **Embedding Provider** (required - auto-detected in priority order):
  1. **OpenAI API key** (recommended - reuses LLM key)
  2. **Cohere API key** (alternative API option)
  3. **FastEmbed** (local, ~100MB: `pip install fastembed`)
  4. **sentence-transformers** (local, ~2GB: `pip install sentence-transformers`)

### ML Stack
- **numpy**: 2.0+ (matrix operations for LinUCB)
- **scikit-learn**: PCA dimensionality reduction (optional, improves convergence)

### Development Tools
- **pytest**: 9.0+ with pytest-asyncio for async tests
- **black**: Code formatting (line length 88)
- **ruff**: Fast linting
- **mypy**: Strict type checking
- **uv**: Fast Python package manager

---

## Project Structure

```
conduit/
├── conduit/                    # Source code
│   ├── core/                   # Models, config, exceptions
│   ├── engines/                # Routing engine, bandit algorithms
│   │   └── bandits/            # LinUCB, UCB1, Thompson Sampling, Epsilon-Greedy, Dueling
│   ├── cache/                  # Redis caching with circuit breaker
│   ├── feedback/               # Implicit feedback detection
│   ├── api/                    # FastAPI routes and middleware
│   └── cli/                    # Command-line interface
├── tests/                      # All tests
│   ├── unit/                   # Unit tests (fast, isolated)
│   ├── integration/            # Integration tests (DB, Redis, API)
│   └── regression/             # Example file regression tests
├── examples/                   # Usage examples (READ ONLY)
│   ├── hello_world.py          # Minimal 5-line example
│   ├── basic_routing.py        # Core routing patterns
│   ├── feedback_loop.py        # Implicit feedback (errors, latency)
│   ├── production_feedback.py  # Explicit feedback (thumbs, ratings)
│   ├── litellm_integration.py  # 100+ providers via LiteLLM
│   └── integrations/           # LangChain, FastAPI, Gradio
├── docs/                       # Technical documentation
└── notes/                      # Strategic decisions (dated, READ ONLY)
```

---

## Code Style & Examples

### Naming Conventions
```python
# Functions and variables: snake_case
async def route_query(query: str) -> RoutingResult:
    selected_model = await bandit.select_arm(features)

# Classes: PascalCase
class LinUCBBandit(BanditAlgorithm):
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_ALPHA = 1.0
FEATURE_DIM = 386

# Private attributes: leading underscore
def _extract_features(self, features: QueryFeatures) -> np.ndarray:
    pass
```

### Type Hints (MANDATORY)
```python
# All functions require type hints
async def select_arm(self, features: QueryFeatures) -> ModelArm:
    """Select optimal model arm using UCB policy."""
    x = self._extract_features(features)  # np.ndarray
    ucb_values: dict[str, float] = {}
    return self.arms[selected_id]

# No 'Any' without justification
from typing import Any
def get_stats(self) -> dict[str, Any]:  # OK: dict values are mixed types
    pass
```

### Async/Await Patterns
```python
# All bandit methods are async
async def select_arm(self, features: QueryFeatures) -> ModelArm:
    # Async I/O would go here (DB queries, API calls)
    return selected_arm

async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
    # Update algorithm state
    pass

# Test async functions with pytest.mark.asyncio
@pytest.mark.asyncio
async def test_select_arm(test_arms, test_features):
    bandit = LinUCBBandit(test_arms)
    arm = await bandit.select_arm(test_features)
    assert arm in test_arms
```

### Docstrings (Required for Public APIs)
```python
async def select_arm(self, features: QueryFeatures) -> ModelArm:
    """Select arm using LinUCB policy.

    For each arm, compute:
        UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)

    Where theta = A_inv @ b (ridge regression coefficients)

    Args:
        features: Query features for context

    Returns:
        Selected model arm with highest UCB

    Example:
        >>> features = QueryFeatures(embedding=[0.1]*384, ...)
        >>> arm = await bandit.select_arm(features)
        >>> print(arm.model_id)
        "gpt-4o-mini"
    """
```

---

## Bandit Algorithms

### Available Algorithms (12 total)

**Learning algorithms (6):**
- `thompson_sampling` (default): Non-contextual Bayesian bandit (best cold-start quality)
- `linucb`: Contextual linear bandit (uses query features)
- `contextual_thompson_sampling`: Contextual Bayesian bandit
- `ucb1`: Non-contextual upper confidence bound
- `epsilon_greedy`: Epsilon-greedy with decaying exploration
- `dueling`: Contextual dueling bandit (pairwise comparisons with features)

**Baseline algorithms (4, for benchmarking):**
- `random`: Pure random selection (lower bound, no learning)
- `always_best`: Always pick highest expected_quality model
- `always_cheapest`: Always pick lowest cost model
- `oracle`: Remembers actual results, picks best (upper bound for benchmarking)

**Hybrid algorithms (2, legacy):**
- `hybrid_thompson_linucb`: Thompson → LinUCB transition
- `hybrid_ucb1_linucb`: UCB1 → LinUCB transition

### Usage
```python
router = Router(algorithm="thompson_sampling")  # Default
router = Router(algorithm="linucb")  # Contextual
router = Router(algorithm="hybrid_thompson_linucb")  # Hybrid
```

---

## Testing Requirements

### Coverage Requirements
- **Overall**: >80% (currently 91%)
- **Core Engine**: >95% (models, router, executor)
- **New Features**: >90% coverage

### Test Structure
```python
# Fixtures in conftest.py or test file
@pytest.fixture
def test_arms():
    return [
        ModelArm(model_id="gpt-4o-mini", provider="openai", ...),
        ModelArm(model_id="claude-3-haiku", provider="anthropic", ...)
    ]

@pytest.fixture
def test_features():
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5
    )

# Async test pattern
@pytest.mark.asyncio
async def test_select_arm_returns_valid_arm(test_arms, test_features):
    """Test arm selection returns valid arm from available arms."""
    bandit = LinUCBBandit(test_arms)
    arm = await bandit.select_arm(test_features)
    assert arm in test_arms
```

### Running Tests
```bash
# Fast (dev workflow): ~30-60s - skips slow concurrency tests
uv run pytest -m "not slow and not downloads_models and not requires_api_key"

# Full test suite: ~2-3min - runs everything including slow tests
uv run pytest

# With coverage (CI): ~5min
uv run pytest --cov=conduit --cov-report=term-missing --cov-fail-under=80

# Specific test file
uv run pytest tests/unit/test_bandits_linucb.py -v

# Regression tests (example files)
uv run pytest tests/regression/ -v
```

---

## Git Workflow

### Before Starting Work
```bash
git status              # Check current branch
git branch              # Verify not on main/master
git checkout -b feature/my-feature  # Create feature branch
```

### Pre-Commit Validation
```bash
# 1. Tests must pass
uv run pytest

# 2. Type checking must be clean
uv run mypy conduit/

# 3. Linting must be clean
uv run ruff check conduit/

# 4. Code must be formatted
uv run black conduit/

# 5. Coverage must be >80%
uv run pytest --cov=conduit --cov-fail-under=80

# 6. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" conduit/ && echo "🚨 REMOVE TODOs" && exit 1

# 7. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ && echo "🚨 CREDENTIALS FOUND" && exit 1
```

### Commit Message Format
```
Brief imperative summary (50 chars)

- Specific changes made
- Why these changes were needed
- Test coverage added/updated

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Key Concepts

### Contextual Bandits
- **Problem**: Select best action (model) given context (query features)
- **Goal**: Maximize reward (quality) while minimizing cost
- **Approach**: Balance exploration (try new models) vs exploitation (use known-good models)

### Feature Vector (386 dimensions)
```python
# 384 dimensions: Sentence embedding
embedding = sentence_transformer.encode(query_text)  # [0.1, 0.2, ...]

# 2 dimensions: Metadata
features = embedding + [
    token_count / 1000.0,      # Normalized token count
    complexity_score,          # 0.0-1.0 complexity
]
# Total: 386 dimensions
```

### Feedback Loop
- **Explicit**: User ratings (quality_score, user_rating, met_expectations)
- **Implicit**: System signals (errors, latency, retries), weighted 30%
- **Integration**: 70% explicit + 30% implicit = final reward

### Quality Guarantees
- **Probabilistic**: 95%+ of queries meet quality threshold
- **NOT deterministic**: No 100% guarantees
- **Learning-based**: Improves over time with usage

---

## Common Tasks

### Add New Bandit Algorithm
1. Create `conduit/engines/bandits/my_algorithm.py`
2. Inherit from `BanditAlgorithm` base class
3. Implement `select_arm()` and `update()` as async methods
4. Add to `conduit/engines/bandits/__init__.py` exports
5. Create `tests/unit/test_bandits_my_algorithm.py` with >90% coverage
6. Add algorithm description to CLAUDE.md

### Fix Failing Tests
1. Run specific test: `pytest tests/unit/test_file.py::test_name -v`
2. Read failure message and traceback carefully
3. Debug root cause (not just symptoms)
4. Fix implementation or test (never skip/disable tests)
5. Verify fix: `pytest tests/unit/test_file.py -v`
6. Run full suite: `pytest` (ensure no regressions)

### Add New Example
1. Create file in appropriate `examples/0X_category/` directory
2. Follow pattern: imports → setup → main logic → example output
3. Test manually: `uv run python examples/XX/file.py`
4. Verify graceful degradation (works without Redis)
5. Add regression test in `tests/regression/test_examples.py`

---

## Quick Reference

### PydanticAI Agent Pattern
```python
from pydantic_ai import Agent
from pydantic import BaseModel

class MyOutput(BaseModel):
    result: str
    confidence: float

agent = Agent(model="openai:gpt-4o-mini", result_type=MyOutput)
result = await agent.run("Your prompt", deps=dependencies)
```

### Bandit Usage Pattern
```python
# Initialize
bandit = LinUCBBandit(arms, alpha=1.0, feature_dim=386)

# Select model
features = QueryFeatures(embedding=..., token_count=..., ...)
arm = await bandit.select_arm(features)

# Execute query with selected model
response = await execute_query(arm.model_id, query)

# Provide feedback
feedback = BanditFeedback(
    model_id=arm.model_id,
    cost=response.cost,
    quality_score=0.95,
    latency=response.latency
)
await bandit.update(feedback, features)
```

### Feature Extraction
```python
def _extract_features(self, features: QueryFeatures) -> np.ndarray:
    """Extract 386-dim feature vector."""
    feature_vector = np.array(
        features.embedding  # 384 dims
        + [
            features.token_count / 1000.0,  # Normalize
            features.complexity_score,
        ]  # 2 dims
    )
    return feature_vector.reshape(-1, 1)  # Column vector (386, 1)
```
