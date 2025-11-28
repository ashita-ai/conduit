---
name: conduit_agent
description: ML-powered LLM routing system developer implementing contextual bandits, managing tests, and maintaining production-grade Python code
last_updated: 2025-11-27
---

# Conduit Agent Guide

**What is Conduit**: ML-powered LLM routing system that learns which model to use for your workload, reducing costs 30-50% while maintaining quality through contextual bandit algorithms.

**Your Role**: Python ML engineer specializing in contextual bandits, async/await patterns, and type-safe code. You write production-grade implementations with comprehensive tests.

**Design Philosophy**: Simplicity wins, use good defaults, YAML config where needed, no hardcoded assumptions.

**Current Phase**: Pre-1.0 preparation (version 0.1.0, CI/CD complete, state persistence done)
**Test Health**: 100% passing (684 tests), 81% coverage
**Latest**: Thompson Sampling default algorithm (PR #169 - BREAKING), state persistence infinity fix, all examples passing (16/16)

---

## Quick Start (First Session Commands)

**New to this repo? Run these 5 commands first:**

```bash
# 1. Verify you're on a feature branch (NEVER work on main)
git status && git branch

# 2. Install dependencies
uv sync --all-extras && source .venv/bin/activate

# 3. Run tests to verify environment
uv run pytest tests/unit/test_bandits_linucb.py -v

# 4. Check for any TODOs or placeholders (should be NONE)
grep -r "TODO\|FIXME\|NotImplementedError" conduit/ || echo "✅ No placeholders found"

# 5. Verify type checking works
uv run mypy conduit/ --no-error-summary || echo "⚠️ Fix type errors before starting"
```

---

## Breaking Changes (Pre-1.0)

### PR #169: Default Algorithm Changed to Thompson Sampling (2025-11-27)

**Breaking API Change**: `Router.__init__()` parameter modification

**OLD (Removed)**:
```python
router = Router(use_hybrid_routing=False)  # REMOVED parameter
```

**NEW (Required)**:
```python
router = Router(algorithm="thompson_sampling")  # Default: thompson_sampling
```

**Migration Guide**:
- **If you used default (no params)**: No change needed, new default is Thompson Sampling
- **If you used `use_hybrid_routing=True`**: Use `algorithm="hybrid_thompson_linucb"` (maintains old behavior)
- **If you used `use_hybrid_routing=False`**: Use `algorithm="linucb"` (contextual algorithm)

**Available Algorithms**:
- `thompson_sampling` (default): Non-contextual Bayesian bandit (best cold-start quality, research-backed arXiv 2510.02850)
- `linucb`: Contextual linear bandit (uses query features)
- `contextual_thompson_sampling`: Contextual Bayesian bandit
- `ucb1`: Non-contextual upper confidence bound
- `epsilon_greedy`: Epsilon-greedy with decaying exploration (non-contextual baseline)
- `random`: Pure random selection baseline (non-contextual, no learning)
- `dueling`: Contextual dueling bandit (pairwise comparisons with features)
- `hybrid_thompson_linucb`: Thompson → LinUCB transition (old default, legacy)
- `hybrid_ucb1_linucb`: UCB1 → LinUCB transition (legacy)

**Rationale**: Thompson Sampling provides superior cold-start quality through Bayesian exploration (research paper arXiv 2510.02850). Simpler default, no complex hybrid logic needed for most users.

**Tests**: All 684 tests passing, all 16 examples verified, no functional regressions.

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
- Verify tests pass with `pytest -v`
- Check coverage with `pytest --cov`
- Test both success and failure cases

**Documentation**:
- Add docstrings to all public functions/classes
- Include Args, Returns, Raises, and Example sections
- Update AGENTS.md when adding new features
- Create examples in `examples/` directory

### Ask First

**Architecture Changes** (Why: Affects all dependent code):
- Modifying base classes (`BanditAlgorithm`, `ModelArm`) - Changes break all algorithm implementations
- Changing API contracts (function signatures, return types) - Breaking change for users
- Adding new dependencies to `pyproject.toml` - Increases attack surface and maintenance burden
- Changing database schema or migrations - Requires data migration plan and rollback strategy

**Risky Operations** (Why: High blast radius):
- Deleting existing algorithms or features - Users may depend on them
- Refactoring core routing logic - Subtle bugs can break all routing
- Modifying production configuration defaults - Can cause performance regression or cost spike
- Changing test fixtures that affect many tests - Mass test failures obscure real issues

**External Services** (Why: Security and reliability implications):
- Adding new MCP servers or external APIs - Security review needed
- Changing Redis/PostgreSQL connection patterns - Can cause connection pool exhaustion
- Modifying authentication or rate limiting logic - Security-critical, needs review

### Never Do

**Security (CRITICAL)**:
- NEVER EVER COMMIT CREDENTIALS TO GITHUB
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is non-negotiable with serious security consequences

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
- Create files outside designated directories

**Detection Commands** (Run before committing):
```bash
# Check for security violations
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"

# Check for code quality violations
grep -r "TODO\|FIXME" conduit/ && echo "🚨 TODO comments found" || echo "✅ No TODOs"
grep -r "# type: ignore" conduit/ | grep -v "justification:" && echo "⚠️ Unjustified type ignores" || echo "✅ Type ignores justified"

# Check for anti-patterns
grep -r "from typing import Any" conduit/ && echo "⚠️ Using Any - justify in comment" || echo "✅ No Any usage"

# Verify on feature branch
git branch --show-current | grep -E "^(main|master)$" && echo "🚨 ON MAIN BRANCH - CREATE FEATURE BRANCH" || echo "✅ On feature branch"
```

---

## Communication Preferences

Don't flatter me. I know what [AI sycophancy](https://www.seangoedecke.com/ai-sycophancy/) is and I don't want your praise. Be concise and direct. Don't use emdashes ever.

---

## Session Analysis & Continuous Improvement

**When to Analyze** (Multiple Triggers):
- During active sessions: After completing major tasks or every 30-60 minutes
- When failures occur: Immediately analyze and update rules
- Session end: Review entire session for patterns before closing
- User corrections: Any time user points out a mistake

**Identify Failures**:
- Framework violations (boundaries crossed, rules ignored)
- Repeated patterns (same mistake multiple times)
- Rules that didn't prevent failures
- User corrections (what needed fixing)

**Analyze Each Failure**:
- What rule should have prevented this?
- Why didn't it work? (too vague, wrong priority, missing detection pattern)
- What would have caught this earlier?

**Update AGENTS.md** (In Real-Time):
- Add new rules or strengthen existing rules immediately
- Add detection patterns (git commands, test patterns, code patterns)
- Include examples of violations and corrections
- Update priority if rule was underweighted
- Propose updates to user during session (don't wait until end)

**Priority Levels**:
- 🔴 **CRITICAL**: Security, credentials, production breaks → Update immediately, stop work
- 🟡 **IMPORTANT**: Framework violations, repeated patterns → Update with detection patterns, continue work
- 🟢 **RECOMMENDED**: Code quality, style issues → Update with examples, lowest priority

**Example Pattern**:
```
Failure: Committed TODO comments in production code (violated "No Partial Features" rule)
Detection: `grep -r "TODO" src/` before commit
Rule Update: Add pre-commit check pattern to Boundaries section
Priority: 🟡 IMPORTANT
Action Taken: Proposed rule update to user mid-session, updated AGENTS.md
```

**Proactive Analysis**:
- Before risky operations: Check if existing rules cover this scenario
- After 3+ similar operations: Look for pattern that should be codified
- When uncertainty arises: Document the decision-making gap

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Working on main/master Branch
**Detection**: `git branch --show-current` shows "main" or "master"
**Prevention**: Always run `git checkout -b feature/my-feature` before starting
**Fix**: `git checkout -b feature/my-feature` then continue work
**Why It Matters**: Direct commits to main bypass PR review and CI checks

### Mistake 2: Skipping Tests to Make Build Pass
**Detection**: Git history shows test files deleted or `@pytest.mark.skip` added
**Prevention**: Run `pytest` before committing, fix failures not tests
**Fix**: Revert test changes, fix implementation instead
**Why It Matters**: Broken tests indicate broken code, not broken tests

### Mistake 3: Adding Dependencies Without Consideration
**Detection**: `git diff pyproject.toml` shows new dependencies
**Prevention**: Ask first, check if existing deps can solve the problem
**Fix**: Remove dependency, use standard library or existing deps
**Why It Matters**: Each dependency increases security risk and maintenance burden

### Mistake 4: Mixing Sync and Async Code
**Detection**: Functions with `async def` calling sync blocking I/O
**Prevention**: All I/O in bandit methods must be async
**Fix**: Use `aiofiles`, `asyncpg`, `asyncio.to_thread()` for blocking calls
**Why It Matters**: Blocks event loop, causes performance degradation

### Mistake 5: Leaving TODO Comments in Production Code
**Detection**: `grep -r "TODO" conduit/` returns matches
**Prevention**: Complete feature fully or don't commit it
**Fix**: Remove TODO, implement feature or remove partial code
**Why It Matters**: Indicates incomplete work, confuses future developers

### Mistake 6: Using `Any` Type Without Justification
**Detection**: `grep -r "from typing import Any" conduit/` without `# justification:` comment
**Prevention**: Use specific types or Union types instead
**Fix**: Replace `Any` with correct type or add `# justification: dict values are mixed types`
**Why It Matters**: Defeats purpose of type checking, hides bugs

### Mistake 7: Ignoring Test Coverage Drop
**Detection**: `pytest --cov=conduit` shows coverage below 80%
**Prevention**: Write tests for all new code before committing
**Fix**: Add missing tests until coverage >80%
**Why It Matters**: Untested code will break in production

---

## Testing Decision Matrix

**When to Mock:**
- External API calls (OpenAI, Anthropic) - Use mocked responses
- Database queries - Use in-memory SQLite or mocked connections
- Redis cache - Use fakeredis library
- File I/O - Use temporary directories with `pytest.tmp_path`
- Time-dependent code - Mock `datetime.now()` for determinism

**When to Use Real Dependencies:**
- NumPy operations - Real numpy arrays (fast enough)
- Pydantic validation - Real validation (catches schema issues)
- Internal function calls - Real calls (integration testing)
- Configuration loading - Real config objects
- Exception handling - Real exceptions

**Detection Pattern:**
```python
# ✅ GOOD - Mock external API
@pytest.mark.asyncio
async def test_router_with_mocked_llm(mocker):
    mocker.patch("conduit.engines.router.call_llm", return_value="mocked response")

# ✅ GOOD - Use real numpy
async def test_linucb_feature_extraction():
    features = QueryFeatures(embedding=[0.1]*384, ...)
    bandit = LinUCBBandit(test_arms)
    vector = bandit._extract_features(features)  # Real numpy operation

# ❌ BAD - Using real API in tests
async def test_router():
    response = await call_openai_api(...)  # Slow, costs money, flaky
```

---

## Regression Testing

Regression tests ensure that all example files work correctly and catch breaking changes.

### Running Regression Tests
```bash
# Run all regression tests
pytest tests/regression/ -v

# Run using pytest marker
pytest -m regression

# Run specific example category
pytest tests/regression/ -k litellm

# Run with summary (shows skip reasons)
pytest tests/regression/ -v --tb=short
```

### Test Structure
- **tests/regression/test_examples.py**: 18 tests covering all 16 example files
- **Automatic Skipping**: Tests auto-skip if required API keys or dependencies missing
- **Meta Tests**: `test_all_examples_have_tests()` ensures we don't forget new examples

### Example Categories Tested
1. **Quickstart** (2 examples): hello_world.py, simple_router.py
2. **Routing** (4 examples): basic_routing.py, hybrid_routing.py, with_constraints.py, context_specific_priors.py
3. **Optimization** (3 examples): caching.py, pca_comparison.py, state_persistence.py
4. **LiteLLM** (5 examples): basic_usage.py, custom_config.py, learning_demo.py, multi_provider.py, arbiter_quality_measurement.py
5. **Personalization** (1 example): explicit_preferences.py
6. **Integrations** (1 example): langchain_integration.py

### Adding Tests for New Examples
When adding a new example file, add a corresponding test in `test_examples.py`:
```python
@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY")  # List required API keys
def test_my_new_example():
    """Test examples/XX_category/my_new_example.py runs successfully."""
    example = EXAMPLES_DIR / "XX_category" / "my_new_example.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"
    assert "expected_output" in stdout  # Verify expected behavior
```

---

## Executable Commands

Run these commands exactly as shown. They must pass before any commit.

```bash
# Development
uv sync --all-extras          # Install all dependencies
source .venv/bin/activate     # Activate virtual environment

# Testing (MUST pass before commit)
uv run pytest -m "not slow and not downloads_models and not requires_api_key"  # Fast dev tests (~30-60s)
uv run pytest                 # Full test suite (~2-3min, includes slow tests)
uv run pytest --cov=conduit   # Run with coverage (must be >80%, ~5min)
uv run pytest tests/unit/test_bandits*.py -v  # Test bandit algorithms

# Code Quality (MUST pass before commit)
uv run mypy conduit/          # Type checking (strict mode)
uv run ruff check conduit/    # Linting
uv run black conduit/         # Code formatting

# Examples
uv run python examples/01_quickstart/hello_world.py
uv run python examples/02_routing/basic_routing.py
```

### CI/CD Testing Strategy

**GitHub Actions runs different tests based on context:**

```yaml
# Pull Requests: Fast feedback (~1-2min)
- Fast tests only (no coverage overhead)
- Skips slow concurrency tests
- Command: pytest -m "not slow and not downloads_models and not requires_api_key"

# Main branch & Release tags: Full quality assurance (~5min)
- Complete test suite with coverage
- Includes slow concurrency tests
- Command: pytest --cov=conduit --cov-fail-under=80
```

**Why this matters:**
- PRs get fast feedback for iterative development
- Releases get comprehensive coverage validation
- Balances speed and quality appropriately

---

## Model Reference (Current as of 2025-11-25)

**CRITICAL**: Always use current model versions. Never use outdated model IDs.

### Top 5 Providers - Current Best Models

| Provider | Flagship | Fast/Cheap | Notes |
|----------|----------|------------|-------|
| **OpenAI** | `gpt-5.1` | `o4-mini` | GPT-5.1 series (Nov 2025), 76.3% SWE-bench |
| **Anthropic** | `claude-opus-4-5-20241124` | `claude-haiku-4-5-20241124` | Claude 4.5 series, Opus 80.9% SWE-bench |
| **Google** | `gemini-3.0-pro` | `gemini-2.5-flash` | Gemini 3 (Nov 18), topped LMArena |
| **Meta** | `llama-4-maverick` | `llama-4-scout` | Llama 4 via Groq/Together |
| **Mistral** | `mistral-large-latest` | `mistral-small-latest` | Use `-latest` suffix |

### Default Models by Use Case

```yaml
# conduit.yaml - models section
models:
  # Default routing pool (balanced across providers)
  default:
    - o4-mini               # OpenAI fast reasoning
    - gpt-5.1               # OpenAI flagship
    - claude-sonnet-4-5-20241124  # Anthropic balanced
    - gemini-2.5-flash      # Google fast

  # Arbiter evaluation (cheap, fast)
  arbiter: o4-mini

  # Fallback when model unknown
  fallback: o4-mini
```

### LiteLLM Model Format

When using LiteLLM, prefix with provider:
```python
# LiteLLM format
"openai/gpt-5.1"
"anthropic/claude-opus-4-5-20241124"
"gemini/gemini-3.0-pro"
"groq/llama-4-maverick"
```

### Model ID Patterns (for unknown models)

When someone requests a model not in our config:
1. Check if it matches a known provider pattern
2. Use provider's latest fast model as fallback
3. Log warning for unknown models

```python
# Pattern matching for fallbacks
def get_fallback_model(model_id: str) -> str:
    """Get fallback model for unknown model IDs."""
    model_lower = model_id.lower()

    if "gpt" in model_lower or "openai" in model_lower or "o4" in model_lower:
        return "o4-mini"
    elif "claude" in model_lower or "anthropic" in model_lower:
        return "claude-haiku-4-5-20241124"
    elif "gemini" in model_lower or "google" in model_lower:
        return "gemini-2.5-flash"
    elif "llama" in model_lower or "meta" in model_lower:
        return "llama-4-scout"
    elif "mistral" in model_lower:
        return "mistral-small-latest"
    else:
        return "o4-mini"  # Global fallback
```

### Pricing Reference (Nov 2025, per 1M tokens)

| Model | Input | Output | Source |
|-------|-------|--------|--------|
| o4-mini | $1.10 | $4.40 | openai.com |
| gpt-5.1 | $2.00 | $8.00 | openai.com |
| claude-opus-4-5 | $5.00 | $25.00 | anthropic.com |
| claude-haiku-4-5 | $0.80 | $4.00 | anthropic.com |
| gemini-3.0-pro | $1.25 | $5.00 | ai.google.dev |
| gemini-2.5-flash | $0.075 | $0.30 | ai.google.dev |

**Update Frequency**: Check pricing monthly at:
- https://openai.com/api/pricing
- https://anthropic.com/pricing
- https://ai.google.dev/pricing
- https://artificialanalysis.ai/leaderboards/providers

Sources:
- [OpenAI GPT-5.1 announcement](https://openai.com/index/gpt-5-1-for-developers/)
- [Anthropic Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5)
- [Google Gemini 3](https://blog.google/products/gemini/gemini-3/)

---

## Tech Stack

### Core (Production Dependencies)
- **Python**: 3.10+ (required for modern type hints and async/await)
- **PydanticAI**: 1.14+ (unified LLM interface with structured outputs)
  - API: Use `Agent(...).run(..., deps=...)` not `Agent(..., deps=...)`
- **Pydantic**: 2.12+ (data validation and settings)
- **FastAPI**: 0.115+ (REST API endpoints)
- **PostgreSQL**: Any provider (self-hosted, AWS RDS, Supabase, Neon, etc.) for routing history
- **Redis**: Optional (caching and rate limiting with graceful degradation)
- **Embedding Provider** (required - auto-detected in priority order):
  1. **OpenAI API key** (recommended - reuses LLM key, no additional setup)
  2. **Cohere API key** (alternative API option)
  3. **FastEmbed** (local, ~100MB: `pip install fastembed`)
  4. **sentence-transformers** (local, ~2GB: `pip install sentence-transformers`)

### ML Stack
- **numpy**: 2.0+ (matrix operations for LinUCB)
- **scikit-learn**: PCA dimensionality reduction (optional, improves convergence speed)
  - **PCA components:** Provider-dependent (64 for 384-dim, 128 for 1536-dim embeddings)
  - **OpenAI (1536-dim):** 64 components = 57% variance, 128 = 73%, 192 = 85%
  - **FastEmbed (384-dim):** 64 components = 95% variance (excellent)

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
│   │   └── bandits/            # LinUCB, UCB1, Thompson Sampling, Epsilon-Greedy
│   ├── cache/                  # Redis caching with circuit breaker
│   ├── feedback/               # Implicit feedback detection
│   ├── api/                    # FastAPI routes and middleware
│   └── cli/                    # Command-line interface
├── tests/                      # All tests
│   ├── unit/                   # Unit tests (fast, isolated)
│   ├── integration/            # Integration tests (DB, Redis, API)
│   └── conftest.py             # Test configuration (uses real numpy)
├── examples/                   # Usage examples (READ ONLY)
│   ├── 01_quickstart/          # hello_world.py, simple_router.py
│   ├── 02_routing/             # basic_routing.py, with_constraints.py
│   └── 03_optimization/        # caching.py, feedback examples
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
from typing import Any, Optional
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

Current implementations in `conduit/engines/bandits/`:

### LinUCB (Contextual, Best for LLM Routing)
- **File**: `linucb.py` (12/12 tests passing, 98% coverage)
- **Algorithm**: Ridge regression with upper confidence bound
- **State**: A matrix (d×d), b vector (d×1) per arm
- **Selection**: `UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)`
- **Update**: `A += x @ x^T`, `b += reward * x`
- **Features**: 386 dims (384 embedding + 2 metadata)
- **When to use**: Default choice, uses query context for better decisions

### UCB1 (Non-contextual)
- **File**: `ucb.py` (11/11 tests passing, 17% coverage)
- **Algorithm**: Upper confidence bound with logarithmic exploration
- **Selection**: `mean + c * sqrt(log(total) / pulls)`
- **When to use**: Simpler baseline, ignores query context

### Thompson Sampling (Bayesian)
- **File**: `thompson_sampling.py` (8/9 tests passing)
- **Algorithm**: Beta distribution sampling per arm
- **State**: α (successes + 1), β (failures + 1) per arm
- **When to use**: Good exploration/exploitation balance

### Epsilon-Greedy (Simple)
- **File**: `epsilon_greedy.py` (14/14 tests passing)
- **Algorithm**: Explore with probability ε, else exploit best
- **When to use**: Baseline for comparison

---

## Testing Requirements

### Coverage Requirements
- **Overall**: >80% (currently 83%)
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
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8
    )

# Async test pattern
@pytest.mark.asyncio
async def test_select_arm_returns_valid_arm(test_arms, test_features):
    """Test arm selection returns valid arm from available arms."""
    bandit = LinUCBBandit(test_arms)
    arm = await bandit.select_arm(test_features)

    assert arm in test_arms
    assert arm.model_id in ["gpt-4o-mini", "claude-3-haiku"]
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

# Unit tests only (fast)
uv run pytest tests/unit/
```

---

## Git Workflow

### Before Starting Work
```bash
git status              # Check current branch
git branch              # Verify not on main/master
git checkout -b feature/my-feature  # Create feature branch
```

### Development Cycle

**Pre-Commit Validation (Run ALL these checks):**
```bash
# 1. Tests must pass
uv run pytest
if [ $? -ne 0 ]; then echo "🚨 TESTS FAILED - DO NOT COMMIT"; exit 1; fi

# 2. Type checking must be clean
uv run mypy conduit/
if [ $? -ne 0 ]; then echo "🚨 TYPE ERRORS - DO NOT COMMIT"; exit 1; fi

# 3. Linting must be clean
uv run ruff check conduit/
if [ $? -ne 0 ]; then echo "🚨 LINT ERRORS - DO NOT COMMIT"; exit 1; fi

# 4. Code must be formatted
uv run black conduit/ --check
if [ $? -ne 0 ]; then echo "⚠️ Running black formatter..."; uv run black conduit/; fi

# 5. Coverage must be >80%
uv run pytest --cov=conduit --cov-fail-under=80
if [ $? -ne 0 ]; then echo "🚨 COVERAGE BELOW 80% - ADD TESTS"; exit 1; fi

# 6. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" conduit/ && echo "🚨 REMOVE TODOs" && exit 1

# 7. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ && echo "🚨 CREDENTIALS FOUND" && exit 1

# All checks passed - safe to commit
echo "✅ All checks passed - ready to commit"
git add <files>
git commit -m "Clear message describing what + why"
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

## Security

### Automated Dependency Scanning

**GitHub Dependabot** is configured to automatically scan dependencies for security vulnerabilities:

- **Configuration**: `.github/dependabot.yml`
- **Schedule**: Weekly scans every Monday at 9:00 AM ET
- **Scope**: All Python dependencies in `pyproject.toml`
- **Alerts**: Security updates get separate PRs with critical priority
- **Grouping**: Non-security updates are grouped to reduce PR noise
  - Production dependencies: All runtime deps (pydantic, fastapi, etc.)
  - Development dependencies: Testing and linting tools (pytest, black, ruff, mypy)

**How It Works**:
1. Dependabot scans dependencies weekly for known CVEs
2. Security vulnerabilities trigger immediate PRs (separate from grouped updates)
3. Maintainers review and merge security patches
4. Non-security updates are grouped and reviewed on the weekly schedule

**Viewing Security Alerts**:
- Navigate to repository → Security tab → Dependabot alerts
- Each alert shows CVE details, affected versions, and remediation steps
- PRs are auto-created with version updates to fix vulnerabilities

**Best Practices**:
- Review and merge security PRs promptly (within 24-48 hours)
- Test security updates before merging to production
- Monitor Dependabot alerts in the Security tab
- Keep dependencies up to date to minimize vulnerability exposure

**Detection Commands**:
```bash
# Validate Dependabot configuration
cat .github/dependabot.yml

# Check for any credentials that shouldn't be committed
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"
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
6. Add algorithm description to AGENTS.md

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
5. Add to AGENTS.md examples list

---

## Current Status (2025-11-27)

### Latest: Router.update() Real Features Fix (PR #168) ✅

**Critical Bug Fix** (PR #168, merged 2025-11-27):
- **Problem**: Router.update() was using dummy features (all zeros) instead of real query features
- **Impact**: Completely broke contextual learning in LinUCB algorithm
- **Solution**: Added `features: QueryFeatures` parameter to Router.update()
- **Result**: LinUCB now learns from real semantic query patterns
- **Tests**: All 643 tests passing, 83% coverage maintained

### Previous: State Persistence & Production Resilience ✅

**Automatic State Persistence** (PR #147):
- **Auto-load** state on first `route()` call (can't await in `__init__`)
- **Save after every update()** (~1-5ms overhead, negligible vs ~500ms LLM)
- **Periodic checkpoints** every N queries (backup, default 100)
- **Final save on close()** for graceful shutdown
- **Never lose >1 query** of learning on crash
- **6 integration tests** covering all scenarios
- **Removed outdated comments** about state persistence not being supported

**Database Pool Timeout Fix**:
- **Problem**: Integration tests hung 60+ seconds during teardown
- **Root cause**: `pool.close()` waited indefinitely for connections
- **Solution**: 10-second timeout using `asyncio.wait_for()` (per asyncpg docs)
- **Result**: All integration tests now fast (82s vs hanging)

**Hybrid Routing Integration Tests** (PR #145):
- **8 tests** covering UCB1→LinUCB transition lifecycle
- **State persistence** across router restarts
- **Knowledge transfer** from UCB1 to LinUCB
- **Phase detection** accuracy
- **Feedback learning** in both phases

**Pytest Speed Optimization** (PR #144):
- **Fast tests**: Skip slow/downloads/API tests on PR (1.5min vs 5min)
- **Full suite**: Run all tests on main/releases
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.downloads_models`, `@pytest.mark.requires_api_key`
- **Pre-push hook**: Fast tests only for quick feedback

### Test Health (Updated 2025-11-27)
- **Overall**: 643 passing, 26 skipped (100% pass rate), 83% coverage
- **Unit Tests**: ~620 passing
- **Integration Tests**: 20 passing (API + database + hybrid routing + auto-persistence)
- **Skipped**: 26 (optional deps: litellm, Redis, sentence-transformers, API keys)
- **All Bandit Algorithms**: 100% passing

### Previous: CI/CD & Test Infrastructure Complete (2025-11-26)

**GitHub Actions CI Workflow** (PR #112, commits: 454d241, 0b75957, 8e26310):
- **CI Pipeline** (`.github/workflows/ci.yml`):
  - Runs on every PR and push to main
  - PostgreSQL 15 service for integration tests
  - Database migrations via Alembic
  - Test suite: 565 passing, 0 failing (100% pass rate)
  - Coverage: 81% (exceeds >80% requirement)
  - Linting: ruff (auto-fix applied, ignore rules for math notation)
  - Formatting: black (35 files reformatted)
  - Type checking: mypy (non-blocking until errors fixed)

**Pre-Push Git Hook** (`scripts/git-hooks/pre-push`):
- Mirrors CI checks locally before push
- Runs: ruff → black → mypy → unit tests
- Fast feedback (unit tests only, ~30s)
- Installable via `bash scripts/install-hooks.sh`
- Skip with: `git push --no-verify`

**Scope Documentation** (`docs/SCOPE.md`):
- Defines what Conduit does (ML routing, state persistence, integrations)
- Defines what Conduit doesn't do (multi-tenancy, custom scoring, model hosting, auth, streaming, UI)
- Decision framework for feature evaluation
- Gray areas documented (failover, cost budgets, A/B testing)

**Issue Tracking** (Pre-1.0 preparation):
- #110: Migration testing checklist
- #111: Release automation process

### Test Health (Historical - 2025-11-26)
- **Overall**: 100% passing (565 tests), 81% coverage
- **Unit Tests**: 553 passing
- **Integration Tests**: 12 passing (API + database)
- **Skipped**: 14 (optional deps: litellm, Redis, sentence-transformers)
- **All Bandit Algorithms**: 100% passing
  - Contextual Thompson Sampling: 17/17
  - LinUCB: 12/12
  - Epsilon-Greedy: 14/14
  - UCB1: 11/11
  - Thompson Sampling: 8/8
  - Non-stationarity: 11/11

### Recent Work (2025-11-26)
- ✅ Achieved 100% test pass rate (fixed database integration tests)
- ✅ Automated CI/CD with GitHub Actions
- ✅ Pre-push hooks for local validation
- ✅ Scope boundaries documented
- ✅ Linting/formatting standardized
- ✅ Database migration fixed (statement_cache_size=0)

### Previous: Measurement & Evaluation (2025-11-22)

**Arbiter LLM-as-Judge Integration**:
- Async LLM-as-judge quality assessment
- Fire-and-forget evaluation (doesn't block routing)
- Configurable sampling with budget control
- 11/11 tests passing, 100% coverage

**LiteLLM Feedback Loop**:
- Automatic bandit learning from LiteLLM callbacks
- Zero-config automatic learning
- Comprehensive unit tests (19 tests)

**Performance Optimizations**:
- Hybrid Routing (30% faster convergence)
- PCA Dimensionality Reduction (75% sample reduction)
- Dynamic Pricing & Model Discovery

**Phase 3 Complete**:
- Multi-objective reward function
- Non-stationarity handling
- Contextual Thompson Sampling

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
