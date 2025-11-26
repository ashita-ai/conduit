# Contributing to Conduit

Thank you for your interest in contributing to Conduit! This guide will help you get started.

## Quick Start

### Prerequisites

- Python 3.11+ (3.13 recommended)
- At least one LLM API key (OpenAI, Anthropic, Google, Groq, etc.)
- Redis (optional - for caching)
- PostgreSQL (optional - for history persistence)

### Setup

```bash
# Clone the repository
git clone https://github.com/ashita-ai/conduit.git
cd conduit

# Create and activate virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install git hooks (recommended)
bash scripts/install-hooks.sh

# Run tests to verify setup
pytest
```

### Configuration

Create `.env` file:
```bash
# At least one LLM provider API key
OPENAI_API_KEY=your_key
# or
ANTHROPIC_API_KEY=your_key

# Optional
DATABASE_URL=postgresql://postgres:password@localhost:5432/conduit
REDIS_URL=redis://localhost:6379
```

## Development Workflow

### 1. Create a Feature Branch

**IMPORTANT**: Never work directly on `main`

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow these coding standards:

**Type Hints (Mandatory)**
```python
async def select_arm(self, features: QueryFeatures) -> ModelArm:
    """All functions require type hints."""
    pass
```

**Naming Conventions**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

**Async/Await**
- All bandit methods must be `async`
- Use `asyncio` for I/O operations
- No blocking calls in async functions

### 3. Write Tests

**Test Coverage Requirements:**
- Overall: >80% (enforced by CI)
- New features: >90% coverage

```python
# Example test structure
@pytest.mark.asyncio
async def test_feature_name(test_fixtures):
    """Test description."""
    # Arrange
    bandit = LinUCBBandit(test_arms)
    
    # Act
    result = await bandit.select_arm(test_features)
    
    # Assert
    assert result in test_arms
```

### 4. Run Quality Checks

Before committing, run all checks:

```bash
# Tests (must pass)
pytest

# Type checking (must be clean)
mypy conduit/

# Linting (must pass)
ruff check conduit/

# Formatting (auto-fix)
black conduit/

# Coverage (must be >80%)
pytest --cov=conduit --cov-fail-under=80
```

**Note**: If you installed git hooks, these run automatically before push.

### 5. Commit Your Changes

```bash
git add <files>
git commit -m "Brief description

- Specific changes made
- Why these changes were needed
- Test coverage added
"
```

### 6. Push and Create Pull Request

```bash
git push -u origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear description of changes
- Link to related issues (`Closes #123`)
- Screenshots (if UI changes)

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` or `difficulty: beginner`:
- Documentation improvements
- Test coverage increases
- Simple bug fixes
- Example code additions

### Areas Needing Help

- **Testing**: Increase coverage in low-coverage areas
- **Documentation**: Improve docstrings, add examples
- **Performance**: Optimize hot paths with benchmarks
- **Integration**: Test with different LLM providers

## Code Standards

### What We Enforce

- ✅ Type hints on all functions (mypy strict mode)
- ✅ Test coverage >80% (pytest-cov)
- ✅ Clean linting (ruff)
- ✅ Consistent formatting (black, line length 88)
- ✅ Async/await patterns for I/O
- ✅ No placeholders or TODO comments in production code

### What We Don't Allow

- ❌ Working directly on `main` branch
- ❌ Skipping or disabling tests
- ❌ Committing credentials or API keys
- ❌ Partial implementations (no `NotImplementedError`)
- ❌ Using `Any` type without justification

## Common Tasks

### Adding a New Bandit Algorithm

1. Create `conduit/engines/bandits/my_algorithm.py`
2. Inherit from `BanditAlgorithm` base class
3. Implement `async def select_arm()` and `async def update()`
4. Add to `conduit/engines/bandits/__init__.py` exports
5. Create `tests/unit/test_bandits_my_algorithm.py` with >90% coverage
6. Add algorithm description to documentation

### Running Examples

```bash
# Basic routing
uv run python examples/01_quickstart/hello_world.py

# With constraints
uv run python examples/02_routing/with_constraints.py

# Caching
uv run python examples/03_optimization/caching.py
```

## Getting Help

- **Issues**: Check existing [issues](https://github.com/ashita-ai/conduit/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/ashita-ai/conduit/discussions)
- **Documentation**: See [docs/](docs/) directory

## Pull Request Checklist

Before submitting your PR, verify:

- [ ] Tests pass: `pytest`
- [ ] Type checking clean: `mypy conduit/`
- [ ] Linting clean: `ruff check conduit/`
- [ ] Formatting applied: `black conduit/`
- [ ] Coverage >80%: `pytest --cov=conduit --cov-fail-under=80`
- [ ] No credentials committed
- [ ] Documentation updated (if needed)
- [ ] Examples added/updated (if needed)

## Code Review Process

1. **Automated Checks**: CI must pass (tests, linting, coverage)
2. **Manual Review**: Maintainer reviews code quality and design
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves and merges

Typical turnaround: 1-3 days for initial review.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
