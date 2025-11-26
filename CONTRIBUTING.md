# Contributing to Conduit

Thank you for your interest in contributing to Conduit! This document provides guidelines for contributing to the ML-powered LLM routing system.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

Be respectful and professional. We're building production-grade ML systems together.

## Getting Started

### Prerequisites

- Python 3.10+ (3.13 recommended)
- At least one LLM provider API key (OpenAI, Anthropic, Google, Groq, Mistral, Cohere)
- Redis (optional, for caching)
- PostgreSQL (optional, for history persistence)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ashita-ai/conduit.git
cd conduit

# Install dependencies
uv sync --all-extras
source .venv/bin/activate

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and connection strings

# Run tests to verify setup
uv run pytest tests/unit/test_bandits_linucb.py -v
```

## Making Changes

### Before You Start

1. **Check existing issues** - Look for related issues or discussions
2. **Create an issue first** - For significant changes, discuss your approach
3. **Create a feature branch** - Never work directly on `main`

```bash
git checkout -b feature/your-feature-name
```

### Code Standards

- **Type hints required** - All functions must have complete type annotations (strict mypy)
- **Async/await patterns** - All bandit methods must be async
- **Docstrings required** - Use Google-style docstrings with Args, Returns, Raises, Example
- **No placeholders** - No TODO comments or NotImplementedError in production code
- **Complete features** - Finish what you start (implementation + tests + docs + examples)

### Bandit Algorithm Pattern

All bandit algorithms must inherit from `BanditAlgorithm`:

```python
class MyBandit(BanditAlgorithm):
    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select optimal model arm using your algorithm.
        
        Args:
            features: Query features for context (387 dims)
            
        Returns:
            Selected model arm with highest expected reward
        """
        pass

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update algorithm state with feedback.
        
        Args:
            feedback: Reward signal (quality, cost, latency)
            features: Query features used for selection
        """
        pass
```

## Testing

### Running Tests

```bash
# All tests (88% passing, 87% coverage)
uv run pytest

# Specific test file
uv run pytest tests/unit/test_bandits_linucb.py -v

# With coverage report
uv run pytest --cov=conduit --cov-report=term-missing

# Fast unit tests only
uv run pytest tests/unit/
```

### Test Requirements

- **Coverage** - Maintain >80% coverage for all new code
- **Unit tests** - Test individual functions with mocked dependencies
- **Integration tests** - Test end-to-end routing flows
- **Mock external APIs** - Don't hit real LLM APIs in unit tests
- **Use real numpy** - Don't mock numpy operations (fast enough)

## Code Quality

### Pre-Commit Checklist

Run these commands before every commit:

```bash
# 1. Tests pass
uv run pytest
# Exit if failed

# 2. Type checking clean
uv run mypy conduit/
# Exit if failed

# 3. Linting clean
uv run ruff check conduit/
# Exit if failed

# 4. Formatted
uv run black conduit/

# 5. Coverage >80%
uv run pytest --cov=conduit --cov-fail-under=80
# Exit if failed

# 6. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" conduit/ && exit 1

# 7. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" conduit/ tests/ && exit 1
```

### Required Tools

- **black** - Code formatting (line length 88)
- **ruff** - Fast linting
- **mypy** - Strict type checking
- **pytest** - Testing framework with pytest-asyncio

## Pull Request Process

### Before Submitting

1. **Update tests** - Add tests for new features or bug fixes
2. **Update documentation** - Update README.md if API changed
3. **Add examples** - Create example file in `examples/` for new features
4. **Update exports** - Add new classes to `__init__.py` files
5. **Run all checks** - Ensure all pre-commit checks pass

### PR Requirements

- **Clear description** - Explain what and why (not just what)
- **Reference issues** - Link to related issues
- **One feature per PR** - Keep PRs focused and reviewable
- **Passing CI** - All checks must pass
- **No merge conflicts** - Rebase on main if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New bandit algorithm
- [ ] Enhancement
- [ ] Documentation
- [ ] Performance optimization

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests pass
- [ ] Coverage >80%

## Bandit Algorithm Validation (if applicable)
- [ ] Implements BanditAlgorithm interface
- [ ] Async select_arm() and update() methods
- [ ] Tested with 387-dim feature vectors
- [ ] Handles multi-objective rewards (quality + cost + latency)

## Checklist
- [ ] Code follows style guidelines (black, ruff, mypy pass)
- [ ] Added/updated docstrings
- [ ] Updated README.md if needed
- [ ] Added example in examples/ if needed
- [ ] Updated __init__.py exports
- [ ] No credentials in code
```

## Issue Guidelines

### Bug Reports

Use the bug report template. Include:
- **Query type and text** - What were you routing?
- **Model selected** - Which model did the router choose?
- **Expected vs actual behavior**
- **Quality score, cost, and latency** - Observed metrics
- **Minimal reproduction code**
- **Error messages and stack traces**

### Feature Requests

Use the feature request template. Include:
- **Use case** - Why is this needed?
- **Proposed solution** - What should it do?
- **Impact on routing** - How does this affect model selection?
- **Alternatives considered** - What else did you consider?

### Questions

For questions about usage:
- Check README.md and examples/ first
- Search existing issues
- Check docs/ directory for architecture and algorithms
- Provide context about your routing scenario

## What We Won't Build

To set clear expectations:

- **Static rule-based routing** - Conduit is ML-driven, not IF/ELSE rules
- **Support for Python <3.10** - Modern type hints and async/await required
- **Hosted routing service** - Self-hosted only
- **Non-contextual routing** - Query features required for all algorithms

## Additional Resources

- **README.md** - User documentation and quickstart
- **AGENTS.md** - Detailed development guide (AI-focused)
- **examples/** - Usage examples organized by category
- **docs/ARCHITECTURE.md** - System architecture
- **docs/BANDIT_ALGORITHMS.md** - Algorithm details
- **docs/EMBEDDING_PROVIDERS.md** - Embedding configuration

## Questions?

- Open an issue for bugs or features
- Check existing issues and examples first
- Be specific and provide routing context

Thank you for contributing to Conduit!
