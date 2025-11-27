"""Regression tests for Conduit.

Regression tests ensure that existing functionality continues to work
across code changes. These tests:

1. Verify all example files execute without errors
2. Catch breaking changes in public APIs
3. Validate graceful degradation (missing API keys, optional dependencies)

Running regression tests:
    pytest tests/regression/ -v                 # Run all regression tests
    pytest -m regression                         # Run using pytest marker
    pytest tests/regression/ -k litellm         # Run only LiteLLM examples

Skipped tests:
    Tests are automatically skipped if:
    - Required API keys are not set (OPENAI_API_KEY, ANTHROPIC_API_KEY)
    - Optional dependencies are not installed (litellm, langchain)
"""
