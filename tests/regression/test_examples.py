"""Regression tests for all example files.

These tests ensure that all examples in the examples/ directory run without errors.
They validate that:
1. Examples execute successfully (no exceptions)
2. Examples produce expected output patterns
3. Examples handle missing dependencies gracefully

Test categories:
- Quickstart examples: Basic router usage
- Routing examples: Different routing strategies
- Optimization examples: Caching, state persistence
- LiteLLM examples: Integration with LiteLLM router
- Personalization examples: User preferences
- Integration examples: External library integrations
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def run_example(example_path: Path, timeout: int = 30) -> tuple[int, str, str]:
    """Run an example script and return exit code, stdout, stderr.

    Args:
        example_path: Path to example Python file
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    # Use uv run to ensure correct environment with conduit installed
    result = subprocess.run(
        ["uv", "run", "python", str(example_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def requires_api_key(*keys: str):
    """Skip test if required API keys are not set."""

    def decorator(func):
        missing = [key for key in keys if not os.getenv(key)]
        skip_reason = f"Requires API key(s): {', '.join(missing)}"
        return pytest.mark.skipif(bool(missing), reason=skip_reason)(func)

    return decorator


def requires_litellm(func):
    """Skip test if litellm is not installed."""
    try:
        import litellm  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(reason="Requires litellm (pip install conduit[litellm])")(
            func
        )


def requires_langchain(func):
    """Skip test if langchain is not installed."""
    try:
        import langchain  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(
            reason="Requires langchain (pip install conduit[langchain])"
        )(func)


# ============================================================
# Quickstart Examples (2 files)
# ============================================================


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_hello_world():
    """Test examples/01_quickstart/hello_world.py runs successfully."""
    example = EXAMPLES_DIR / "01_quickstart" / "hello_world.py"
    exit_code, stdout, stderr = run_example(example)

    # Verify execution
    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Verify expected output pattern
    assert "Route to:" in stdout, "Expected 'Route to:' in output"
    assert "confidence:" in stdout, "Expected 'confidence:' in output"


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_simple_router():
    """Test examples/01_quickstart/simple_router.py runs successfully."""
    example = EXAMPLES_DIR / "01_quickstart" / "simple_router.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # simple_router.py should have similar output to hello_world.py
    assert "Route to:" in stdout or "Selected Model:" in stdout


# ============================================================
# Routing Examples (4 files)
# ============================================================


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_basic_routing():
    """Test examples/02_routing/basic_routing.py runs successfully."""
    example = EXAMPLES_DIR / "02_routing" / "basic_routing.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Verify routing output structure
    assert "Routing Results" in stdout or "Selected Model" in stdout
    assert "Confidence:" in stdout or "confidence:" in stdout


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_context_specific_priors():
    """Test examples/02_routing/context_specific_priors.py runs successfully."""
    example = EXAMPLES_DIR / "02_routing" / "context_specific_priors.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate domain-specific routing
    assert "Route to:" in stdout or "Selected Model" in stdout


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_hybrid_routing():
    """Test examples/02_routing/hybrid_routing.py runs successfully."""
    example = EXAMPLES_DIR / "02_routing" / "hybrid_routing.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Hybrid routing should show transition from UCB1 to LinUCB
    assert "Route to:" in stdout or "Selected" in stdout


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_with_constraints():
    """Test examples/02_routing/with_constraints.py runs successfully."""
    example = EXAMPLES_DIR / "02_routing" / "with_constraints.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should respect constraints in routing
    assert "Route to:" in stdout or "Selected" in stdout


# ============================================================
# Optimization Examples (3 files)
# ============================================================


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_caching():
    """Test examples/03_optimization/caching.py runs successfully."""
    example = EXAMPLES_DIR / "03_optimization" / "caching.py"
    exit_code, stdout, stderr = run_example(example, timeout=60)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate cache speedup
    assert "cache" in stdout.lower() or "cached" in stdout.lower()


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_pca_comparison():
    """Test examples/03_optimization/pca_comparison.py runs successfully."""
    example = EXAMPLES_DIR / "03_optimization" / "pca_comparison.py"
    exit_code, stdout, stderr = run_example(example, timeout=60)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should compare PCA vs full features
    assert "PCA" in stdout or "dimensionality" in stdout.lower()


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_state_persistence():
    """Test examples/03_optimization/state_persistence.py runs successfully."""
    example = EXAMPLES_DIR / "03_optimization" / "state_persistence.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate state save/load
    assert "state" in stdout.lower() or "persist" in stdout.lower()


# ============================================================
# LiteLLM Examples (5 files)
# ============================================================


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY")
def test_basic_usage():
    """Test examples/04_litellm/basic_usage.py runs successfully."""
    example = EXAMPLES_DIR / "04_litellm" / "basic_usage.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should show LiteLLM + Conduit integration
    assert "Conduit" in stdout and ("routing" in stdout.lower() or "selected" in stdout.lower())


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY")
def test_custom_config():
    """Test examples/04_litellm/custom_config.py runs successfully."""
    example = EXAMPLES_DIR / "04_litellm" / "custom_config.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate custom configuration
    assert "config" in stdout.lower() or "custom" in stdout.lower()


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY")
def test_learning_demo():
    """Test examples/04_litellm/learning_demo.py runs successfully."""
    example = EXAMPLES_DIR / "04_litellm" / "learning_demo.py"
    exit_code, stdout, stderr = run_example(example, timeout=180)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate learning over time
    assert "learn" in stdout.lower() or "feedback" in stdout.lower()


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_multi_provider():
    """Test examples/04_litellm/multi_provider.py runs successfully."""
    example = EXAMPLES_DIR / "04_litellm" / "multi_provider.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should use multiple providers
    assert "provider" in stdout.lower() or "multi" in stdout.lower()


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY")
def test_arbiter_quality_measurement():
    """Test examples/04_litellm/arbiter_quality_measurement.py runs successfully."""
    example = EXAMPLES_DIR / "04_litellm" / "arbiter_quality_measurement.py"
    exit_code, stdout, stderr = run_example(example, timeout=180)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate Arbiter quality evaluation
    assert "arbiter" in stdout.lower() or "quality" in stdout.lower()


# ============================================================
# Personalization Examples (1 file)
# ============================================================


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_explicit_preferences():
    """Test examples/05_personalization/explicit_preferences.py runs successfully."""
    example = EXAMPLES_DIR / "05_personalization" / "explicit_preferences.py"
    exit_code, stdout, stderr = run_example(example)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate user preferences
    assert "preference" in stdout.lower() or "optimize" in stdout.lower()


# ============================================================
# Integration Examples (1 file)
# ============================================================


@pytest.mark.regression
@requires_langchain
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_langchain_integration():
    """Test examples/06_integrations/langchain_integration.py runs successfully."""
    example = EXAMPLES_DIR / "06_integrations" / "langchain_integration.py"
    exit_code, stdout, stderr = run_example(example, timeout=60)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate LangChain integration
    assert "langchain" in stdout.lower() or "chain" in stdout.lower()


# ============================================================
# Meta Tests
# ============================================================


@pytest.mark.regression
def test_all_examples_have_tests():
    """Verify all example files have corresponding regression tests.

    This test ensures we don't forget to add tests for new examples.
    """
    example_files = sorted(EXAMPLES_DIR.rglob("*.py"))

    # Get all test functions in this module
    test_module = sys.modules[__name__]
    test_functions = {
        name
        for name in dir(test_module)
        if name.startswith("test_") and name != "test_all_examples_have_tests"
    }

    # Verify each example has a test
    missing_tests = []
    for example_file in example_files:
        # Convert path to test name (e.g., "01_quickstart/hello_world.py" -> "test_hello_world")
        relative_path = example_file.relative_to(EXAMPLES_DIR)
        parts = relative_path.parts
        test_name = f"test_{parts[-1].replace('.py', '')}"

        if test_name not in test_functions:
            missing_tests.append(str(relative_path))

    assert not missing_tests, f"Missing tests for examples: {', '.join(missing_tests)}"


@pytest.mark.regression
def test_examples_directory_exists():
    """Verify examples directory exists and is not empty."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found: {EXAMPLES_DIR}"

    example_files = list(EXAMPLES_DIR.rglob("*.py"))
    assert len(example_files) > 0, "No example files found in examples/"

    # Expected structure
    expected_dirs = [
        "01_quickstart",
        "02_routing",
        "03_optimization",
        "04_litellm",
        "05_personalization",
        "06_integrations",
    ]

    for expected_dir in expected_dirs:
        dir_path = EXAMPLES_DIR / expected_dir
        assert dir_path.exists(), f"Expected directory not found: {expected_dir}"
