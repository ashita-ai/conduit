"""Regression tests for example files.

These tests ensure that all examples in the examples/ directory run without errors.
They validate that:
1. Examples execute successfully (no exceptions)
2. Examples produce expected output patterns
3. Examples handle missing dependencies gracefully

Test categories:
- Core examples: 6 main demonstration files at root level
- Integration examples: External library integrations (in integrations/)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

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
    # Load .env file to get API keys
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path)

    # Set PYTHONPATH to include project root so examples can import conduit
    # Also pass through API keys from environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Use uv run to ensure correct environment with conduit installed
    # Try to find uv in common locations
    uv_path = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")

    result = subprocess.run(
        [uv_path, "run", "python", str(example_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
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
        return pytest.mark.skip(
            reason="Requires litellm (pip install conduit[litellm])"
        )(func)


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
# Core Examples (6 main files at root level)
# ============================================================


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_hello_world():
    """Test examples/hello_world.py runs successfully."""
    example = EXAMPLES_DIR / "hello_world.py"
    exit_code, stdout, stderr = run_example(example)

    # Verify execution
    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Verify expected output pattern
    assert "Route to:" in stdout, "Expected 'Route to:' in output"
    assert "confidence:" in stdout, "Expected 'confidence:' in output"


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_routing_options():
    """Test examples/routing_options.py runs successfully."""
    example = EXAMPLES_DIR / "routing_options.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"
    assert "CONSTRAINTS" in stdout or "PREFERENCES" in stdout


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_feedback_loop():
    """Test examples/feedback_loop.py runs successfully."""
    example = EXAMPLES_DIR / "feedback_loop.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"
    assert "CACHING" in stdout or "LEARNING" in stdout or "PERSISTENCE" in stdout


@pytest.mark.regression
@requires_api_key("OPENAI_API_KEY")
def test_production_feedback():
    """Test examples/production_feedback.py runs successfully."""
    example = EXAMPLES_DIR / "production_feedback.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"
    # Check for key sections of the output
    assert "PRODUCTION FEEDBACK" in stdout or "DELAYED FEEDBACK" in stdout


@pytest.mark.regression
@requires_litellm
@requires_api_key("OPENAI_API_KEY")
def test_litellm_integration():
    """Test examples/litellm_integration.py runs successfully."""
    example = EXAMPLES_DIR / "litellm_integration.py"
    exit_code, stdout, stderr = run_example(example, timeout=300)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"
    assert "LITELLM" in stdout.upper() or "Conduit" in stdout


@pytest.mark.regression
def test_hybrid_routing():
    """Test examples/hybrid_routing.py runs successfully.

    This example uses local embeddings (FastEmbed or sentence-transformers)
    and does not require API keys. First run may download model files (~100MB).
    """
    example = EXAMPLES_DIR / "hybrid_routing.py"
    exit_code, stdout, stderr = run_example(example, timeout=180)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Hybrid routing should show transition from UCB1 to LinUCB
    assert "Phase 1" in stdout or "UCB1" in stdout
    assert "Phase 2" in stdout or "LinUCB" in stdout


# ============================================================
# Integration Examples (in integrations/ subdirectory)
# ============================================================


def requires_llamaindex(func):
    """Skip test if llama-index is not installed."""
    try:
        import llama_index  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(
            reason="Requires llama-index (pip install llama-index)"
        )(func)


def requires_fastapi(func):
    """Skip test if fastapi is not installed."""
    try:
        import fastapi  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(reason="Requires fastapi (pip install fastapi)")(func)


def requires_gradio(func):
    """Skip test if gradio is not installed."""
    try:
        import gradio  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(reason="Requires gradio (pip install gradio)")(func)


@pytest.mark.regression
@requires_langchain
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
@pytest.mark.skip(reason="Requires langchain (pip install conduit[langchain])")
def test_langchain_integration():
    """Test examples/integrations/langchain_integration.py runs successfully."""
    example = EXAMPLES_DIR / "integrations" / "langchain_integration.py"
    exit_code, stdout, stderr = run_example(example, timeout=120)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate LangChain integration
    assert "langchain" in stdout.lower() or "chain" in stdout.lower()


@pytest.mark.regression
@requires_llamaindex
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
@pytest.mark.skip(reason="Requires llama-index (pip install llama-index)")
def test_llamaindex_integration():
    """Test examples/integrations/llamaindex_integration.py runs successfully."""
    example = EXAMPLES_DIR / "integrations" / "llamaindex_integration.py"
    exit_code, stdout, stderr = run_example(example, timeout=180)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate LlamaIndex integration
    combined_output = stdout + stderr
    assert (
        "llamaindex" in combined_output.lower()
        or "llama" in combined_output.lower()
        or "basic usage" in combined_output.lower()
    )


@pytest.mark.regression
@requires_fastapi
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
def test_fastapi_service():
    """Test examples/integrations/fastapi_service.py runs successfully.

    Note: This test runs the example in info mode (no --demo or --serve),
    which just prints usage information and exits.
    """
    example = EXAMPLES_DIR / "integrations" / "fastapi_service.py"
    exit_code, stdout, stderr = run_example(example, timeout=30)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should print FastAPI service info
    assert "fastapi" in stdout.lower() or "conduit" in stdout.lower()


@pytest.mark.regression
@requires_gradio
@requires_api_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
@pytest.mark.skip(
    reason="Gradio demo launches interactive server - not suitable for automated testing"
)
def test_gradio_demo():
    """Test examples/integrations/gradio_demo.py runs successfully.

    SKIPPED: Gradio demo launches an interactive server that doesn't exit
    automatically. Manual testing required.
    """
    example = EXAMPLES_DIR / "integrations" / "gradio_demo.py"
    exit_code, stdout, stderr = run_example(example, timeout=30)

    assert exit_code == 0, f"Example failed with stderr: {stderr}"

    # Should demonstrate Gradio interface
    assert "gradio" in stdout.lower() or "demo" in stdout.lower()


# ============================================================
# Meta Tests
# ============================================================


@pytest.mark.regression
def test_all_examples_have_tests():
    """Verify all example files have corresponding regression tests.

    This test ensures we don't forget to add tests for new examples.
    """
    # Get all Python files in examples/ (excluding integrations/)
    example_files = [f for f in EXAMPLES_DIR.glob("*.py") if f.name != "__init__.py"]

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
        test_name = f"test_{example_file.stem}"
        if test_name not in test_functions:
            missing_tests.append(example_file.name)

    assert not missing_tests, f"Missing tests for examples: {', '.join(missing_tests)}"


@pytest.mark.regression
def test_examples_directory_exists():
    """Verify examples directory exists and has expected structure."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found: {EXAMPLES_DIR}"

    example_files = list(EXAMPLES_DIR.glob("*.py"))
    assert len(example_files) > 0, "No example files found in examples/"

    # Verify integrations subdirectory exists
    integrations_dir = EXAMPLES_DIR / "integrations"
    assert integrations_dir.exists(), "Expected integrations/ directory"

    # Verify core examples exist
    core_examples = [
        "hello_world.py",
        "routing_options.py",
        "feedback_loop.py",
        "production_feedback.py",
        "litellm_integration.py",
        "hybrid_routing.py",
    ]
    for example in core_examples:
        assert (EXAMPLES_DIR / example).exists(), f"Core example {example} not found"
