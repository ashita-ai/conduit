"""Evaluation module for measuring routing quality and performance.

The ArbiterEvaluator requires the 'arbiter' package to be installed.
Install with: pip install conduit[arbiter] or pip install arbiter
"""

from typing import TYPE_CHECKING

# Lazy import to avoid requiring arbiter for basic usage
if TYPE_CHECKING:
    from conduit.evaluation.arbiter_evaluator import ArbiterEvaluator

__all__ = ["ArbiterEvaluator"]


def __getattr__(name: str):
    """Lazy import ArbiterEvaluator only when accessed."""
    if name == "ArbiterEvaluator":
        try:
            from conduit.evaluation.arbiter_evaluator import ArbiterEvaluator

            return ArbiterEvaluator
        except ImportError as e:
            raise ImportError(
                "ArbiterEvaluator requires the 'arbiter' package. "
                "Install with: pip install conduit[arbiter] or pip install arbiter"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
