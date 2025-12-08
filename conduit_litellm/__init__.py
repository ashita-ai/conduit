"""Conduit LiteLLM Plugin - ML-powered routing strategy for LiteLLM.

This plugin integrates Conduit's ML-based routing algorithms with LiteLLM's
router, enabling intelligent model selection across 100+ LLM providers.

The feedback loop (ConduitFeedbackLogger) is automatically initialized when using
ConduitRoutingStrategy, enabling bandit learning from LiteLLM response metadata.

Usage:
    from litellm import Router
    from conduit_litellm import ConduitRoutingStrategy

    router = Router(model_list=[...])
    strategy = ConduitRoutingStrategy()  # Hybrid routing always enabled
    ConduitRoutingStrategy.setup_strategy(router, strategy)
"""

from conduit_litellm.feedback import ConduitFeedbackLogger
from conduit_litellm.model_registry import (
    ModelRegistry,
    get_global_registry,
    normalize_model_name,
)
from conduit_litellm.strategy import ConduitRoutingStrategy
from conduit_litellm.utils import map_litellm_to_conduit

__version__ = "0.1.0"
__all__ = [
    "ConduitRoutingStrategy",
    "ConduitFeedbackLogger",
    "ModelRegistry",
    "get_global_registry",
    "normalize_model_name",
    "map_litellm_to_conduit",
]
