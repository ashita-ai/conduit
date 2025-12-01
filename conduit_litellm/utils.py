"""Utility functions for Conduit LiteLLM plugin."""

from typing import Any, Optional

from conduit.core.config import load_litellm_config


def validate_litellm_model_list(model_list: list[dict[str, Any]]) -> None:
    """Validate LiteLLM model_list format.

    Validates that model_list is a non-empty list of dictionaries.
    Note: model_info.id is optional and will be auto-generated if missing.

    Args:
        model_list: LiteLLM router model_list configuration.

    Raises:
        ValueError: If model_list format is invalid.

    Example:
        >>> model_list = [
        ...     {
        ...         "model_name": "gpt-4",
        ...         "litellm_params": {"model": "openai/gpt-4"}
        ...     }
        ... ]
        >>> validate_litellm_model_list(model_list)
    """
    if not isinstance(model_list, list):
        raise ValueError("model_list must be a list")

    if not model_list:
        raise ValueError("model_list cannot be empty")

    for i, deployment in enumerate(model_list):
        if not isinstance(deployment, dict):
            raise ValueError(f"Deployment {i} must be a dictionary")

        # Require either model_name or litellm_params.model
        if "model_name" not in deployment and (
            "litellm_params" not in deployment
            or "model" not in deployment.get("litellm_params", {})
        ):
            raise ValueError(
                f"Deployment {i} must have 'model_name' or 'litellm_params.model'"
            )


def _generate_model_id(deployment: dict[str, Any], index: int) -> str:
    """Generate a model ID from deployment configuration.

    Tries to extract ID from:
    1. model_info.id (if present)
    2. model_info (if it's a string)
    3. model_name
    4. litellm_params.model (normalized)
    5. Fallback to "model-{index}"

    Args:
        deployment: Deployment dictionary from model_list.
        index: Index of deployment in model_list.

    Returns:
        Generated model ID string.
    """
    # Try model_info.id first
    if "model_info" in deployment:
        model_info = deployment["model_info"]
        if isinstance(model_info, dict) and "id" in model_info:
            return model_info["id"]
        if isinstance(model_info, str):
            return model_info

    # Try model_name
    if "model_name" in deployment:
        return normalize_model_name(deployment["model_name"])

    # Try litellm_params.model
    if "litellm_params" in deployment:
        litellm_params = deployment["litellm_params"]
        if isinstance(litellm_params, dict) and "model" in litellm_params:
            return normalize_model_name(litellm_params["model"])

    # Fallback to index-based ID
    return f"model-{index}"


def extract_model_ids(model_list: list[dict[str, Any]]) -> list[str]:
    """Extract model IDs from LiteLLM model_list.

    Auto-generates IDs if model_info.id is missing, making it compatible
    with standard LiteLLM model_list format.

    Args:
        model_list: LiteLLM router model_list configuration.

    Returns:
        List of model IDs (extracted or auto-generated).

    Example:
        >>> # Standard LiteLLM format (no model_info.id)
        >>> model_list = [
        ...     {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
        ...     {"model_name": "claude-3", "litellm_params": {"model": "anthropic/claude-3-opus"}}
        ... ]
        >>> extract_model_ids(model_list)
        ['gpt-4', 'claude-3']

        >>> # With explicit model_info.id
        >>> model_list = [
        ...     {"model_info": {"id": "gpt-4-openai"}},
        ...     {"model_info": {"id": "claude-3-opus"}}
        ... ]
        >>> extract_model_ids(model_list)
        ['gpt-4-openai', 'claude-3-opus']
    """
    validate_litellm_model_list(model_list)
    return [
        _generate_model_id(deployment, i) for i, deployment in enumerate(model_list)
    ]


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to Conduit format.

    Args:
        model_name: Model name in various formats.

    Returns:
        Normalized model name.

    Example:
        >>> normalize_model_name("openai/gpt-4")
        'gpt-4'
        >>> normalize_model_name("anthropic/claude-3-opus")
        'claude-3-opus'
    """
    # Remove provider prefix if present
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def map_litellm_to_conduit(litellm_model: str) -> str:
    """Map LiteLLM model ID to Conduit model ID.

    Handles common LiteLLM model names and translates them to Conduit's
    standardized model IDs used in conduit.yaml priors. Loads mappings from
    conduit.yaml litellm.model_mappings configuration.

    Args:
        litellm_model: Model ID from LiteLLM (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")

    Returns:
        Conduit model ID (e.g., "o4-mini", "claude-haiku-4.5")

    Example:
        >>> map_litellm_to_conduit("gpt-4o-mini-2024-07-18")
        'o4-mini'
        >>> map_litellm_to_conduit("claude-3-5-sonnet-20241022")
        'claude-sonnet-4.5'
        >>> map_litellm_to_conduit("unknown-model")
        'unknown-model'
    """
    # Remove provider prefix if present
    model_name = normalize_model_name(litellm_model)

    # Load mapping from config
    config = load_litellm_config()
    mappings = config.get("model_mappings", {})

    # Check direct mapping
    if model_name in mappings:
        return mappings[model_name]

    # No mapping found - return as-is
    return model_name


def format_routing_metadata(
    selected_model: str,
    confidence: float,
    reasoning: str,
    features: Any
) -> dict[str, Any]:
    """Format Conduit routing decision metadata for LiteLLM.

    Args:
        selected_model: Model ID selected by Conduit.
        confidence: Routing confidence score (0-1).
        reasoning: Explanation for selection.
        features: Query features used for routing.

    Returns:
        Metadata dictionary for logging/monitoring.

    Example:
        >>> metadata = format_routing_metadata(
        ...     selected_model="gpt-4-openai",
        ...     confidence=0.85,
        ...     reasoning="High quality model for complex query",
        ...     features=query_features
        ... )
    """
    return {
        "conduit_selected_model": selected_model,
        "conduit_confidence": confidence,
        "conduit_reasoning": reasoning,
        "conduit_complexity": getattr(features, "complexity_score", None),
        "conduit_domain": getattr(features, "domain", None),
    }


def extract_query_text(
    messages: Optional[list[dict[str, Any]]] = None,
    input_data: Any = None,
) -> str:
    """Extract query text from LiteLLM messages or input format.

    Extracts the query text from either messages (OpenAI format) or input_data
    (completion API format) for feature regeneration in feedback callbacks.

    Args:
        messages: LiteLLM messages list (OpenAI chat format).
        input_data: LiteLLM input string or list (completion format).

    Returns:
        Query text extracted from messages or input, empty string if not found.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ]
        >>> extract_query_text(messages=messages)
        'What is 2+2?'
        >>> extract_query_text(input_data="Direct prompt")
        'Direct prompt'
    """
    # Try messages first (chat format)
    if messages:
        # Find the last user message
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue

            role = message.get("role", "")
            if role == "user":
                content = message.get("content", "")
                # Handle string content
                if isinstance(content, str):
                    return content
                # Handle list content (e.g., multimodal messages)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
                        if isinstance(part, str):
                            return part
                break  # User message found but no valid content

    # Try input_data (completion format)
    if input_data is not None:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, list):
            # Join list elements
            return " ".join(str(item) for item in input_data)

    return ""
