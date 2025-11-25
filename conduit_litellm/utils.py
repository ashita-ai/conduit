"""Utility functions for Conduit LiteLLM plugin."""

from typing import Any, Optional


def validate_litellm_model_list(model_list: list[dict[str, Any]]) -> None:
    """Validate LiteLLM model_list format.

    Args:
        model_list: LiteLLM router model_list configuration.

    Raises:
        ValueError: If model_list format is invalid.

    Example:
        >>> model_list = [
        ...     {
        ...         "model_name": "gpt-4",
        ...         "litellm_params": {"model": "openai/gpt-4"},
        ...         "model_info": {"id": "gpt-4-openai"}
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

        if "model_info" not in deployment:
            raise ValueError(f"Deployment {i} missing 'model_info'")

        if "id" not in deployment["model_info"]:
            raise ValueError(f"Deployment {i} missing 'model_info.id'")


def extract_model_ids(model_list: list[dict[str, Any]]) -> list[str]:
    """Extract model IDs from LiteLLM model_list.

    Args:
        model_list: LiteLLM router model_list configuration.

    Returns:
        List of model IDs.

    Example:
        >>> model_list = [
        ...     {"model_info": {"id": "gpt-4-openai"}},
        ...     {"model_info": {"id": "claude-3-opus"}}
        ... ]
        >>> extract_model_ids(model_list)
        ['gpt-4-openai', 'claude-3-opus']
    """
    validate_litellm_model_list(model_list)
    return [deployment["model_info"]["id"] for deployment in model_list]


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
