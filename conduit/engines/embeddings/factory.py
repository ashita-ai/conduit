"""Factory for creating embedding providers."""

import logging
from typing import Any, Optional

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider
from conduit.engines.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)

logger = logging.getLogger(__name__)


def create_embedding_provider(
    provider: str = "huggingface",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create embedding provider based on configuration.

    Args:
        provider: Provider type ("huggingface", "openai", "cohere", "sentence-transformers")
        model: Model identifier (provider-specific)
        api_key: API key (if required by provider)
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider type is invalid
        ImportError: If required package not installed

    Example:
        >>> # Free default (no API key needed)
        >>> provider = create_embedding_provider("huggingface")
        >>>
        >>> # OpenAI (requires API key)
        >>> provider = create_embedding_provider(
        ...     "openai",
        ...     model="text-embedding-3-small",
        ...     api_key="sk-..."
        ... )
        >>>
        >>> # Cohere (requires API key)
        >>> provider = create_embedding_provider(
        ...     "cohere",
        ...     api_key="..."
        ... )
    """
    provider_lower = provider.lower()

    if provider_lower == "huggingface":
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddingProvider(
            model=model,
            api_key=api_key,
            timeout=kwargs.get("timeout", 30.0),
        )

    elif provider_lower == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(
            model=model,
            api_key=api_key,
            dimensions=kwargs.get("dimensions"),
            timeout=kwargs.get("timeout", 30.0),
        )

    elif provider_lower == "cohere":
        model = model or "embed-english-v3.0"
        return CohereEmbeddingProvider(
            model=model,
            api_key=api_key,
            input_type=kwargs.get("input_type", "search_query"),
            timeout=kwargs.get("timeout", 30.0),
        )

    elif provider_lower in ("sentence-transformers", "sentence_transformers"):
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformersEmbeddingProvider(model=model)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: huggingface, openai, cohere, sentence-transformers"
        )

