"""Factory for creating embedding providers with intelligent fallback."""

import logging
import os
from typing import Any

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider
from conduit.engines.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)

logger = logging.getLogger(__name__)


def _try_import_fastembed() -> bool:
    """Check if fastembed is available."""
    try:
        import fastembed  # noqa: F401

        return True
    except ImportError:
        return False


def _try_import_sentence_transformers() -> bool:
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def create_embedding_provider(
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create embedding provider with intelligent fallback.

    Priority order for 'auto' mode:
        1. OpenAI embeddings (if OPENAI_API_KEY exists) - Fast, high quality
        2. Cohere embeddings (if COHERE_API_KEY exists) - Fast, high quality
        3. FastEmbed (if installed) - Lightweight ONNX (~100MB), no API key
        4. Sentence-transformers (if installed) - Full PyTorch (~2GB), no API key
        5. Error with clear installation instructions

    Args:
        provider: Provider type ("auto", "openai", "cohere", "fastembed",
                 "sentence-transformers", "huggingface"). Default: "auto"
        model: Model identifier (provider-specific, optional)
        api_key: API key (if required by provider, optional)
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider type is invalid
        ImportError: If required package not installed
        RuntimeError: If 'auto' mode finds no available providers

    Example:
        >>> # Automatic fallback (recommended)
        >>> provider = create_embedding_provider("auto")
        >>>
        >>> # Explicit provider with API key
        >>> provider = create_embedding_provider(
        ...     "openai",
        ...     model="text-embedding-3-small",
        ...     api_key="sk-..."
        ... )
        >>>
        >>> # Local provider (no API key needed)
        >>> provider = create_embedding_provider("fastembed")
    """
    provider_lower = provider.lower()

    # Auto mode: intelligent fallback chain
    if provider_lower == "auto":
        # Try OpenAI first (fast, high quality)
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Auto mode: Using OpenAI embeddings (API key found)")
            return create_embedding_provider(
                "openai", model=model, api_key=api_key, **kwargs
            )

        # Try Cohere second (fast, high quality)
        if os.getenv("COHERE_API_KEY"):
            logger.info("Auto mode: Using Cohere embeddings (API key found)")
            return create_embedding_provider(
                "cohere", model=model, api_key=api_key, **kwargs
            )

        # Try FastEmbed third (lightweight, no API key)
        if _try_import_fastembed():
            logger.info(
                "Auto mode: Using FastEmbed (lightweight ONNX, no API key needed)"
            )
            return create_embedding_provider(
                "fastembed", model=model, api_key=api_key, **kwargs
            )

        # Try sentence-transformers last (full PyTorch, no API key)
        if _try_import_sentence_transformers():
            logger.info(
                "Auto mode: Using sentence-transformers (PyTorch, no API key needed)"
            )
            return create_embedding_provider(
                "sentence-transformers", model=model, api_key=api_key, **kwargs
            )

        # No providers available
        raise RuntimeError(
            "No embedding providers available. Install one of:\n"
            "  pip install openai  # Fast API-based (requires OPENAI_API_KEY)\n"
            "  pip install cohere  # Fast API-based (requires COHERE_API_KEY)\n"
            "  pip install fastembed  # Lightweight local (~100MB)\n"
            "  pip install sentence-transformers  # Full local (~2GB)"
        )

    # Explicit provider selection
    if provider_lower == "huggingface":
        # Legacy HuggingFace API (requires API key)
        model = model or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
        hf_api_key = (
            api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        )
        return HuggingFaceEmbeddingProvider(
            model=model,
            api_key=hf_api_key,
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

    elif provider_lower == "fastembed":
        from conduit.engines.embeddings.fastembed_provider import FastEmbedProvider

        model = model or "BAAI/bge-small-en-v1.5"
        return FastEmbedProvider(
            model=model,
            cache_dir=kwargs.get("cache_dir"),
        )

    elif provider_lower in ("sentence-transformers", "sentence_transformers"):
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformersEmbeddingProvider(model=model)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: auto, openai, cohere, fastembed, sentence-transformers, huggingface"
        )
