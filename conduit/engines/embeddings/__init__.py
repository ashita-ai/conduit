"""Embedding providers for query feature extraction.

Supports multiple embedding backends with automatic fallback:
- Auto mode (default): Tries OpenAI → Cohere → FastEmbed → sentence-transformers
- OpenAI embeddings (fast API, requires OPENAI_API_KEY)
- Cohere embeddings (fast API, requires COHERE_API_KEY)
- FastEmbed (lightweight ONNX ~100MB, no API key)
- Sentence-transformers (full PyTorch ~2GB, no API key)
- HuggingFace API (legacy, requires API key)
"""

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.factory import create_embedding_provider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider
from conduit.engines.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)

try:
    from conduit.engines.embeddings.fastembed_provider import FastEmbedProvider

    __all__ = [
        "EmbeddingProvider",
        "HuggingFaceEmbeddingProvider",
        "OpenAIEmbeddingProvider",
        "CohereEmbeddingProvider",
        "SentenceTransformersEmbeddingProvider",
        "FastEmbedProvider",
        "create_embedding_provider",
    ]
except ImportError:
    # FastEmbed is optional
    __all__ = [
        "EmbeddingProvider",
        "HuggingFaceEmbeddingProvider",
        "OpenAIEmbeddingProvider",
        "CohereEmbeddingProvider",
        "SentenceTransformersEmbeddingProvider",
        "create_embedding_provider",
    ]
