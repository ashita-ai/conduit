"""Embedding providers for query feature extraction.

Supports multiple embedding backends:
- HuggingFace Inference API (free default, no API key needed)
- OpenAI embeddings (recommended for production)
- Cohere embeddings (recommended for production)
- Sentence-transformers (optional, for offline use)
"""

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.factory import create_embedding_provider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider
from conduit.engines.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "SentenceTransformersEmbeddingProvider",
    "create_embedding_provider",
]

