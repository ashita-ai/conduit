"""Sentence-transformers embedding provider (optional, for offline use)."""

import asyncio
import logging
from typing import Optional

from conduit.engines.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped,unused-ignore]
except ImportError:
    SentenceTransformer = None  # type: ignore


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence-transformers embedding provider.

    Optional provider for offline use. Requires sentence-transformers package.
    Useful when you want to avoid API calls or work offline.

    Default model: all-MiniLM-L6-v2 (384 dims)
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize sentence-transformers embedding provider.

        Args:
            model: HuggingFace model identifier (default: all-MiniLM-L6-v2)

        Raises:
            ImportError: If sentence-transformers package not installed

        Example:
            >>> provider = SentenceTransformersEmbeddingProvider()
            >>> embedding = await provider.embed("Hello world")
            >>> len(embedding)
            384
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package required for SentenceTransformersEmbeddingProvider. "
                "Install with: pip install sentence-transformers"
            )

        self.model = model
        self.embedder = SentenceTransformer(model)

        # Determine dimension based on model
        if "all-MiniLM-L6-v2" in model:
            self._dimension = 384
        elif "all-mpnet-base-v2" in model:
            self._dimension = 768
        else:
            # Try to get dimension from model
            try:
                # Encode a test string to get dimension
                test_emb = self.embedder.encode("test", convert_to_numpy=True)
                self._dimension = len(test_emb)
            except Exception:
                # Fallback to common dimension
                self._dimension = 384

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If embedding generation fails
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Offload CPU work to thread pool to avoid blocking event loop
            embeddings = await asyncio.to_thread(self.embedder.encode, texts)

            # Convert numpy arrays to lists
            result: list[list[float]] = []
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    result.append(emb.tolist())
                else:
                    result.append(list(emb))

            return result

        except Exception as e:
            logger.error(f"Sentence-transformers embedding error: {e}")
            raise RuntimeError(f"Sentence-transformers embedding failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "sentence-transformers"

