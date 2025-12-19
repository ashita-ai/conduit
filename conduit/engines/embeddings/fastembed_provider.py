"""FastEmbed embedding provider (lightweight ONNX Runtime).

FastEmbed uses ONNX Runtime instead of PyTorch, making it significantly lighter
(~100MB vs ~2GB) while maintaining competitive performance.

Advantages:
- Lightweight: No PyTorch dependencies
- Fast: ONNX Runtime optimization
- Serverless-friendly: Works in Lambda/containers
- Same models: Supports bge-small, all-MiniLM, etc.

Installation: pip install fastembed
"""

import logging
from typing import TYPE_CHECKING

from conduit.engines.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """FastEmbed embedding provider using ONNX Runtime.

    Lighter weight alternative to sentence-transformers.
    Uses ONNX Runtime (~100MB) instead of PyTorch (~2GB).

    Default model: BAAI/bge-small-en-v1.5 (384 dims)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ):
        """Initialize FastEmbed provider.

        Args:
            model: FastEmbed model identifier
                   Options: "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
            cache_dir: Optional cache directory for model files

        Raises:
            ImportError: If fastembed not installed
            RuntimeError: If model loading fails

        Example:
            >>> provider = FastEmbedProvider()
            >>> embedding = await provider.embed("Hello world")
            >>> len(embedding)
            384
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed not installed. Install with: pip install fastembed"
            ) from e

        self.model_name = model
        self._dimension = self._get_dimension_for_model(model)

        try:
            # Initialize FastEmbed model
            self.model: TextEmbedding = TextEmbedding(
                model_name=model,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error(f"Failed to load FastEmbed model {model}: {e}")
            raise RuntimeError(f"Failed to load FastEmbed model: {e}") from e

    def _get_dimension_for_model(self, model: str) -> int:
        """Get embedding dimension for known models.

        Args:
            model: Model identifier

        Returns:
            Embedding dimension

        Note:
            Returns 384 as default for unknown models (common for small models)
        """
        # Map of known models to dimensions
        known_dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
        }

        return known_dimensions.get(model, 384)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If embedding generation fails

        Example:
            >>> embedding = await provider.embed("What is photosynthesis?")
            >>> len(embedding)
            384
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

        Note:
            FastEmbed processes in batches internally for efficiency.
        """
        if not texts:
            return []

        try:
            # FastEmbed returns generator of numpy arrays
            # Convert to list of lists
            embeddings_generator = self.model.embed(texts)
            embeddings = [emb.tolist() for emb in embeddings_generator]

            return embeddings

        except Exception as e:
            logger.error(f"FastEmbed embedding failed: {e}")
            raise RuntimeError(f"FastEmbed embedding generation failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "fastembed"
