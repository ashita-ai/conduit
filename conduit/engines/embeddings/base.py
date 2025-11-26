"""Base embedding provider interface."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different backends (API, local, etc.).
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension for this provider.

        Returns:
            Number of dimensions in embedding vector
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for logging/debugging.

        Returns:
            Provider identifier (e.g., "huggingface", "openai", "cohere")
        """
        pass
