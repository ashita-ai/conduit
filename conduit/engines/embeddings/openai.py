"""OpenAI embedding provider (recommended for production)."""

import logging
from typing import Optional

from conduit.engines.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider.

    Recommended for production use. Provides high-quality embeddings
    with consistent performance.

    Default model: text-embedding-3-small (1536 dims, can be reduced)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model (default: text-embedding-3-small)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            dimensions: Optional dimension reduction (for text-embedding-3-* models)
            timeout: Request timeout in seconds (default: 30s)

        Raises:
            ImportError: If openai package not installed
            ValueError: If API key not provided

        Example:
            >>> provider = OpenAIEmbeddingProvider(api_key="sk-...")
            >>> embedding = await provider.embed("Hello world")
            >>> len(embedding)
            1536
        """
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package required for OpenAIEmbeddingProvider. "
                "Install with: pip install openai"
            )

        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            import os

            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
                )

        # Initialize client
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)

        # Determine dimension based on model
        if "text-embedding-3" in model:
            # text-embedding-3 models support dimension reduction
            self._dimension = dimensions or 1536
        elif "text-embedding-ada-002" in model:
            self._dimension = 1536
        else:
            # Default fallback
            self._dimension = dimensions or 1536

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If API request fails
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
            RuntimeError: If API request fails
        """
        if not texts:
            return []

        try:
            # Build request parameters
            params: dict[str, any] = {
                "model": self.model,
                "input": texts,
            }

            # Add dimension reduction if specified
            if self.dimensions and "text-embedding-3" in self.model:
                params["dimensions"] = self.dimensions

            # Call OpenAI API
            response = await self.client.embeddings.create(**params)

            # Extract embeddings
            result: list[list[float]] = []
            for item in response.data:
                result.append(item.embedding)

            return result

        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            raise RuntimeError(f"OpenAI embedding API failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "openai"

