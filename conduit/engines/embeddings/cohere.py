"""Cohere embedding provider (recommended for production)."""

import logging

import httpx

from conduit.engines.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider.

    Recommended for production use. Provides high-quality embeddings
    optimized for semantic search and retrieval.

    Default model: embed-english-v3.0 (1024 dims)
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_query",
        timeout: float = 30.0,
    ):
        """Initialize Cohere embedding provider.

        Args:
            model: Cohere embedding model (default: embed-english-v3.0)
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            input_type: Input type for embeddings ("search_query" or "search_document")
            timeout: Request timeout in seconds (default: 30s)

        Raises:
            ValueError: If API key not provided

        Example:
            >>> provider = CohereEmbeddingProvider(api_key="...")
            >>> embedding = await provider.embed("Hello world")
            >>> len(embedding)
            1024
        """
        self.model = model
        self.input_type = input_type
        self.timeout = timeout

        # Get API key from parameter or environment
        import os

        resolved_key = api_key or os.getenv("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Cohere API key required. Set COHERE_API_KEY env var or pass api_key parameter."
            )
        self.api_key: str = resolved_key

        # API endpoint
        self.api_url = "https://api.cohere.ai/v1/embed"

        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Determine dimension based on model
        if "v3.0" in model:
            self._dimension = 1024
        elif "v2" in model:
            self._dimension = 4096
        else:
            # Default fallback
            self._dimension = 1024

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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "texts": texts,
                        "input_type": self.input_type,
                    },
                    headers=self.headers,
                )
                response.raise_for_status()

                data = response.json()

                # Cohere returns embeddings in "embeddings" field
                if "embeddings" not in data:
                    raise RuntimeError(f"Unexpected response format: {data.keys()}")

                # Convert to list of lists
                result: list[list[float]] = []
                for emb in data["embeddings"]:
                    result.append([float(x) for x in emb])

                return result

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Cohere API error: {e.response.status_code} - {e.response.text}"
                )
                raise RuntimeError(
                    f"Cohere embedding API failed: {e.response.status_code}"
                ) from e
            except httpx.RequestError as e:
                logger.error(f"Cohere API request error: {e}")
                raise RuntimeError(f"Cohere embedding API request failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "cohere"
