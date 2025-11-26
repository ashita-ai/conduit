"""HuggingFace Inference API embedding provider (free default)."""

import logging

import httpx

from conduit.engines.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace Inference API embedding provider.

    Free default option - requires HF_TOKEN for the router API.
    Uses HuggingFace Inference API router endpoint.

    Default model: BAAI/bge-small-en-v1.5 (384 dims)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize HuggingFace embedding provider.

        Args:
            model: HuggingFace model identifier (default: BAAI/bge-small-en-v1.5)
            api_key: API key (HF_TOKEN) - required for router API
            timeout: Request timeout in seconds (default: 30s)

        Example:
            >>> provider = HuggingFaceEmbeddingProvider(api_key="hf_...")
            >>> embedding = await provider.embed("Hello world")
            >>> len(embedding)
            384
        """
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._dimension = 384  # bge-small-en-v1.5 dimension

        # Build API URL (router.huggingface.co/hf-inference/models/{model})
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model}"

        # Build headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats (384 dimensions)

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
                    json={"inputs": texts},
                    headers=self.headers,
                )
                response.raise_for_status()

                # HuggingFace returns list of embeddings (one per input)
                embeddings = response.json()

                # Ensure we have the right shape
                if not isinstance(embeddings, list):
                    raise RuntimeError(
                        f"Unexpected response format: {type(embeddings)}"
                    )

                # Convert to list of lists
                result: list[list[float]] = []
                for emb in embeddings:
                    if isinstance(emb, list):
                        result.append([float(x) for x in emb])
                    else:
                        # Single embedding (shouldn't happen with batch, but handle it)
                        result.append([float(emb)])

                return result

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HuggingFace API error: {e.response.status_code} - {e.response.text}"
                )
                raise RuntimeError(
                    f"HuggingFace embedding API failed: {e.response.status_code}"
                ) from e
            except httpx.RequestError as e:
                logger.error(f"HuggingFace API request error: {e}")
                raise RuntimeError(
                    f"HuggingFace embedding API request failed: {e}"
                ) from e

    @property
    def dimension(self) -> int:
        """Get embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "huggingface"
