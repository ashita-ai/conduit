"""Tests for embedding providers."""

import pytest

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.factory import create_embedding_provider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider


@pytest.mark.asyncio
async def test_huggingface_provider():
    """Test HuggingFace embedding provider (free default)."""
    provider = HuggingFaceEmbeddingProvider()
    assert provider.provider_name == "huggingface"
    assert provider.dimension == 384

    # Test single embedding
    embedding = await provider.embed("Hello world")
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

    # Test batch embedding
    embeddings = await provider.embed_batch(["Hello", "world"])
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_factory_huggingface():
    """Test factory creates HuggingFace provider by default."""
    provider = create_embedding_provider("huggingface")
    assert isinstance(provider, HuggingFaceEmbeddingProvider)
    assert provider.dimension == 384


def test_factory_openai_requires_key():
    """Test OpenAI provider requires API key."""
    with pytest.raises(ValueError, match="API key required"):
        create_embedding_provider("openai")


def test_factory_cohere_requires_key():
    """Test Cohere provider requires API key."""
    with pytest.raises(ValueError, match="API key required"):
        create_embedding_provider("cohere")


def test_factory_invalid_provider():
    """Test factory raises error for invalid provider."""
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider("invalid")


def test_sentence_transformers_optional():
    """Test sentence-transformers provider is optional."""
    try:
        provider = create_embedding_provider("sentence-transformers")
        assert isinstance(provider, EmbeddingProvider)
        assert provider.dimension == 384
    except ImportError:
        # Expected if sentence-transformers not installed
        pytest.skip("sentence-transformers not installed")


@pytest.mark.asyncio
async def test_provider_interface():
    """Test all providers implement EmbeddingProvider interface."""
    providers = [
        HuggingFaceEmbeddingProvider(),
    ]

    for provider in providers:
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")
        assert hasattr(provider, "dimension")
        assert hasattr(provider, "provider_name")

        # Test async methods
        embedding = await provider.embed("test")
        assert isinstance(embedding, list)
        assert len(embedding) == provider.dimension

        embeddings = await provider.embed_batch(["test1", "test2"])
        assert len(embeddings) == 2
        assert all(len(emb) == provider.dimension for emb in embeddings)

