"""Tests for embedding providers."""

import os

import pytest

from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.cohere import CohereEmbeddingProvider
from conduit.engines.embeddings.factory import create_embedding_provider
from conduit.engines.embeddings.huggingface import HuggingFaceEmbeddingProvider
from conduit.engines.embeddings.openai import OpenAIEmbeddingProvider


@pytest.mark.asyncio
@pytest.mark.downloads_models
@pytest.mark.slow
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


def test_factory_openai_requires_key(monkeypatch):
    """Test OpenAI provider requires API key."""
    # Ensure no API key in environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key required"):
        create_embedding_provider("openai")


def test_factory_cohere_requires_key(monkeypatch):
    """Test Cohere provider requires API key."""
    # Ensure no API key in environment
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Cohere API key required"):
        create_embedding_provider("cohere")


def test_factory_invalid_provider():
    """Test factory raises error for invalid provider."""
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider("invalid")


@pytest.mark.downloads_models
@pytest.mark.slow
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
@pytest.mark.downloads_models
@pytest.mark.slow
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


# FastEmbed Tests
def test_fastembed_optional():
    """Test FastEmbed provider is optional."""
    try:
        provider = create_embedding_provider("fastembed")
        assert isinstance(provider, EmbeddingProvider)
        assert provider.dimension == 384
        assert provider.provider_name == "fastembed"
    except ImportError:
        # Expected if fastembed not installed
        pytest.skip("fastembed not installed")


@pytest.mark.asyncio
@pytest.mark.downloads_models
@pytest.mark.slow
async def test_fastembed_provider():
    """Test FastEmbed embedding provider."""
    try:
        from conduit.engines.embeddings.fastembed_provider import FastEmbedProvider
    except ImportError:
        pytest.skip("fastembed not installed")

    provider = FastEmbedProvider()
    assert provider.provider_name == "fastembed"
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
@pytest.mark.downloads_models
@pytest.mark.slow
async def test_fastembed_empty_batch():
    """Test FastEmbed handles empty batch."""
    try:
        from conduit.engines.embeddings.fastembed_provider import FastEmbedProvider
    except ImportError:
        pytest.skip("fastembed not installed")

    provider = FastEmbedProvider()
    embeddings = await provider.embed_batch([])
    assert embeddings == []


@pytest.mark.downloads_models
@pytest.mark.slow
def test_fastembed_model_dimensions():
    """Test FastEmbed correctly identifies model dimensions."""
    try:
        from conduit.engines.embeddings.fastembed_provider import FastEmbedProvider
    except ImportError:
        pytest.skip("fastembed not installed")

    # Test default model
    provider_small = FastEmbedProvider("BAAI/bge-small-en-v1.5")
    assert provider_small.dimension == 384

    # Test base model
    provider_base = FastEmbedProvider("BAAI/bge-base-en-v1.5")
    assert provider_base.dimension == 768


# Auto Mode Tests
def test_factory_auto_mode_no_providers(monkeypatch):
    """Test auto mode raises error when no providers available."""
    # Clear all API keys
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    # Mock the helper functions to return False (imports fail)
    from conduit.engines.embeddings import factory

    monkeypatch.setattr(factory, "_try_import_fastembed", lambda: False)
    monkeypatch.setattr(factory, "_try_import_sentence_transformers", lambda: False)

    with pytest.raises(RuntimeError, match="No embedding providers available"):
        create_embedding_provider("auto")


def test_factory_auto_mode_openai_priority(monkeypatch):
    """Test auto mode prefers OpenAI when API key exists."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = create_embedding_provider("auto")
    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_factory_auto_mode_cohere_fallback(monkeypatch):
    """Test auto mode falls back to Cohere when OpenAI unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    provider = create_embedding_provider("auto")
    assert isinstance(provider, CohereEmbeddingProvider)


def test_factory_auto_mode_fastembed_fallback(monkeypatch):
    """Test auto mode falls back to FastEmbed when API providers unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    try:
        provider = create_embedding_provider("auto")
        # Should be FastEmbed if installed, otherwise sentence-transformers
        assert isinstance(provider, EmbeddingProvider)
    except RuntimeError:
        # Expected if no local providers installed
        pytest.skip("No local embedding providers installed")

