# Embedding Providers Guide

Conduit supports multiple embedding providers for query feature extraction. Choose the provider that best fits your needs based on cost, quality, and deployment requirements.

## Quick Comparison

| Provider | Cost | API Key Required | Quality | Best For |
|----------|------|------------------|---------|----------|
| **HuggingFace API** (default) | Free | No | Good | Development, testing, lightweight deployments |
| **OpenAI** | Paid | Yes | Excellent | Production, high-quality embeddings |
| **Cohere** | Paid | Yes | Excellent | Production, semantic search optimization |
| **sentence-transformers** | Free (local) | No | Good | Offline use, air-gapped environments |

## Default Behavior

**Conduit uses HuggingFace Inference API by default** - no API key or additional dependencies required. This provides a lightweight, free option that works out of the box.

```python
from conduit.engines.router import Router

# Uses HuggingFace API automatically (free, no setup needed)
router = Router()
```

## Configuration

### Environment Variables

Set embedding provider via environment variables:

```bash
# .env file
EMBEDDING_PROVIDER=huggingface  # Options: huggingface, openai, cohere, sentence-transformers
EMBEDDING_MODEL=  # Optional, uses provider default if empty

# Provider-specific API keys (if required)
OPENAI_API_KEY=sk-...  # For OpenAI embeddings (reuses LLM API key)
COHERE_API_KEY=...     # For Cohere embeddings (separate key required)
```

### Programmatic Configuration

Configure embedding provider when creating the router:

```python
from conduit.engines.router import Router

# HuggingFace (default, free)
router = Router()

# OpenAI (recommended for production)
router = Router(
    embedding_provider_type="openai",
    embedding_model="text-embedding-3-small",
    embedding_api_key="sk-..."  # Optional, uses OPENAI_API_KEY env var if not provided
)

# Cohere (recommended for production)
router = Router(
    embedding_provider_type="cohere",
    embedding_model="embed-english-v3.0",
    embedding_api_key="..."  # Optional, uses COHERE_API_KEY env var if not provided
)

# Sentence-transformers (offline use)
router = Router(
    embedding_provider_type="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2"
)
```

## Provider Details

### HuggingFace API (Default)

**Free, no API key required** - perfect for development and lightweight deployments.

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Cost**: Free (public models)
- **Setup**: None required - works out of the box
- **Limitations**: Rate limits apply (reasonable for most use cases)

```python
# Default - no configuration needed
router = Router()

# Custom model
router = Router(
    embedding_provider_type="huggingface",
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # 768 dims
)
```

**When to use**: Development, testing, prototypes, lightweight production deployments.

### OpenAI Embeddings

**High-quality embeddings** - recommended for production use.

- **Models**:
  - `text-embedding-3-small` (1536 dims, default)
  - `text-embedding-3-large` (3072 dims)
  - `text-embedding-ada-002` (1536 dims, legacy)
- **Cost**: ~$0.02 per 1M tokens (text-embedding-3-small)
- **Setup**: Requires `OPENAI_API_KEY` (reuses LLM provider key)
- **Features**: Dimension reduction support (text-embedding-3-* models)

```python
# Uses OPENAI_API_KEY from environment
router = Router(
    embedding_provider_type="openai",
    embedding_model="text-embedding-3-small"
)

# With dimension reduction (smaller embeddings)
router = Router(
    embedding_provider_type="openai",
    embedding_model="text-embedding-3-small",
    # Note: dimension reduction configured via provider kwargs
)
```

**When to use**: Production deployments requiring high-quality embeddings, when you already use OpenAI for LLM calls.

### Cohere Embeddings

**Optimized for semantic search** - excellent for production use.

- **Models**:
  - `embed-english-v3.0` (1024 dims, default)
  - `embed-english-light-v3.0` (1024 dims, faster)
  - `embed-multilingual-v3.0` (1024 dims, multilingual)
- **Cost**: Free tier available, then ~$0.10 per 1M tokens
- **Setup**: Requires `COHERE_API_KEY` (separate from LLM provider)
- **Features**: Input type specification (search_query vs search_document)

```python
# Uses COHERE_API_KEY from environment
router = Router(
    embedding_provider_type="cohere",
    embedding_model="embed-english-v3.0"
)

# With custom API key
router = Router(
    embedding_provider_type="cohere",
    embedding_api_key="your-cohere-api-key"
)
```

**When to use**: Production deployments focused on semantic search and retrieval, when you want separate embedding provider from LLM provider.

### Sentence-Transformers (Optional)

**Local embeddings** - for offline use or air-gapped environments.

- **Model**: `all-MiniLM-L6-v2` (384 dims, default)
- **Cost**: Free (runs locally)
- **Setup**: Requires `pip install conduit-router[embeddings]`
- **Limitations**: Larger dependency footprint, slower than API-based options

```python
# Requires: pip install conduit-router[embeddings]
router = Router(
    embedding_provider_type="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2"
)
```

**When to use**: Offline deployments, air-gapped environments, when you want to avoid API calls entirely.

## Embedding Dimensions

Different providers produce different embedding dimensions:

| Provider | Default Model | Dimensions |
|----------|---------------|------------|
| HuggingFace | all-MiniLM-L6-v2 | 384 |
| OpenAI | text-embedding-3-small | 1536 |
| Cohere | embed-english-v3.0 | 1024 |
| sentence-transformers | all-MiniLM-L6-v2 | 384 |

**Note**: Feature dimensions = embedding_dim + 3 metadata features (token_count, complexity_score, domain_confidence).

Conduit automatically handles different embedding dimensions. If using PCA, ensure your PCA model is fitted on embeddings from the same provider.

## Migration Guide

### From sentence-transformers to API-based

**Before** (required sentence-transformers dependency):
```python
router = Router(embedding_model="all-MiniLM-L6-v2")
```

**After** (no dependencies needed):
```python
# HuggingFace API (free, no API key)
router = Router()  # Uses HuggingFace API automatically

# Or explicitly:
router = Router(embedding_provider_type="huggingface")
```

### Switching Providers

To switch providers, update your configuration:

```bash
# .env file
EMBEDDING_PROVIDER=openai  # Change from huggingface to openai
OPENAI_API_KEY=sk-...      # Ensure API key is set
```

Or programmatically:

```python
router = Router(embedding_provider_type="openai")
```

## Security Best Practices

**Never commit API keys to git**. Always use environment variables:

```bash
# ✅ CORRECT: Use .env file (gitignored)
COHERE_API_KEY=your-actual-key-here

# ❌ WRONG: Hardcode in code
router = Router(embedding_api_key="your-actual-key-here")  # Don't do this!
```

If you must pass API keys programmatically (e.g., in tests), use environment variables or secure credential stores:

```python
import os

router = Router(
    embedding_provider_type="cohere",
    embedding_api_key=os.getenv("COHERE_API_KEY")  # ✅ Safe
)
```

## Troubleshooting

### ImportError: sentence-transformers

**Problem**: `ImportError: sentence-transformers package required`

**Solution**: Install optional dependencies:
```bash
pip install conduit-router[embeddings]
```

Or switch to an API-based provider (HuggingFace, OpenAI, Cohere) which don't require sentence-transformers.

### ValueError: API key required

**Problem**: `ValueError: OpenAI/Cohere API key required`

**Solution**: Set the appropriate environment variable:
```bash
export OPENAI_API_KEY=sk-...
# or
export COHERE_API_KEY=...
```

### Dimension Mismatch with PCA

**Problem**: PCA model fitted on different embedding dimensions

**Solution**: Re-fit PCA model on embeddings from the new provider, or disable PCA:
```python
router = Router(use_pca=False)  # Disable PCA
```

## Performance Considerations

- **API-based providers** (HuggingFace, OpenAI, Cohere): ~50-200ms per embedding (network latency)
- **Local providers** (sentence-transformers): ~100-300ms per embedding (CPU-bound)
- **Caching**: All providers benefit from Redis caching (5ms cache hit vs 200ms cache miss)

Enable caching for best performance:
```python
router = Router()  # Caching enabled by default if REDIS_URL is set
```

## Examples

See `examples/` directory for complete usage examples:

- `examples/01_quickstart/hello_world.py` - Basic router usage (uses default HuggingFace)
- `examples/02_routing/basic_routing.py` - Custom embedding provider configuration

## Related Documentation

- **Architecture**: See `docs/ARCHITECTURE.md` for system design
- **PCA Guide**: See `docs/PCA_GUIDE.md` for dimensionality reduction
- **Configuration**: See `conduit/core/config.py` for all configuration options

