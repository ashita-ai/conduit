# Changelog

All notable changes to Conduit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- State conversion for low switch_threshold scenarios (issue #182)
- Type checking now enforced in CI (removed `|| true` workaround)

## [1.0.0] - TBD

### Added
- **Core Routing**: ML-powered LLM routing using contextual bandit algorithms
- **12 Routing Algorithms**:
  - Learning: Thompson Sampling (default), LinUCB, Contextual Thompson Sampling, UCB1, Epsilon-Greedy, Dueling
  - Baselines: Random, Always Best, Always Cheapest, Oracle
  - Hybrid: Thompson-to-LinUCB, UCB1-to-LinUCB transitions
- **Hybrid Routing**: Automatic Thompson Sampling to LinUCB transition after convergence
- **Cost Budget Enforcement**: `max_cost_per_query` constraint for cost control
- **Multi-Replica State Management**: PostgreSQL-based distributed state with conflict resolution
- **6 Embedding Providers**: HuggingFace API, OpenAI, Cohere, sentence-transformers, FastEmbed, with auto-detection
- **Runtime Embedding Dimension Detection**: Automatically adapts to provider's embedding size
- **PostgreSQL Persistence**: Durable bandit state storage with Alembic migrations
- **Redis Caching**: Optional embedding cache with graceful degradation
- **REST API**: FastAPI-based with authentication, rate limiting, and size limits
- **Feedback System**: Pluggable adapters for explicit feedback (thumbs, ratings, task success)
- **Implicit Feedback Detection**: Error rates, latency, retries as learning signals
- **LiteLLM Integration**: Support for 100+ LLM providers via unified interface
- **Arbiter LLM-as-Judge**: Quality evaluation with configurable sampling
- **OpenTelemetry Observability**: Tracing and metrics for production monitoring
- **CLI Tool**: Command-line interface for configuration and testing
- **Comprehensive Documentation**: Architecture guides, algorithm explanations, troubleshooting

### Changed
- Pricing source migrated from llm-prices.com to LiteLLM model_cost for better accuracy
- Thompson Sampling uses vectorized Beta sampling for improved performance
- Default algorithm changed to Thompson Sampling (best cold-start behavior)

### Fixed
- Hybrid routing example updated for new HybridRouter API
- Example files fixed for feature_dim and langchain imports
- Dueling bandit database serialization for preference_counts
- Router.update() now uses real features instead of dummy values

## [0.1.0] - 2025-11-01

### Added
- Initial pre-release with core routing functionality
- Basic Thompson Sampling and LinUCB algorithms
- PostgreSQL state persistence
- FastAPI REST endpoints
