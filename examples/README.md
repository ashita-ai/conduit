# Conduit Examples

Demonstrations of Conduit's intelligent LLM routing capabilities.

## Getting Started

```bash
# Install Conduit
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."

# Optional: Install Redis for caching and implicit feedback
brew install redis && redis-server
```

## Example Structure

### 01_quickstart/ - Get Started in 5 Minutes

**hello_world.py** - Absolute minimum example (5 lines)
- Shows basic routing in simplest form
- Perfect first example to verify installation
- No configuration or setup required

**model_discovery.py** - See what models you can use
- Lists all 71+ supported models across 5 providers
- Shows which models YOU can use (based on .env API keys)
- Filter by quality, cost, or provider
- Dynamic detection of configured providers
- Auto-fetches pricing from llm-prices.com (24h cache)

**simple_router.py** - Basic routing with custom models
- Define available models and route queries
- See routing decisions and confidence scores
- Introduction to Router API

### 02_routing/ - Smart Model Selection

**basic_routing.py** - Core routing functionality
- Query analysis and feature extraction
- Model selection with confidence scoring
- Routing metadata and decision reasoning

**with_constraints.py** - Cost, latency, and quality constraints
- Budget-aware routing (max_cost constraint)
- Quality-focused routing (min_quality constraint)
- Latency-optimized routing (max_latency constraint)
- Provider preferences (preferred_provider)

**contextual_thompson.py** - Bayesian contextual bandit routing
- Contextual Thompson Sampling with Bayesian linear regression
- Posterior distribution tracking (μ, Σ)
- Natural exploration via posterior sampling
- Sliding window for non-stationarity adaptation
- Uncertainty quantification and confidence metrics

### 03_optimization/ - Performance & Learning

**caching.py** - 10-40x speedup with Redis caching
- Query feature caching for repeated queries
- Cache hit/miss statistics and performance metrics
- Graceful fallback when Redis unavailable
- Circuit breaker pattern for reliability

**explicit_feedback.py** - User ratings and quality scores
- Submit quality_score (0-1), user_rating (1-5), met_expectations (bool)
- Updates Thompson Sampling for better routing
- Weighted 70% (explicit) vs 30% (implicit)

**implicit_feedback.py** - "Observability Trinity" (Errors + Latency + Retries)
- Error detection (model failures, empty responses, error patterns)
- Latency analysis (user patience tolerance categorization)
- Retry detection (semantic similarity-based query matching)
- Learn without explicit ratings from behavioral signals

**combined_feedback.py** - Explicit + implicit feedback together
- Demonstrates weighted feedback system (70% explicit, 30% implicit)
- Shows how mixed signals are resolved
- Real-world learning scenarios

### 04_litellm/ - LiteLLM Integration

**basic_usage.py** - Simple LiteLLM + Conduit routing
- Zero-config ML-powered model selection
- Automatic feedback learning from every request
- Works with 100+ LLM providers via LiteLLM
- Demonstrates ConduitRoutingStrategy setup

**custom_config.py** - Advanced configuration options
- Hybrid routing (UCB1→LinUCB warm start)
- Redis caching (10-40x performance improvement)
- Custom bandit parameters and embedding models
- Performance metrics and benchmarking

**multi_provider.py** - Cross-provider intelligent routing
- 5+ provider support (OpenAI, Anthropic, Google, etc.)
- Automatic failover and provider redundancy
- Cost optimization across providers
- Provider-specific strengths learning

**feedback_loop.py** - Automatic learning demonstration
- Deep dive into ConduitFeedbackLogger
- Cost, latency, quality capture from responses
- Composite reward calculation explanation
- Zero manual feedback required

**performance_comparison.py** - ML vs rule-based routing
- Benchmarks against round-robin strategy
- Cost and latency analysis
- Model distribution comparison
- Demonstrates 20-40% cost savings with ML

**README.md** - Comprehensive guide
- Setup instructions and prerequisites
- Example walkthroughs with expected outputs
- Configuration reference
- Troubleshooting and best practices

### 04_production/ - Production Deployment

*Coming soon:*
- fastapi_endpoint.py - REST API integration
- batch_processing.py - High-throughput batch routing
- monitoring.py - Observability and metrics

## Feature Matrix

| Feature | Example | Redis Required? |
|---------|---------|-----------------|
| Basic routing | hello_world.py | No |
| Custom models | simple_router.py | No |
| Constraints | with_constraints.py | No |
| Caching | caching.py | Yes |
| Explicit feedback | explicit_feedback.py | No |
| Implicit feedback | implicit_feedback.py | Yes (retry detection) |
| Combined feedback | combined_feedback.py | Yes (retry detection) |
| LiteLLM basic | basic_usage.py | No |
| LiteLLM caching | custom_config.py | Yes (optional) |
| Multi-provider | multi_provider.py | No |
| ML comparison | performance_comparison.py | No |

## Running Examples

```bash
# Quickstart
python examples/01_quickstart/hello_world.py
python examples/01_quickstart/simple_router.py

# Routing
python examples/02_routing/basic_routing.py
python examples/02_routing/with_constraints.py

# Optimization (Redis optional but recommended)
redis-server  # Start Redis in another terminal
python examples/03_optimization/caching.py
python examples/03_optimization/explicit_feedback.py
python examples/03_optimization/implicit_feedback.py
python examples/03_optimization/combined_feedback.py

# LiteLLM Integration (requires OPENAI_API_KEY or other provider keys)
export OPENAI_API_KEY="sk-..."  # Set API key
python examples/04_litellm/basic_usage.py
python examples/04_litellm/custom_config.py
python examples/04_litellm/multi_provider.py
python examples/04_litellm/feedback_loop.py
python examples/04_litellm/performance_comparison.py
```

## Key Concepts

### Thompson Sampling
Bayesian bandit algorithm that balances exploration (trying models) with exploitation (using best performers). Updates model success rates based on feedback.

### Query Features
Semantic embeddings and extracted features (complexity, domain, sentiment) used for contextual routing decisions.

### Feedback Loop
- **Explicit**: User ratings (quality_score, user_rating, met_expectations)
- **Implicit**: System signals (errors, latency, retries)
- **Weighted**: 70% explicit, 30% implicit by default

### Caching
Redis-based caching of query features to dramatically improve routing performance on repeated or similar queries.

### Graceful Degradation
All features work without Redis - caching and retry detection are simply disabled if Redis is unavailable.

## Learning Path

**Recommended order for new users:**

1. **Start**: hello_world.py - Verify installation (5 lines)
2. **Discover**: model_discovery.py - See what models you can use
3. **Routing**: basic_routing.py - Understand core concepts
4. **Constraints**: with_constraints.py - Cost/quality control
5. **Caching**: caching.py - Performance optimization
6. **Explicit Feedback**: explicit_feedback.py - User ratings
7. **Implicit Feedback**: implicit_feedback.py - Behavioral signals
8. **Combined**: combined_feedback.py - Full feedback system
9. **LiteLLM**: basic_usage.py - LiteLLM plugin integration
10. **Multi-Provider**: multi_provider.py - Cross-provider routing

## Redis Setup

### macOS
```bash
brew install redis
brew services start redis
```

### Linux
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

### Docker
```bash
docker run -d -p 6379:6379 redis:alpine
```

## Troubleshooting

### Redis Connection Errors
Examples gracefully degrade without Redis:
- **Caching**: Falls back to no caching (slower but functional)
- **Feedback**: Retry detection disabled, error/latency still work

### Model API Errors
Set API keys in environment:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Tips

- **Start simple**: Begin with hello_world.py before advanced features
- **Redis optional**: Core routing works without Redis
- **Read output**: Examples show decision-making process
- **Experiment**: Modify queries and constraints to see behavior changes

## Additional Resources

- **[Main README](../README.md)** - Project overview and installation
- **[CLAUDE.md](../CLAUDE.md)** - Development guidelines and architecture
- **[docs/](../docs/)** - Design documentation and decisions

## CLI Alternative

Prefer CLI? Try the built-in commands:

```bash
# Run a single query
conduit run --query "What is 2+2?"

# Run with constraints
conduit run --query "Write a poem" --max-cost 0.001

# Run a demo (10 queries)
conduit demo

# JSON output for scripting
conduit run --query "Hello" --json
```
