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

### 02_routing/ - Smart Model Selection

**basic_routing.py** - Core routing functionality
- Query analysis and feature extraction
- Model selection with confidence scoring
- Routing metadata and decision reasoning
- Introduction to Router API

**with_constraints.py** - Cost, latency, and quality constraints
- Budget-aware routing (max_cost constraint)
- Quality-focused routing (min_quality constraint)
- Latency-optimized routing (max_latency constraint)
- Provider preferences (preferred_provider)

**hybrid_routing.py** - UCB1→LinUCB warm start (default strategy)
- Phase 1 (0-2,000 queries): UCB1 (fast exploration, no embeddings)
- Phase 2 (2,000+ queries): LinUCB (contextual, smart routing)
- 30% faster convergence vs pure LinUCB
- Smooth transition with knowledge transfer
- **Note**: Router uses hybrid routing by default

**context_specific_priors.py** - Context-aware cold start
- Context detection (code, creative, analysis, simple_qa)
- Context-specific priors from YAML configuration
- Faster convergence (200 queries per context vs 500 general)
- Better first-query quality (0.85-0.92 vs 0.72 baseline)
- Priors sourced from Vellum LLM Leaderboard


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

**See [04_litellm/README.md](04_litellm/README.md) for complete documentation.**

**demo.py** - Comprehensive LiteLLM integration demo
- ML-powered routing for LiteLLM across 100+ providers
- Automatic model selection based on query features
- Cost optimization and quality maximization
- Multi-provider support (OpenAI, Anthropic, Google, Groq)

**custom_config.py** - Customize Conduit behavior in LiteLLM
- Hybrid routing configuration
- Redis caching integration
- Custom embedding models
- Cost tracking

**multi_provider.py** - Multi-provider intelligent routing
- Route across OpenAI + Anthropic + Google + Groq
- Automatic provider selection per query type
- Cost optimization across providers

**arbiter_quality_measurement.py** - Quality evaluation integration
- LLM-as-judge quality assessment
- Automatic feedback for bandit learning
- Configurable sampling and budget control

### 04_production/ - Production Deployment

**failure_scenarios.py** - Production resilience demonstrations
- Redis storage unavailable (graceful degradation)
- All LLM models fail (circuit breaker activation)
- Database connection loss (in-memory operation)
- Combined failures (worst case scenario)
- Shows fail-safe design and automatic recovery

### 05_personalization/ - User Preferences

**user_preferences.py** - Optimization presets for routing
- 4 optimization presets: balanced, quality, cost, speed
- Control how Conduit balances quality/cost/latency
- Different from explicit_feedback.py (this is about preferences, not quality scores)
- Per-query optimization control

## Feature Matrix

| Feature | Example | Redis Required? |
|---------|---------|-----------------|
| Basic routing | hello_world.py, basic_routing.py | No |
| Model discovery | model_discovery.py | No |
| Constraints | with_constraints.py | No |
| Hybrid routing | hybrid_routing.py | No |
| Context priors | context_specific_priors.py | No |
| User preferences | user_preferences.py | No |
| Caching | caching.py | Yes |
| Explicit feedback | explicit_feedback.py | No |
| Implicit feedback | implicit_feedback.py | Yes (retry detection) |
| Combined feedback | combined_feedback.py | Yes (retry detection) |
| Failure scenarios | failure_scenarios.py | No |
| LiteLLM integration | demo.py (see 04_litellm/) | Optional |

## Running Examples

```bash
# Quickstart
python examples/01_quickstart/hello_world.py
python examples/01_quickstart/model_discovery.py

# Routing
python examples/02_routing/basic_routing.py
python examples/02_routing/with_constraints.py
python examples/02_routing/hybrid_routing.py
python examples/02_routing/context_specific_priors.py

# Optimization (Redis optional but recommended)
redis-server  # Start Redis in another terminal
python examples/03_optimization/caching.py
python examples/03_optimization/explicit_feedback.py
python examples/03_optimization/implicit_feedback.py
python examples/03_optimization/combined_feedback.py

# Production resilience
python examples/04_production/failure_scenarios.py

# LiteLLM integration (see 04_litellm/README.md)
python examples/04_litellm/demo.py
python examples/04_litellm/custom_config.py
python examples/04_litellm/multi_provider.py

# Personalization
python examples/05_personalization/user_preferences.py
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

### Production Resilience
Conduit handles common production failures gracefully:
- **Redis unavailable**: Cache disabled, routing continues (slower)
- **Model failures**: Circuit breaker activation, automatic retry, fallback to default model
- **Database unavailable**: In-memory operation, no persistence (ephemeral state)
- **Combined failures**: Degraded mode, core functionality preserved

See `examples/04_production/failure_scenarios.py` for detailed demonstrations.

## Learning Path

**Recommended order for new users:**

1. **Start**: hello_world.py - Verify installation (5 lines)
2. **Discover**: model_discovery.py - See what models you can use
3. **Routing**: basic_routing.py - Understand core concepts
4. **Constraints**: with_constraints.py - Cost/quality control
5. **Hybrid Routing**: hybrid_routing.py - Understand default strategy
6. **Context Priors**: context_specific_priors.py - Faster cold start
7. **Caching**: caching.py - Performance optimization (requires Redis)
8. **Explicit Feedback**: explicit_feedback.py - User ratings
9. **Implicit Feedback**: implicit_feedback.py - Behavioral signals
10. **Combined**: combined_feedback.py - Full feedback system
11. **Production**: failure_scenarios.py - Resilience patterns
12. **LiteLLM**: demo.py - Integration with LiteLLM (see 04_litellm/README.md)

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
