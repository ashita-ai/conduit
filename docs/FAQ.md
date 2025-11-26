# Frequently Asked Questions (FAQ)

Common questions about Conduit Router, especially for Hacker News readers.

---

## Why not just use LiteLLM Proxy/Gateway?

**Short answer**: LiteLLM Proxy and Gateway are **static** routing systems. Conduit is **dynamic** and **learns** from your data.

**Detailed comparison**:

| Feature | LiteLLM Proxy/Gateway | Conduit Router |
|---------|------------------------|----------------|
| Routing Strategy | Static (if/else rules) | ML-powered (contextual bandits) |
| Learning | No - manual configuration | Yes - automatic adaptation |
| Cost Optimization | Manual model selection | Automatic cost/quality trade-off |
| Context Awareness | No | Yes - uses query embeddings |
| Cold Start | Immediate (but suboptimal) | Fast warm-up (30% faster convergence) |

**When to use each**:
- **LiteLLM**: Simple load balancing, rate limiting, API key management
- **Conduit**: Cost optimization, quality guarantees, adaptive routing based on query patterns

**Can I use both?** Yes! Use LiteLLM for API management and Conduit for intelligent routing.

---

## How does it handle rate limits?

Conduit includes built-in rate limiting support:

- **File**: `conduit/api/ratelimit.py`
- **Features**:
  - Per-model rate limit tracking
  - Automatic fallback to alternative models
  - Configurable limits and windows
  - Redis-backed distributed rate limiting

**Example**:
```python
from conduit.engines.router import Router

router = Router()
# Rate limits are automatically handled
decision = await router.route(Query(text="Your query"))
```

Rate limits are enforced at the API layer and integrated with the routing decision logic.

---

## Can I use this with LangChain?

**Yes!** Conduit has a LangChain integration.

**Quick start**:
```python
from conduit.engines.router import Router
from examples.integrations.langchain_integration import ConduitLangChainLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

router = Router()
llm = ConduitLangChainLLM(router)

prompt = PromptTemplate.from_template("Explain {topic}")
chain = LLMChain(llm=llm, prompt=prompt)

result = await chain.ainvoke({"topic": "quantum computing"})
```

**Full example**: See `examples/06_integrations/langchain_integration.py`

**Compatibility**:
- Works with LangChain chains (LLMChain, SequentialChain, etc.)
- Supports async operations (`chain.ainvoke()`)
- Drop-in replacement for LangChain LLM classes

---

## Do you have benchmarks showing it actually saves money?

**Yes!** We have comprehensive benchmarks demonstrating cost savings.

**Results** (from `benchmarks/cost_savings_comparison.py` with 1000 queries):
- **vs. Always Best Model**: **50.0% cost savings** ($1.4070 vs $2.8140)
  - Quality: 0.896 vs 0.930 (96% of quality, but 50% cheaper)
  - Conduit maintains high quality while cutting costs in half
- **vs. Always Cheap Model**: **Better quality** (0.896 vs 0.798) with reasonable cost
  - Conduit: $1.4070, Static Cheap: $0.1407
  - Conduit provides 12% better quality (0.896 vs 0.798)
- **Adaptive routing**: Automatically selects optimal model per query based on complexity
  - Uses 5 different models: claude-opus-4.5 (24.4%), gpt-5.1 (24.3%), claude-sonnet-4.5 (24.3%), o4-mini (13.9%), gemini-2.5-pro (13.1%)
  - Query mix: 59.3% simple queries, 40.7% complex queries
  - Learns optimal routing patterns over time

**Visualization**: See `cost_savings_graph.png` for cumulative cost comparison over time.

**Run benchmarks yourself**:
```bash
# Full benchmark (1000 queries)
python benchmarks/cost_savings_comparison.py

# Quick test (20 queries)
python benchmarks/cost_savings_comparison.py 20
```

**Methodology**:
- 1000+ queries with varying complexity (60% simple, 40% complex)
- Compares Conduit vs. static routing strategies
- Tracks cumulative cost and quality scores
- Shows learning curve over time
- Uses OpenAI embeddings for realistic feature extraction

---

## Why not just use gpt-4o-mini for everything?

**Short answer**: You'll overpay for simple queries and underperform on complex ones.

**The problem**:
- **Simple queries** (e.g., "What is 2+2?"): Don't need expensive models
- **Complex queries** (e.g., "Design a distributed system"): Need capable models
- **Static routing**: Can't adapt to query complexity

**Conduit's solution**:
- **Learns** which models work best for which query types
- **Routes** simple queries to cheaper models (saves money)
- **Routes** complex queries to capable models (maintains quality)
- **Adapts** automatically as your query patterns change

**Example savings**:
- Simple query on `gpt-4o-mini`: $0.0001
- Simple query on `gpt-4o`: $0.002 (20x more expensive, no quality gain)
- Complex query on `gpt-4o-mini`: Poor quality (may need retries)
- Complex query on `gpt-4o`: High quality (worth the cost)

Conduit automatically makes these trade-offs based on learned patterns.

---

## How does the learning work?

Conduit uses **contextual bandit algorithms** (specifically LinUCB) to learn optimal routing.

**The process**:
1. **Feature extraction**: Each query is converted to a 387-dim feature vector (embedding + metadata)
2. **Model selection**: Algorithm selects model using Upper Confidence Bound (UCB) policy
3. **Feedback collection**: After execution, collects cost, quality, latency
4. **Learning**: Updates internal model to improve future routing decisions

**Algorithms**:
- **LinUCB**: Contextual bandit with ridge regression (default)
- **Thompson Sampling**: Bayesian approach for exploration/exploitation
- **UCB1**: Non-contextual baseline
- **Epsilon-Greedy**: Simple exploration strategy

**Cold start**: Uses hybrid routing (UCB1 → LinUCB) for 30% faster convergence.

**Documentation**: See `docs/BANDIT_ALGORITHMS.md` for mathematical details.

---

## What's the latency overhead?

**Minimal**. Conduit adds <1ms overhead per query:

- **Feature extraction**: ~0.3ms (embedding + metadata)
- **Routing decision**: ~0.1ms (bandit algorithm)
- **Total overhead**: <0.5ms (negligible compared to LLM API latency)

**Benchmarks**:
- Selection: 3,000+ QPS (0.33ms latency) for standard features
- Selection: 11,500+ QPS (0.087ms latency) for PCA features

**Optimizations**:
- Cached embeddings (Redis)
- Incremental matrix updates (Sherman-Morrison)
- PCA dimensionality reduction (optional)

See `benchmarks/linucb_sherman_morrison_benchmark.py` for detailed performance metrics.

---

## Can I disable hybrid routing?

**Yes!** You can start directly with LinUCB (skipping the UCB1 warm-up phase):

```python
from conduit.engines.router import Router

# Disable hybrid routing (start with LinUCB immediately)
router = Router(use_hybrid_routing=False)
```

**When to disable**:
- You have sufficient historical data
- You want pure contextual routing from the start
- You prefer LinUCB's context-aware decisions

**Default**: Hybrid routing is enabled (recommended for faster cold start).

---

## How do I configure the algorithms?

**Router-level configuration**:
```python
router = Router(
    use_hybrid_routing=True,  # Enable/disable hybrid routing
    # Algorithm selection (when use_hybrid_routing=False)
    # Currently always uses LinUCB in hybrid mode
)
```

**Advanced configuration**: Modify `conduit.yaml` or environment variables:
- `HYBRID_SWITCH_THRESHOLD`: Query count to switch from UCB1 to LinUCB
- `LINUCB_ALPHA`: Exploration parameter (higher = more exploration)
- `REWARD_WEIGHT_QUALITY`: Weight for quality in reward function
- `REWARD_WEIGHT_COST`: Weight for cost in reward function
- `REWARD_WEIGHT_LATENCY`: Weight for latency in reward function

See `docs/HYBRID_ROUTING.md` for detailed configuration options.

---

## What models are supported?

**All models** supported by your LLM providers:

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
- **Anthropic**: claude-3-5-sonnet, claude-3-opus, claude-3-haiku, etc.
- **Google**: gemini-1.5-pro, gemini-1.5-flash, etc.
- **Any provider**: Conduit is provider-agnostic

**Model discovery**: Conduit automatically discovers available models from your configuration.

**Pricing**: Uses `pricing.yaml` for cost calculations. See `scripts/sync_pricing.py` to update.

---

## How do I add my own models?

**Option 1: Configuration file** (`conduit.yaml`):
```yaml
models:
  - my-custom-model
  - another-model
```

**Option 2: Router initialization**:
```python
router = Router(models=["my-custom-model", "another-model"])
```

**Option 3: Dynamic discovery**: Conduit automatically discovers models from your provider APIs.

**Pricing**: Add pricing information to `pricing.yaml` or use default pricing estimates.

---

## Is it production-ready?

**Yes!** Conduit is production-ready with:

- **High performance**: 3,000+ QPS routing decisions
- **Type safety**: Strict mypy checking, Pydantic validation
- **Resilience**: Circuit breakers, retry logic, graceful degradation
- **Observability**: Full tracing, metrics, error tracking
- **Testing**: 88% test coverage, comprehensive test suite

**Production features**:
- Redis caching (optional, graceful degradation)
- PostgreSQL storage (optional, for routing history)
- Rate limiting
- Error handling
- Timeout management

**See**: `docs/ARCHITECTURE.md` for production deployment guide.

---

## How do I contribute?

**Contributions welcome!** See `CONTRIBUTING.md` for guidelines.

**Areas needing help**:
- Additional LLM provider integrations
- More bandit algorithms
- Performance optimizations
- Documentation improvements
- Example use cases

**Development setup**:
```bash
git clone https://github.com/yourusername/conduit
cd conduit
uv sync --all-extras
pytest
```

---

## Where can I learn more?

**Documentation**:
- `README.md`: Quick start and overview
- `docs/ARCHITECTURE.md`: System design
- `docs/BANDIT_ALGORITHMS.md`: Algorithm details
- `docs/HYBRID_ROUTING.md`: Hybrid routing strategy

**Examples**:
- `examples/01_quickstart/`: Basic usage
- `examples/02_routing/`: Advanced routing
- `examples/06_integrations/`: LangChain integration

**Research**:
- LinUCB paper: https://arxiv.org/abs/1003.0146
- Tutorial: https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/

---

**Still have questions?** Open an issue on GitHub or check the documentation.

