# Zero-Config Demo

A self-contained demonstration of Conduit's learning capabilities with zero external dependencies.

## Quick Start

```bash
python examples/00_demo/zero_config_demo.py
```

No API keys, database, or Redis required. Completes in under 5 seconds.

## What This Demo Shows

1. **Exploration Phase**: The bandit explores different models while gathering data
2. **Exploitation Phase**: After learning, the bandit makes smarter routing decisions
3. **Measurable Improvement**: Phase 2 achieves higher rewards than Phase 1

## How It Works

The demo uses Conduit's **Thompson Sampling** bandit algorithm with simulated LLM responses:

- **Simple queries** (e.g., "What is 2+2?"): Cheap models perform as well as expensive ones
- **Complex queries** (e.g., "Analyze economic implications"): Expensive models outperform

The bandit learns this pattern automatically through feedback.

### Simulated Models

| Model | Cost (per 1M tokens) | Simple Query Quality | Complex Query Quality |
|-------|---------------------|---------------------|----------------------|
| gpt-4o-mini | $0.15 | 88% | 72% |
| gpt-4o | $2.50 | 90% | 95% |
| claude-3-5-haiku | $0.80 | 86% | 78% |

### Reward Calculation

Conduit's default reward formula balances three objectives:
- **Quality** (70%): How well the model answered
- **Cost** (20%): How much the response cost
- **Latency** (10%): How fast the response was

## Key Concepts

### Thompson Sampling

Thompson Sampling is a Bayesian bandit algorithm that:
1. Maintains a Beta distribution for each model's expected quality
2. Samples from these distributions to select models
3. Updates beliefs based on observed rewards

This provides optimal exploration-exploitation balance, learning quickly while still discovering better options.

### Why No Embeddings?

Thompson Sampling is non-contextual - it learns which models are best overall, not per-query-type. This makes it perfect for cold-start scenarios:
- No embedding provider required
- Works immediately from query 1
- Transitions to contextual algorithms (LinUCB) after gathering data

In production, Conduit uses Thompson Sampling for the first ~2000 queries, then switches to contextual algorithms that use query embeddings.

## Next Steps

1. **With real LLMs**: See `examples/quickstart.py`
2. **Feedback patterns**: See `examples/feedback_loop.py`
3. **Full documentation**: See `docs/ARCHITECTURE.md`
