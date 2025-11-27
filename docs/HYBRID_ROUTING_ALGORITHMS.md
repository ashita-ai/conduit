# Hybrid Routing Algorithms

**Version**: 0.1.0
**Last Updated**: 2025-11-27

---

## Overview

Conduit's hybrid routing system supports **4 configurable algorithm combinations** for optimizing the cold-start vs warm-routing trade-off. This flexibility allows you to choose the best strategy for your workload based on quality requirements, cost constraints, and convergence speed needs.

## The 4 Algorithm Combinations

### 1. UCB1 → LinUCB (Fast Cold Start)

**Phase 1**: UCB1 (non-contextual)
**Phase 2**: LinUCB (contextual linear UCB)

**Best for**: Applications prioritizing fast convergence and low cost during cold start.

**Characteristics**:
- **Cold Start (0-2,000 queries)**: UCB1 explores quickly without expensive embeddings
- **Warm Routing (2,000+ queries)**: LinUCB uses query features for smart routing
- **Cost**: Lowest (no embeddings until query 2,000)
- **Quality**: Moderate during cold start, improves significantly after transition

```python
from conduit.engines.hybrid_router import HybridRouter

router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
    phase1_algorithm="ucb1",  # Faster convergence
    phase2_algorithm="linucb",
    switch_threshold=2000
)
```

---

### 2. Thompson Sampling → LinUCB (Higher Quality Cold Start, Default)

**Phase 1**: Thompson Sampling (Bayesian non-contextual)
**Phase 2**: LinUCB (contextual linear UCB)

**Best for**: Applications where quality matters more than cost, even during cold start.

**Characteristics**:
- **Cold Start (0-2,000 queries)**: Thompson Sampling achieves higher quality through Bayesian exploration
- **Warm Routing (2,000+ queries)**: LinUCB's proven contextual routing
- **Cost**: Slightly higher (Thompson Sampling explores more expensive models)
- **Quality**: Superior performance on complex reasoning tasks vs UCB1

```python
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o", "claude-opus-4-5"],
    phase1_algorithm="thompson_sampling",  # Default: quality-first cold start
    phase2_algorithm="linucb",
    switch_threshold=2000
)
```

---

### 3. UCB1 → Contextual Thompson Sampling

**Phase 1**: UCB1 (non-contextual)
**Phase 2**: Contextual Thompson Sampling (Bayesian linear regression)

**Best for**: Applications needing Bayesian uncertainty quantification in warm routing.

**Characteristics**:
- **Cold Start**: Fast UCB1 exploration
- **Warm Routing**: Bayesian linear regression with natural exploration-exploitation balance
- **Use Case**: When you need posterior uncertainty estimates for decision-making
- **Quality**: Similar to LinUCB but with probabilistic uncertainty

```python
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o", "gemini-2-5-pro"],
    phase1_algorithm="ucb1",
    phase2_algorithm="contextual_thompson_sampling",  # Bayesian warm routing
    thompson_lambda=1.0,  # Regularization parameter
    switch_threshold=2000
)
```

---

### 4. Thompson Sampling → Contextual Thompson Sampling (Full Bayesian)

**Phase 1**: Thompson Sampling (Bayesian non-contextual)
**Phase 2**: Contextual Thompson Sampling (Bayesian contextual)

**Best for**: Research applications, maximum quality requirements, Bayesian decision-making.

**Characteristics**:
- **Cold Start**: Optimal Bayesian exploration
- **Warm Routing**: Full Bayesian posterior with contextual features
- **Quality**: Highest quality across both phases
- **Cost**: Highest (explores expensive models more frequently)
- **Use Case**: When quality is paramount and cost is secondary

```python
router = HybridRouter(
    models=["gpt-4o", "claude-opus-4-5", "gemini-3-0-pro"],
    phase1_algorithm="thompson_sampling",  # Full Bayesian
    phase2_algorithm="contextual_thompson_sampling",
    thompson_lambda=1.0,
    switch_threshold=2000
)
```

---

## Algorithm Comparison Table

| Combination | Cold Start Quality | Warm Routing | Cold Start Cost | Use Case |
|-------------|-------------------|--------------|-----------------|----------|
| **Thompson → LinUCB** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰 | **Default - Quality-first** |
| **UCB1 → LinUCB** | ⭐⭐ | ⭐⭐⭐⭐ | 💰 | Fast convergence |
| **UCB1 → C.Thompson** | ⭐⭐ | ⭐⭐⭐⭐ | 💰 | Bayesian uncertainty |
| **Thompson → C.Thompson** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Research/Max quality |

---

## State Conversion & Migration

### Optimistic State Conversion

When you change algorithms, Conduit **automatically converts saved state** to preserve learned knowledge. This allows you to experiment with different algorithms without losing progress.

**Example**: Switching from UCB1→LinUCB to Thompson→LinUCB

```python
# Original router (saved state with UCB1 → LinUCB)
old_router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="ucb1",
    phase2_algorithm="linucb"
)
await old_router.route(query)  # 1,500 queries processed
await old_router.save_state(store, "production-router")

# New router with different algorithms
new_router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="thompson_sampling",  # Changed!
    phase2_algorithm="linucb"
)

# Load state with automatic conversion
await new_router.load_state(store, "production-router", allow_conversion=True)
# ✅ UCB1 state converted to Thompson Sampling state
# Learned quality estimates preserved!
```

### How Conversion Works

Conversion is **mathematically lossless** for compatible algorithm pairs:

**UCB1 ↔ Thompson Sampling**:
```python
# UCB1 → Thompson
alpha = 1 + mean_reward * arm_pulls
beta = 1 + (1 - mean_reward) * arm_pulls

# Thompson → UCB1
mean_reward = alpha / (alpha + beta)
arm_pulls = (alpha + beta) - 2
```

**LinUCB ↔ Contextual Thompson**:
```python
# LinUCB → Contextual Thompson
mu = A^-1 @ b      # Ridge regression → Posterior mean
Sigma = A^-1        # Uncertainty → Posterior covariance

# Contextual Thompson → LinUCB
A = Sigma^-1        # Posterior covariance → Design matrix
b = A @ mu          # Since theta = A^-1 @ b
```

**Cross-category conversions** (non-contextual ↔ contextual) use warm-start initialization:
- Extract mean quality from non-contextual algorithm
- Initialize contextual algorithm's first dimension with this prior
- Remaining dimensions start neutral (identity covariance)

### Strict Mode (No Conversion)

For A/B testing or benchmarking, disable conversion:

```python
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="thompson_sampling"
)

# Load with strict mode (error if algorithms don't match)
try:
    await router.load_state(store, "production-router", allow_conversion=False)
except ValueError as e:
    print(f"Algorithm mismatch: {e}")
    # Start fresh for fair benchmark
```

---

## Configuration Parameters

### Router Initialization

```python
HybridRouter(
    models: list[str],                    # Models to route between
    switch_threshold: int = 2000,         # Query count for phase transition
    phase1_algorithm: str = "ucb1",       # Cold start algorithm
    phase2_algorithm: str = "linucb",     # Warm routing algorithm
    feature_dim: int = 387,               # Feature dimensions (387 or 67 with PCA)
    ucb1_c: float = 1.5,                  # UCB1 exploration (if using UCB1)
    linucb_alpha: float = 1.0,            # LinUCB exploration (if using LinUCB)
    thompson_lambda: float = 1.0,         # Thompson regularization
    reward_weights: dict | None = None,   # Multi-objective weights
    window_size: int = 0,                 # Sliding window for non-stationarity
    analyzer: QueryAnalyzer | None = None # Feature extractor
)
```

### Valid Algorithm Values

**Phase 1 (Non-contextual)**:
- `"ucb1"` (default)
- `"thompson_sampling"`

**Phase 2 (Contextual)**:
- `"linucb"` (default)
- `"contextual_thompson_sampling"`

---

## When to Use Each Configuration

### Production Default

```python
# Default: Thompson Sampling → LinUCB (quality-first)
router = HybridRouter(models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"])
# Uses thompson_sampling by default for superior cold-start quality
```

**Why**: Better model selection during cold start through Bayesian exploration, proven LinUCB performance in warm routing.

### Fast Convergence (Lower Cost)

```python
# When speed > quality: UCB1 → LinUCB
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
    phase1_algorithm="ucb1",  # Faster but lower quality
    phase2_algorithm="linucb"
)
```

**Why**: Fastest convergence, lowest cost during cold start.
**Use for**: High-volume applications where cost optimization is critical.

### Research & Experimentation

```python
# Maximum exploration: Thompson → Contextual Thompson
router = HybridRouter(
    models=["gpt-4o", "claude-opus-4-5"],
    phase1_algorithm="thompson_sampling",
    phase2_algorithm="contextual_thompson_sampling",
    thompson_lambda=1.0
)
```

**Why**: Full Bayesian posterior, optimal exploration-exploitation, uncertainty quantification.
**Use for**: Algorithm research, A/B testing, uncertainty-aware decision-making.

---

## Migration Guide

### Existing Deployments

**New Default (Thompson Sampling → LinUCB)**:

```python
# New default (version 0.2.0+)
router = HybridRouter(models=["gpt-4o-mini", "gpt-4o"])
# Uses thompson_sampling by default (quality-first cold start)
```

**Upgrading from UCB1 → LinUCB**:

If you have existing routers with UCB1 state, they'll automatically convert:

```python
# Old router had UCB1 state saved
router = HybridRouter(models=["gpt-4o-mini", "gpt-4o"])
# New default is thompson_sampling

await router.load_state(store, "router-1", allow_conversion=True)
# ✅ Automatically converts UCB1 → Thompson Sampling state
```

**Staying with UCB1** (opt-out):

```python
# Explicitly use UCB1 if you prefer fast convergence over quality
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="ucb1",
    phase2_algorithm="linucb"
)
```

### Fresh Start for Benchmarking

```python
# Disable conversion for fair A/B test
router_thompson = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="thompson_sampling"
)
# Don't load old state - start fresh for benchmark
```

---

## Advanced Topics

### Custom Switch Threshold

```python
# Longer cold start for more exploration
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="thompson_sampling",
    switch_threshold=5000  # Explore for 5K queries before LinUCB
)
```

### Multi-Objective Optimization

```python
# Prioritize quality over cost
router = HybridRouter(
    models=["gpt-4o", "claude-opus-4-5"],
    phase1_algorithm="thompson_sampling",
    reward_weights={
        "quality": 0.80,  # 80% quality (default: 70%)
        "cost": 0.10,     # 10% cost (default: 20%)
        "latency": 0.10   # 10% latency (default: 10%)
    }
)
```

### Non-Stationarity Handling

```python
# Adapt to changing model quality over time
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="thompson_sampling",
    window_size=1000  # Keep only last 1K observations
)
```

---

## Troubleshooting

### Algorithm Mismatch Errors

**Error**: `ValueError: Algorithm mismatch (allow_conversion=False)`

**Solution**: Enable conversion or use matching algorithms:
```python
# Option 1: Enable conversion
await router.load_state(store, "router-1", allow_conversion=True)

# Option 2: Match algorithms
router = HybridRouter(
    models=["gpt-4o-mini", "gpt-4o"],
    phase1_algorithm="ucb1",  # Match saved state
    phase2_algorithm="linucb"
)
```

### State Loading Failures

**Error**: `StateStore returned None`

**Solution**: No saved state exists, router will start fresh:
```python
loaded = await router.load_state(store, "router-1")
if not loaded:
    print("Starting fresh (no saved state)")
```

---

## Performance Considerations

### Memory Usage

| Algorithm | Phase1 Memory | Phase2 Memory |
|-----------|---------------|---------------|
| UCB1 | ~1KB per arm | - |
| Thompson | ~2KB per arm | - |
| LinUCB | - | ~150KB per arm (387 dims) |
| C. Thompson | - | ~150KB per arm (387 dims) |

**PCA Reduction**: Use `feature_dim=67` (64 PCA + 3 metadata) to reduce phase2 memory by ~40x.

### Computational Cost

| Algorithm | Selection Time | Update Time |
|-----------|---------------|-------------|
| UCB1 | O(k) | O(1) |
| Thompson | O(k) | O(1) |
| LinUCB | O(k × d²) | O(d³) |
| C. Thompson | O(k × d²) | O(d³) |

Where: k = number of arms, d = feature dimensions

**Recommendation**: Use LinUCB for d ≤ 100, consider PCA for higher dimensions.

---

## References

- **UCB1**: Auer et al. (2002) - "Finite-time Analysis of the Multiarmed Bandit Problem"
- **Thompson Sampling**: Agrawal & Goyal (2012) - "Analysis of Thompson Sampling"
- **LinUCB**: Li et al. (2010) - "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- **Contextual Thompson**: Agrawal & Goyal (2013) - "Thompson Sampling for Contextual Bandits with Linear Payoffs"

---

**Last Updated**: 2025-11-27
**Version**: 0.1.0 (Initial release with 4 algorithm combinations)
