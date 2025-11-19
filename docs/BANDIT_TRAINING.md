# Bandit Training Strategy

**Key Insight**: Thompson Sampling (contextual bandits) is NOT like fine-tuning LLMs - it's fundamentally different.

## Thompson Sampling ≠ LLM Fine-Tuning

### Critical Differences

| Aspect | LLM Fine-Tuning | Thompson Sampling (Bandits) |
|--------|-----------------|----------------------------|
| **Training Type** | Offline, pre-deployment | Online, during usage |
| **Data Required** | Labeled training corpus | Real-time feedback |
| **Computation** | GPU hours, expensive | Simple arithmetic (alpha++, beta++) |
| **Parameters** | Neural network weights (millions) | Beta distribution (2 params per model) |
| **Pre-Training** | Required before deployment | NOT needed - zero-shot deployment |
| **Learning Speed** | Hours/days | Converges in 200-1000 queries |
| **Cost** | $100s-$1000s in compute | Negligible |

### Why This Matters

**LLM fine-tuning**:
```python
# Traditional approach - offline training
model = GPT4()
model.fine_tune(
    training_data=labeled_corpus,  # Need 1000s of examples
    epochs=3,
    gpu_hours=10,                  # Expensive!
    cost="$500"
)
# Deploy only after training complete
```

**Bandit learning**:
```python
# Our approach - online learning
bandit = ContextualBandit(models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"])

# Deploy immediately with zero training!
# Learns from every single query:
for query in production_traffic:
    model = bandit.select(features)      # Pick model (exploration/exploitation)
    result = execute_llm(model, query)   # Execute
    feedback = collect_feedback(result)  # Errors, latency, ratings
    bandit.update(model, reward)         # Learn! (just arithmetic)
```

**Zero-shot deployment** - this is the magic! No training data, no GPU hours, no pre-deployment phase.

## How Thompson Sampling Learns

### Initialization: Uniform Priors

```python
# All models start equal - we know nothing
models = {
    "gpt-4o-mini": Beta(alpha=1, beta=1),  # Uniform distribution
    "gpt-4o": Beta(alpha=1, beta=1),
    "claude-sonnet-4": Beta(alpha=1, beta=1)
}

# This means: "Each model has 50% expected success rate, but high uncertainty"
```

### Learning Phase: Feedback Accumulation

Every query provides feedback that updates the distributions:

```python
# Query 1: Route to gpt-4o-mini
result = execute("gpt-4o-mini", "What is Python?")
feedback = detect_signals(result)

if feedback.success:
    models["gpt-4o-mini"].alpha += 1  # Success count
else:
    models["gpt-4o-mini"].beta += 1   # Failure count

# After 100 queries:
models = {
    "gpt-4o-mini": Beta(alpha=45, beta=15),  # 75% success rate, moderate confidence
    "gpt-4o": Beta(alpha=38, beta=8),        # 82% success rate, moderate confidence
    "claude-sonnet-4": Beta(alpha=40, beta=12) # 77% success rate, moderate confidence
}
```

### Convergence: Stable Routing

After sufficient feedback, distributions stabilize:

```python
# After 1000 queries:
models = {
    "gpt-4o-mini": Beta(alpha=450, beta=150),  # 75% success, high confidence
    "gpt-4o": Beta(alpha=380, beta=80),        # 82% success, high confidence
    "claude-sonnet-4": Beta(alpha=400, beta=120) # 77% success, high confidence
}

# Routing becomes consistent (but still explores occasionally)
# The more data, the more confident and consistent the routing
```

## The Learning Curve

### Phase 1: Cold Start (Queries 1-100)

**Characteristics**:
- High exploration (tries everything randomly)
- Inconsistent routing decisions
- Learning basic model capabilities
- High variance in performance

**User Experience**: Some suboptimal routing, but gathering crucial data

**Example**:
```
Query 1:  "Debug this code" → gpt-4o-mini (random) → Success ✓
Query 2:  "What is 2+2?" → gpt-4o (random, expensive!) → Success ✓
Query 3:  "Write a poem" → claude-sonnet-4 (random) → Success ✓
Query 50: Pattern emerging, but still exploring heavily
```

### Phase 2: Learning (Queries 100-500)

**Characteristics**:
- Moderate exploration (balances known-good with trying new)
- Patterns emerge (complex → premium, simple → budget)
- Routing becomes smarter
- Reduced variance

**User Experience**: Mostly good routing, occasional exploration

**Example**:
```
Query 150: "Debug this code" → gpt-4o (learned: code is hard) ✓
Query 200: "What is 2+2?" → gpt-4o-mini (learned: simple is cheap) ✓
Query 300: Still exploring edge cases, refining boundaries
```

### Phase 3: Converged (Queries 500-1000+)

**Characteristics**:
- Low exploration (mostly exploit learned patterns)
- Consistent routing decisions
- High confidence in model capabilities
- Minimal variance

**User Experience**: Optimal routing, rare exploration

**Example**:
```
Query 600: "Debug this code" → gpt-4o (high confidence) ✓
Query 800: "What is 2+2?" → gpt-4o-mini (high confidence) ✓
Query 1000: Routing is now tuned to YOUR specific workload
```

## Training Data = Feedback Signals

### Explicit Feedback (User Ratings)

When users provide ratings:

```python
feedback = Feedback(
    quality_score=0.95,    # 0-1 scale (0.95 = 95% quality)
    user_rating=5,         # 1-5 stars
    met_expectations=True  # Boolean
)

# Convert to reward (0-1 scale)
reward = (
    quality_score * 0.6 +           # 60% weight on quality
    (user_rating / 5) * 0.4         # 40% weight on rating
) * 0.7  # 70% weight for explicit feedback

# Example: (0.95 * 0.6 + 1.0 * 0.4) * 0.7 = 0.665
```

### Implicit Feedback (Behavioral Signals)

Automatic signals from system behavior:

```python
feedback = ImplicitFeedback(
    error_occurred=False,        # No errors
    latency_seconds=0.8,         # Fast response (< 10s)
    latency_tolerance="high",    # User happy with speed
    retry_detected=False         # Didn't retry query
)

# Convert to reward with priority rules:
if error_occurred:
    reward = 0.0  # Hard failure
elif retry_detected:
    reward = 0.3  # Strong negative signal
else:
    # Latency-based reward
    reward = {
        "high": 0.9,    # < 10s
        "medium": 0.7,  # 10-30s
        "low": 0.5      # > 30s
    }[latency_tolerance]

reward *= 0.3  # 30% weight for implicit feedback

# Example: 0.9 * 0.3 = 0.27
```

### Combined Weighted Feedback

```python
# Total reward combines both signals
total_reward = explicit_reward + implicit_reward

# Example:
# Explicit: 0.665 (from rating)
# Implicit: 0.27 (from fast latency)
# Total: 0.935 → Strong success signal

# Update bandit
if total_reward >= 0.7:  # Success threshold
    model.alpha += 1  # Increment success count
else:
    model.beta += 1   # Increment failure count
```

## No Training Corpus Needed

**Key Advantage**: The bandit learns from YOUR specific workload, not generic benchmarks.

**Why this matters**:
- Different users have different query distributions
- Customer support queries ≠ code generation queries ≠ creative writing
- Generic training wouldn't work - your patterns are unique

**Example Workload Differences**:

```python
# User A: Customer Support
query_distribution = {
    "simple_faq": 60%,      # "Where is my order?"
    "moderate": 30%,        # "How do I return this?"
    "complex": 10%          # "Explain your refund policy"
}
# Optimal: Route 60% to gpt-4o-mini → 40% cost savings

# User B: Code Generation
query_distribution = {
    "simple_faq": 10%,      # "What is Python?"
    "moderate": 30%,        # "Write a function to..."
    "complex": 60%          # "Debug this architecture..."
}
# Optimal: Route 60% to gpt-4o → Different pattern!

# The bandit learns YOUR pattern, not generic expectations
```

## The Data Moat

**Competitive Advantage**: The more you use Conduit, the smarter it gets for YOUR workload.

```python
# New deployment: Generic routing (not optimized)
cost_per_1000_queries = $5.00

# After 1000 queries: Learned YOUR patterns
cost_per_1000_queries = $2.50  # 50% savings

# After 10,000 queries: Highly optimized for YOUR use case
cost_per_1000_queries = $2.25  # Even better!

# Competitor starts from scratch: Back to $5.00
# Your data creates switching costs
```

**This is the moat**: Your usage data makes Conduit increasingly valuable to you specifically.

## Convergence Metrics

### How to Measure Convergence

```python
# Track distribution stability over rolling windows
window_size = 100

# Calculate variance in model selection
recent_selections = last_100_queries.model_counts()
variance = std(recent_selections)

# Converged when variance stabilizes
if variance < threshold:
    print("Bandit has converged!")
```

### Expected Convergence Timeline

Based on typical workloads:

- **Query 100**: Initial patterns visible, still exploring heavily
- **Query 200**: Moderate confidence, reasonable routing decisions
- **Query 500**: High confidence, mostly stable routing
- **Query 1000**: Converged, minimal changes to distributions

**Factors affecting convergence speed**:
- **Workload diversity**: More diverse = slower convergence
- **Feedback quality**: Better signals = faster convergence
- **Model differences**: Bigger capability gaps = faster convergence

## Implementation Details

### Current Implementation

```python
# conduit/engines/bandit.py
class ContextualBandit:
    def __init__(self, models: list[str]):
        self.model_states = {
            model: BanditState(alpha=1.0, beta=1.0)  # Uniform priors
            for model in models
        }

    def select(self, features: QueryFeatures) -> str:
        # Thompson Sampling: Sample from each Beta distribution
        samples = {
            model: np.random.beta(state.alpha, state.beta)
            for model, state in self.model_states.items()
        }
        return max(samples, key=samples.get)

    def update(self, model: str, reward: float,
               success_threshold: float = 0.7):
        if reward >= success_threshold:
            self.model_states[model].alpha += 1.0
        else:
            self.model_states[model].beta += 1.0
```

### Key Parameters

```python
# Success threshold: When is a query "successful"?
success_threshold = 0.7  # 70% reward or higher = success

# Initial priors: Where do we start?
alpha_init = 1.0  # Uniform (no prior knowledge)
beta_init = 1.0

# Feedback weights: How to balance signals?
explicit_weight = 0.7  # User ratings: 70%
implicit_weight = 0.3  # Behavioral signals: 30%
```

## Production Considerations

### Monitoring Learning Progress

```python
# Track metrics to validate learning
metrics = {
    "queries_processed": 1000,
    "cost_per_query": 0.0025,  # $2.50 per 1000
    "avg_latency": 1.2,        # seconds
    "error_rate": 0.05,        # 5%

    # Bandit-specific metrics
    "model_distribution": {
        "gpt-4o-mini": 0.60,   # 60% of queries
        "gpt-4o": 0.30,
        "claude-sonnet-4": 0.10
    },
    "exploration_rate": 0.05,  # 5% random exploration
    "convergence_score": 0.92  # High = stable routing
}
```

### When to Reset the Bandit

Consider resetting if:
- Workload changes dramatically (customer support → code generation)
- New models added to the pool
- Pricing changes significantly
- Model capabilities change (GPT-5 released, etc.)

```python
# Soft reset: Keep some prior knowledge
bandit.reset(retain_fraction=0.5)  # Keep 50% of learned distributions

# Hard reset: Start from scratch
bandit.reset(retain_fraction=0.0)  # Back to uniform priors
```

## References

- **Implementation**: `conduit/engines/bandit.py`
- **Feedback System**: `docs/IMPLICIT_FEEDBACK.md`
- **Cold Start Solutions**: `docs/COLD_START.md`
- **Benchmark Strategy**: `docs/BENCHMARK_STRATEGY.md`
