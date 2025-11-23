# Cold Start Problem & Solutions

**Problem**: How do we make good routing decisions before the bandit has learned anything?

## Problem Definition

### The Cold Start Dilemma

With uniform priors (alpha=1, beta=1), the bandit starts knowing nothing:

```python
# Query 1: All models look equally good
models = {
    "gpt-4o-mini": Beta(α=1, β=1) → 50% expected success, high uncertainty
    "gpt-4o": Beta(α=1, β=1) → 50% expected success, high uncertainty
    "claude-sonnet-4": Beta(α=1, β=1) → 50% expected success, high uncertainty
}

# Thompson Sampling samples randomly from these distributions
# Result: Completely random routing initially
```

### Consequences

**Poor User Experience**:
- Complex queries might route to weak models
- Simple queries might route to expensive models
- Inconsistent quality during learning phase

**Wasted Resources**:
- Paying for premium models on simple queries
- Failing complex queries with budget models
- User frustration leading to abandonment

**Slow Convergence**:
- Need 100-200 queries before reasonable routing
- Each mistake is a learning opportunity (but also a cost)
- Risk of users abandoning before system learns

## Solution Approaches

### 1. Informed Priors (RECOMMENDED)

**Concept**: Start with reasonable expectations instead of uniform ignorance

**Implementation**:
```python
# Based on general LLM benchmarks and common knowledge
model_priors = {
    "gpt-4o-mini": {
        "alpha": 7,  # Expect ~70% success
        "beta": 3,
        "rationale": "Budget model, decent quality, fails on complex tasks"
    },
    "gpt-4o": {
        "alpha": 9,  # Expect ~90% success
        "beta": 1,
        "rationale": "Premium model, excellent quality, rarely fails"
    },
    "claude-sonnet-4": {
        "alpha": 9,  # Expect ~90% success
        "beta": 1,
        "rationale": "Premium model, excellent quality"
    },
    "claude-haiku": {
        "alpha": 6,  # Expect ~60% success
        "beta": 4,
        "rationale": "Ultra-budget model, okay for simple tasks"
    }
}

# Initialize bandit with informed priors
bandit = ContextualBandit(
    models=list(model_priors.keys()),
    priors=model_priors
)
```

**Effect on Cold Start**:
```python
# Before (uniform priors):
# Query 1: Random routing (33% chance each model)
# Query 50: Still exploring heavily, inconsistent

# After (informed priors):
# Query 1: Biased toward premium for uncertain queries (safer)
# Query 50: Already showing reasonable patterns
```

**Pros**:
- Better initial decisions
- Faster convergence (fewer wasted queries)
- Leverages public knowledge (benchmarks, pricing)
- Easy to implement

**Cons**:
- Priors might not match specific workload
- Requires maintaining prior estimates
- Could bias against actually-good budget models

**When to Use**: Always - this is low-risk, high-reward

### 2. Contextual Heuristics + Learning Blend

**Concept**: Use simple rules initially, fade to learned behavior

**Implementation**:
```python
class HybridRouter:
    def route(self, query: str, query_count: int):
        features = self.analyzer.analyze(query)

        # Phase 1: Pure heuristics (queries 1-50)
        if query_count < 50:
            return self._heuristic_route(features)

        # Phase 2: Blended (queries 50-200)
        elif query_count < 200:
            # Gradually shift from heuristics to bandit
            blend_ratio = (query_count - 50) / 150

            if random.random() < blend_ratio:
                return self.bandit.select(features)  # Learned routing
            else:
                return self._heuristic_route(features)  # Rule-based

        # Phase 3: Pure bandit (queries 200+)
        else:
            return self.bandit.select(features)

    def _heuristic_route(self, features: QueryFeatures) -> str:
        """Simple rule-based routing."""
        if features.complexity < 0.3:
            return "gpt-4o-mini"  # Simple → cheap

        elif features.complexity > 0.7:
            return "gpt-4o"  # Complex → premium

        elif features.domain == "code":
            return "gpt-4o"  # Code → strong model

        else:
            return "gpt-4o-mini"  # Default to budget
```

**Effect on Cold Start**:
```python
# Query 1-50: Deterministic routing based on complexity
# - Simple queries → gpt-4o-mini (save money)
# - Complex queries → gpt-4o (ensure quality)

# Query 50-200: Gradual handoff
# - 0% bandit at query 50
# - 50% bandit at query 125
# - 100% bandit at query 200

# Query 200+: Fully learned routing (may beat heuristics!)
```

**Pros**:
- Protects user experience early
- Smooth transition to learning
- Leverages existing QueryAnalyzer features
- Can't be worse than pure random

**Cons**:
- Heuristics might teach bad habits
- Complexity in blending logic
- Need to tune blend schedule
- Heuristics need maintenance

**When to Use**: When user experience during cold start is critical

### 3. Conservative Bias with Decay

**Concept**: Bias toward safer (premium) models initially, gradually trust the bandit

**Implementation**:
```python
def select_with_safety_bias(features: QueryFeatures, query_count: int) -> str:
    # Calculate bias factor (1.0 → 0.0 over first 100 queries)
    bias_factor = max(0, 1.0 - (query_count / 100))

    # Sample from bandit as usual
    samples = {
        model: np.random.beta(state.alpha, state.beta)
        for model, state in self.model_states.items()
    }

    # Apply safety bias to premium models
    if bias_factor > 0:
        premium_models = ["gpt-4o", "claude-sonnet-4"]
        for model in premium_models:
            samples[model] *= (1 + bias_factor * 0.5)  # Boost premium

    return max(samples, key=samples.get)
```

**Effect on Cold Start**:
```python
# Query 1: Premium models 50% more likely to be selected
# Query 50: Premium models 25% more likely
# Query 100: No bias, pure Thompson Sampling
```

**Pros**:
- Protects against catastrophic failures early
- Smooth decay to unbiased routing
- Configurable safety level
- Good for risk-averse users

**Cons**:
- Wastes money on premium models unnecessarily
- Slows learning (less exploration)
- Arbitrary bias parameters
- May never learn budget models work fine

**When to Use**: For enterprise customers who prioritize safety over cost

### 4. Accelerated Feedback Collection

**Concept**: Gather feedback more aggressively during cold start

**Implementation**:
```python
class ColdStartAccelerator:
    def get_reward(self, feedback: ImplicitFeedback, query_count: int) -> float:
        # During cold start, use stricter criteria for "success"
        if query_count < 100:
            # Lower latency tolerance
            if feedback.latency_seconds > 5:  # vs 10s normally
                return 0.3  # Penalize heavily

            # Stricter error detection
            if feedback.error_occurred:
                return 0.0  # Hard failure

            # Higher success threshold
            return 0.9 if self._is_excellent(feedback) else 0.5

        else:
            # Normal feedback after cold start
            return self._standard_reward(feedback)

    def force_exploration(self, query_count: int) -> str:
        """Ensure each model tried at least N times."""
        min_trials = 10

        for model, state in self.bandit.model_states.items():
            trials = state.alpha + state.beta - 2  # Total trials
            if trials < min_trials:
                # Force this model to be tried
                return model

        # All models have minimum trials
        return None
```

**Effect on Cold Start**:
```python
# Queries 1-100:
# - Lower thresholds for success
# - Higher penalties for failures
# - Forced exploration of untried models
# Result: Faster accumulation of strong signals

# After 100 queries:
# - More confident distributions
# - Better learned patterns
# - Shorter cold start period (100 vs 200 queries)
```

**Pros**:
- Faster convergence
- Ensures balanced exploration
- No heuristic bias
- Learns your patterns faster

**Cons**:
- Might over-penalize during learning
- Forces suboptimal choices
- More complex reward logic
- Could discourage exploration later

**When to Use**: When you want shortest possible cold start period

### 5. Transfer Learning from Benchmarks

**Concept**: Use public benchmark data to initialize priors

**Implementation**:
```python
# Public benchmark results (e.g., MMLU, HumanEval)
benchmark_scores = {
    "gpt-4o": {
        "mmlu": 0.88,      # 88% on MMLU
        "humaneval": 0.90,  # 90% on HumanEval
        "average": 0.89
    },
    "claude-sonnet-4": {
        "mmlu": 0.86,
        "humaneval": 0.88,
        "average": 0.87
    },
    "gpt-4o-mini": {
        "mmlu": 0.72,
        "humaneval": 0.75,
        "average": 0.74
    }
}

def benchmark_to_prior(score: float, confidence: int = 20) -> tuple[float, float]:
    """Convert benchmark score to Beta distribution parameters.

    Args:
        score: Success rate from benchmark (0-1)
        confidence: How many "virtual trials" the benchmark represents

    Returns:
        (alpha, beta) parameters for Beta distribution
    """
    alpha = score * confidence
    beta = (1 - score) * confidence
    return (alpha, beta)

# Initialize with benchmark-informed priors
priors = {
    model: benchmark_to_prior(scores["average"])
    for model, scores in benchmark_scores.items()
}

# Example output:
# gpt-4o: Beta(α=17.8, β=2.2)  # Strong prior for 89% success
# gpt-4o-mini: Beta(α=14.8, β=5.2)  # Weaker prior for 74% success
```

**Effect on Cold Start**:
```python
# Query 1: Already knows gpt-4o > gpt-4o-mini (from benchmarks)
# Query 50: Refining benchmark knowledge with workload-specific patterns
# Query 200: Mostly workload-specific, benchmark influence fading
```

**Pros**:
- Data-driven priors (not arbitrary)
- Reflects real model capabilities
- Free data source (public benchmarks)
- Continuously updatable

**Cons**:
- Benchmarks might not match your workload
- Adds dependency on external data
- Need to maintain benchmark pipeline
- Generic vs specific trade-off

**When to Use**: For research-backed initialization

### 6. Contextual Features from Query Analysis

**Concept**: Use extracted query features for smarter initial routing

**Implementation**:
```python
def feature_based_cold_start(features: QueryFeatures) -> str:
    """Route based on query features during cold start."""

    # Code queries → Strong model
    if features.domain == "code":
        if features.complexity > 0.6:
            return "gpt-4o"  # Complex code needs premium
        else:
            return "gpt-4o-mini"  # Simple code snippets

    # Math/logic → Premium model
    elif features.domain == "math" or features.domain == "logic":
        return "gpt-4o"  # Precision matters

    # Creative writing → Cheaper models work
    elif features.domain == "creative":
        return "claude-sonnet-4" if features.complexity > 0.7 else "gpt-4o-mini"

    # FAQ/simple → Budget model
    elif features.complexity < 0.3:
        return "gpt-4o-mini"

    # Unknown/complex → Premium model (safe default)
    else:
        return "gpt-4o"
```

**Effect on Cold Start**:
```python
# Already leveraging QueryAnalyzer features:
features = QueryFeatures(
    complexity=0.8,      # High complexity
    domain="code",       # Code generation
    sentiment=0.0,       # Neutral
    embedding=[...]      # Semantic vector
)

# Immediate smart routing without learning
# "Write a REST API" → gpt-4o (code + complex)
# "What is Python?" → gpt-4o-mini (code + simple)
```

**Pros**:
- Leverages existing infrastructure
- Workload-aware from query 1
- No additional dependencies
- Can combine with bandit learning

**Cons**:
- Feature extraction must be accurate
- Rules still somewhat arbitrary
- Doesn't capture all nuances
- Need to maintain domain/complexity logic

**When to Use**: We already have QueryAnalyzer - use it!

### 7. Epsilon-Greedy with Decay

**Concept**: Explicit exploration/exploitation trade-off

**Implementation**:
```python
def epsilon_greedy_select(query_count: int) -> str:
    # Decay epsilon from 0.3 to 0.05 over first 200 queries
    epsilon = max(0.05, 0.3 - (query_count / 200) * 0.25)

    if random.random() < epsilon:
        # Explore: Try random model
        return random.choice(self.models)
    else:
        # Exploit: Use best known model
        return self._best_model()

    def _best_model(self) -> str:
        """Select model with highest mean success rate."""
        return max(
            self.model_states.items(),
            key=lambda x: x[1].alpha / (x[1].alpha + x[1].beta)
        )[0]
```

**Effect on Cold Start**:
```python
# Query 1: epsilon=0.30 → 30% random, 70% best
# Query 100: epsilon=0.18 → 18% random, 82% best
# Query 200: epsilon=0.05 → 5% random, 95% best
```

**Pros**:
- Simple and well-understood
- Explicit control over exploration
- Guaranteed minimum exploration
- Easy to tune

**Cons**:
- Less sophisticated than Thompson Sampling
- Deterministic exploitation (no uncertainty modeling)
- Arbitrary epsilon schedule
- Thompson already handles this well!

**When to Use**: Research comparison, not recommended for production

## Recommended Strategy

### Tier 1: Implement Immediately

**1. Informed Priors** + **2. Contextual Heuristics**

```python
# Initialize with informed priors
bandit = ContextualBandit(
    models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"],
    priors={
        "gpt-4o-mini": (7, 3),      # 70% expected
        "gpt-4o": (9, 1),            # 90% expected
        "claude-sonnet-4": (9, 1)    # 90% expected
    }
)

# Use contextual heuristics for first 50 queries
if query_count < 50:
    # Simple heuristic based on features
    if features.complexity < 0.3:
        model = "gpt-4o-mini"
    elif features.complexity > 0.7 or features.domain == "code":
        model = "gpt-4o"
    else:
        model = bandit.select(features)  # Let bandit explore middle
else:
    # Trust the bandit after 50 queries
    model = bandit.select(features)
```

**Expected Results**:
- Cold start period: 2,500-5,000 queries (vs 5,000-10,000 without)
- Convergence: 7,500-12,000 queries (vs 15,000-20,000 without)
- 30-50% reduction in learning phase cost
- Better initial routing quality (fewer expensive mistakes)
- Smooth transition to learned behavior
- Minimal added complexity

### Tier 2: Add for Enterprise

**3. Conservative Bias** (optional safety net)

```python
# For risk-averse customers
if user.risk_tolerance == "low":
    model = select_with_safety_bias(features, query_count)
else:
    model = bandit.select(features)
```

### Tier 3: Future Research

**5. Transfer Learning** from benchmarks (interesting but complex)

## Implementation Roadmap

### Phase 1: Informed Priors (EASY WIN)

```python
# Add to ContextualBandit.__init__
def __init__(
    self,
    models: list[str],
    initial_alpha: float = 1.0,  # NEW: Allow custom priors
    initial_beta: float = 1.0,   # NEW
    priors: dict[str, tuple[float, float]] | None = None  # NEW
):
    if priors:
        # Use custom priors
        self.model_states = {
            model: BanditState(alpha=priors[model][0], beta=priors[model][1])
            for model in models
        }
    else:
        # Default to uniform
        self.model_states = {
            model: BanditState(alpha=initial_alpha, beta=initial_beta)
            for model in models
        }
```

### Phase 2: Contextual Heuristics (MEDIUM EFFORT)

```python
# Add to Router class
class Router:
    def __init__(self, cold_start_queries: int = 50):
        self.cold_start_queries = cold_start_queries
        self.query_count = 0

    async def route(self, query: Query) -> RoutingDecision:
        features = await self.analyzer.analyze(query.text)

        # Cold start heuristic
        if self.query_count < self.cold_start_queries:
            model = self._cold_start_heuristic(features)
        else:
            model = self.bandit.select(features)

        self.query_count += 1
        return RoutingDecision(selected_model=model, ...)

    def _cold_start_heuristic(self, features: QueryFeatures) -> str:
        # Simple rules based on existing features
        if features.complexity < 0.3:
            return "gpt-4o-mini"
        elif features.complexity > 0.7:
            return "gpt-4o"
        else:
            return self.bandit.select(features)  # Explore middle ground
```

### Phase 3: Conservative Bias (OPTIONAL)

```python
# Add as configuration option
router = Router(
    cold_start_mode="conservative",  # "aggressive", "balanced", "conservative"
    cold_start_queries=100
)
```

## Measuring Cold Start Performance

### Key Metrics

```python
cold_start_metrics = {
    # Learning speed (REVISED - realistic expectations)
    "queries_to_basic_patterns": 5000,   # Initial patterns established
    "queries_to_convergence": 12000,     # Fully converged (with cold start solutions)
    "queries_to_convergence_baseline": 17500,  # Without cold start solutions

    # Quality during phases
    "phase_1_error_rate": 0.12,      # Errors in first 5k queries
    "phase_2_error_rate": 0.08,      # Errors queries 5k-15k
    "phase_3_error_rate": 0.06,      # Errors after convergence

    # Cost efficiency by phase
    "phase_1_cost_per_1k": 2.30,     # First 5k queries (learning)
    "phase_2_cost_per_1k": 1.90,     # Queries 5k-15k (refining)
    "phase_3_cost_per_1k": 1.60,     # Post-convergence (optimal)

    # User experience
    "phase_1_satisfaction": 0.75,    # User ratings during cold start
    "phase_2_satisfaction": 0.85,    # User ratings during learning
    "phase_3_satisfaction": 0.92     # User ratings post-convergence
}
```

### Success Criteria (Revised)

**Good Cold Start Solution**:
- Convergence in < 12,000 queries (vs 15,000-20,000 without)
- Error rate < 12% during cold start phase (0-5k queries)
- Error rate < 8% during learning phase (5k-15k queries)
- Cost within 30% of post-convergence optimal during cold start
- Cost within 15% of optimal during learning phase
- User satisfaction > 75% during cold start

**Excellent Cold Start Solution**:
- Convergence in < 10,000 queries
- Error rate < 10% during cold start
- Cost within 25% of optimal during cold start
- Demonstrable improvement curve (visible progress)

## See Also

- [HYBRID_ROUTING.md](./HYBRID_ROUTING.md) - UCB1→LinUCB warm start (30% faster convergence)
- [PCA_GUIDE.md](./PCA_GUIDE.md) - Dimensionality reduction (75% sample reduction)
- **Combined Impact**: Hybrid + PCA = 1,500-2,500 queries to production (vs 10,000+ baseline)
- [BANDIT_ALGORITHMS.md](./BANDIT_ALGORITHMS.md) - Algorithm details
- [BANDIT_TRAINING.md](./BANDIT_TRAINING.md) - How online learning works
