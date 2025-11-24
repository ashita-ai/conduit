# Troubleshooting Guide

**Purpose**: Debug common issues with Conduit's ML-powered routing system

**Last Updated**: 2025-11-24

---

## Table of Contents

1. [Poor Routing Decisions](#poor-routing-decisions)
2. [Slow Convergence](#slow-convergence)
3. [Model Deprecation Handling](#model-deprecation-handling)
4. [Debugging Tools & Commands](#debugging-tools--commands)
5. [Common Issues](#common-issues)

---

## Poor Routing Decisions

### Symptom: Always Choosing Cheapest Model

**Problem**: The bandit routes all queries to the cheapest model (e.g., GPT-4o-mini, Claude Haiku), ignoring quality.

**Root Causes**:
- **Exploitation too early**: Insufficient exploration of other models
- **Reward weights**: Cost weight too high relative to quality
- **Cold start**: Not enough data to learn quality differences

**Debug Steps**:

1. **Check exploration rate**:
```python
from conduit.engines.router import Router

router = Router()
stats = router.bandit.get_stats()

# For Epsilon-Greedy
print(f"Current epsilon: {stats.get('epsilon', 'N/A')}")
# Should be >0.05 during learning phase

# For UCB1/LinUCB
print(f"Arm pulls: {stats.get('arm_pulls', {})}")
# All arms should have >10 pulls for initial exploration

# For Thompson Sampling
print(f"Arm statistics: {stats.get('arm_statistics', {})}")
# Check alpha/beta values - high variance means more exploration
```

2. **Check reward weights**:
```python
from conduit.core.defaults import DEFAULT_REWARD_WEIGHTS

print(f"Reward weights: {DEFAULT_REWARD_WEIGHTS}")
# Default: {'quality': 0.70, 'cost': 0.20, 'latency': 0.10}
# If cost weight is too high, quality is undervalued
```

3. **Inspect recent routing decisions**:
```python
# Check last 10 routing decisions
stats = router.bandit.get_stats()
print(f"Total queries: {stats['total_queries']}")

# For each arm
for model_id, pulls in stats.get('arm_pulls', {}).items():
    mean_reward = stats.get('arm_mean_rewards', {}).get(model_id, 0.0)
    print(f"{model_id}: {pulls} pulls, mean reward: {mean_reward:.3f}")
```

**Solutions**:

**Solution 1: Increase Exploration (Epsilon-Greedy)**
```python
from conduit.engines.bandits import EpsilonGreedyBandit

# Increase epsilon to 20% exploration
bandit = EpsilonGreedyBandit(
    arms=arms,
    epsilon=0.20,  # Default: 0.10
    decay=0.999,   # Slow decay
    min_epsilon=0.05  # Higher minimum
)
```

**Solution 2: Reset Bandit (Start Fresh)**
```python
# Reset to explore all arms again
router.bandit.reset()

# Or create new router with different algorithm
from conduit.engines.router import Router

router = Router(algorithm="thompson_sampling")
# Thompson Sampling explores more naturally via sampling
```

**Solution 3: Adjust Reward Weights**
```python
# In .env file, increase quality weight
REWARD_WEIGHT_QUALITY=0.80  # Up from 0.70
REWARD_WEIGHT_COST=0.15     # Down from 0.20
REWARD_WEIGHT_LATENCY=0.05  # Down from 0.10

# Or in code:
from conduit.engines.bandits import LinUCBBandit

bandit = LinUCBBandit(
    arms=arms,
    reward_weights={
        "quality": 0.80,
        "cost": 0.15,
        "latency": 0.05
    }
)
```

**Solution 4: Use Contextual Thompson Sampling**
```python
from conduit.engines.router import Router

# Better exploration via Bayesian sampling
router = Router(algorithm="contextual_thompson_sampling")
# Naturally balances exploration/exploitation
```

### Symptom: Inconsistent Model Selection

**Problem**: Routing decisions seem random or change frequently for similar queries.

**Root Causes**:
- **High exploration rate**: Still in exploration phase
- **Similar arm rewards**: Models perform similarly, so sampling is noisy
- **Insufficient data**: Not enough queries to distinguish models

**Debug Steps**:

1. **Check exploration phase**:
```python
stats = router.bandit.get_stats()
total_queries = stats['total_queries']

# For Hybrid Routing
if hasattr(router, 'hybrid_router'):
    phase = router.hybrid_router.current_phase
    print(f"Phase: {phase}, Queries: {total_queries}")
    # Phase 1 (0-2000): UCB1 exploration expected
    # Phase 2 (2000+): LinUCB should be more stable

# For other algorithms
min_pulls = min(stats.get('arm_pulls', {}).values())
if min_pulls < 50:
    print(f"Warning: Some arms have <50 pulls (min: {min_pulls})")
    print("Still in exploration phase - inconsistency is normal")
```

2. **Check reward distributions**:
```python
import numpy as np

stats = router.bandit.get_stats()
rewards = stats.get('arm_mean_rewards', {})
reward_values = list(rewards.values())

if len(reward_values) > 1:
    std_dev = np.std(reward_values)
    print(f"Reward std dev: {std_dev:.3f}")
    # Low std dev (<0.1) means models perform similarly
    # High variance in selection is expected
```

**Solutions**:

**Solution 1: Wait for Convergence**
```python
# Check if you have enough data
stats = router.bandit.get_stats()
total = stats['total_queries']

if total < 500:
    print("⚠️  Still in early exploration (<500 queries)")
    print("Wait for more data before expecting stable routing")
elif total < 2000:
    print("⚠️  Approaching convergence (500-2000 queries)")
    print("Consider switching to LinUCB for contextual learning")
else:
    print("✅ Sufficient data (>2000 queries)")
    print("If still inconsistent, check reward distributions")
```

**Solution 2: Use Hybrid Routing for Faster Convergence**
```python
from conduit.engines.hybrid_router import HybridRouter

# Converges 30% faster than pure LinUCB
router = HybridRouter()
# Phase 1 (0-2000): Fast UCB1 convergence
# Phase 2 (2000+): Contextual LinUCB refinement
```

**Solution 3: Reduce Exploration After Learning**
```python
# For Epsilon-Greedy
stats = router.bandit.get_stats()
if stats['total_queries'] > 1000:
    # Manually reduce epsilon
    router.bandit.epsilon = 0.01  # Down to 1% exploration
```

---

## Slow Convergence

### Symptom: Regret Rate Not Decreasing

**Problem**: The cumulative regret keeps increasing or doesn't flatten out, indicating the bandit isn't learning efficiently.

**Root Causes**:
- **Poor context features**: Query features don't capture important distinctions
- **Feature correlation**: Redundant features adding noise
- **High dimensionality**: Too many features (387 dims) for available data
- **Non-stationarity**: Model quality/costs changing over time

**Debug Steps**:

1. **Calculate regret**:
```python
import numpy as np

# Estimate regret by comparing to oracle (always best model)
stats = router.bandit.get_stats()
total_queries = stats['total_queries']
mean_rewards = stats.get('arm_mean_rewards', {})

# Oracle: Best possible reward (assuming we knew best model)
oracle_reward = max(mean_rewards.values()) if mean_rewards else 1.0

# Current: Actual average reward
arm_pulls = stats.get('arm_pulls', {})
total_reward = sum(
    mean_rewards.get(model, 0) * pulls 
    for model, pulls in arm_pulls.items()
)
current_avg = total_reward / max(total_queries, 1)

# Regret
regret = oracle_reward - current_avg
regret_rate = regret / max(total_queries, 1)

print(f"Oracle reward: {oracle_reward:.3f}")
print(f"Current avg reward: {current_avg:.3f}")
print(f"Total regret: {regret:.3f}")
print(f"Regret rate: {regret_rate:.6f}")
# Regret rate should decrease over time
# >0.001 after 1000 queries indicates slow convergence
```

2. **Check feature importance** (LinUCB/Contextual algorithms):
```python
# For LinUCB: Check theta (feature weights)
if hasattr(router.bandit, 'theta'):
    for model_id, theta in router.bandit.theta.items():
        # First dimension is intercept
        # Dimensions 1-384 are embedding
        # Dimensions 385-387 are metadata
        
        embedding_weight = np.linalg.norm(theta[1:385])
        metadata_weight = np.linalg.norm(theta[385:388])
        
        print(f"{model_id}:")
        print(f"  Embedding weight: {embedding_weight:.3f}")
        print(f"  Metadata weight: {metadata_weight:.3f}")
        
        # If metadata weight >> embedding weight, 
        # embeddings might not be informative
```

3. **Analyze feature correlation**:
```python
from conduit.engines.analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

# Collect features from recent queries
queries = [
    "What is 2+2?",
    "Explain quantum mechanics",
    "Write a Python function to sort a list",
    # ... more diverse queries
]

features_list = []
for query in queries:
    from conduit.core.models import Query
    q = Query(text=query)
    features = await analyzer.analyze(q)
    features_list.append(features.embedding)

# Check correlation
import numpy as np
feature_matrix = np.array(features_list)
correlation = np.corrcoef(feature_matrix.T)

# High correlations (>0.9) indicate redundant features
high_corr = np.sum(np.abs(correlation) > 0.9) - len(queries)
print(f"High correlations: {high_corr}")
```

**Solutions**:

**Solution 1: Use PCA for Dimensionality Reduction**
```python
from conduit.engines.hybrid_router import HybridRouter

# Reduce 387 dims → 67 dims with PCA
router = HybridRouter(use_pca=True, pca_components=67)
# Reduces sample requirement by 75%
# Converges in 2,000-3,000 queries vs 10,000+ without PCA
```

**Solution 2: Add More Informative Features**
```python
# Extend QueryFeatures with domain-specific metadata
from conduit.core.models import QueryFeatures

# Example: Add query type classification
query_type = "code" if "function" in query.lower() else "general"
features.metadata["query_type"] = query_type

# Example: Add urgency/latency sensitivity
features.metadata["latency_sensitive"] = "fast" in query.lower()

# LinUCB will learn which models work best for each query type
```

**Solution 3: Switch to LinUCB for Contextual Learning**
```python
from conduit.engines.router import Router

# LinUCB learns query-specific patterns
router = Router(algorithm="linucb")
# Better than non-contextual (UCB1, Epsilon-Greedy) 
# when query characteristics matter
```

**Solution 4: Handle Non-Stationarity with Sliding Windows**
```python
from conduit.engines.bandits import LinUCBBandit

# Only keep last 1000 observations
# Adapts to model quality/cost changes
bandit = LinUCBBandit(
    arms=arms,
    window_size=1000  # Default: 0 (unlimited)
)

# For Thompson Sampling
from conduit.engines.bandits import ThompsonSamplingBandit

bandit = ThompsonSamplingBandit(
    arms=arms,
    window_size=1000
)
```

### Symptom: High Sample Requirement

**Problem**: Needs >10,000 queries before making good decisions.

**Root Causes**:
- **High dimensionality**: 387-dim features require many samples
- **Pure LinUCB**: No warm start, learns from scratch
- **Many arms**: More models = more exploration needed

**Solutions**:

**Solution 1: Use Hybrid Routing (RECOMMENDED)**
```python
from conduit.engines.hybrid_router import HybridRouter

# 30% faster convergence than pure LinUCB
# Production-ready in 2,000-3,000 queries vs 10,000-15,000
router = HybridRouter()
```

**Solution 2: Enable PCA**
```python
from conduit.engines.hybrid_router import HybridRouter

# Reduces sample requirement by 75%
router = HybridRouter(use_pca=True, pca_components=67)
```

**Solution 3: Reduce Number of Arms**
```python
from conduit.models import available_models

# Start with fewer high-quality models
arms = available_models(
    min_quality=0.80,  # Only quality models
    providers=["openai", "anthropic"]  # Limit to 2 providers
)
# Fewer arms = faster convergence
# Can add more models after initial learning
```

**Solution 4: Use Informed Priors (Thompson Sampling)**
```python
from conduit.engines.bandits import ThompsonSamplingBandit
from conduit.engines.bandits.base import ModelArm

# Set informed priors based on general knowledge
arms = [
    ModelArm(
        model_id="gpt-4o",
        provider="openai",
        model_name="gpt-4o",
        cost_per_input_token=0.0025,
        cost_per_output_token=0.010,
        expected_quality=0.95  # High prior for premium model
    ),
    ModelArm(
        model_id="gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        cost_per_input_token=0.00015,
        cost_per_output_token=0.0006,
        expected_quality=0.75  # Lower prior for budget model
    )
]

# Priors guide initial exploration
# See docs/COLD_START.md for details
```

---

## Model Deprecation Handling

### Symptom: Routing to Deprecated Models

**Problem**: Bandit continues routing to models that are deprecated, sunset, or no longer available.

**Root Causes**:
- **Stale model registry**: Not updating available models
- **Cached routing decisions**: Old data influencing new decisions
- **API key changes**: Model access lost but registry not updated

**Proper Deprecation Workflow**:

### Step 1: Identify Deprecated Models

```python
from conduit.models import supported_models

# Check current model registry
all_models = supported_models()

# Check which models you can actually use
from conduit.models import available_models
my_models = available_models()

# Compare
all_model_ids = {m.model_id for m in all_models}
my_model_ids = {m.model_id for m in my_models}

deprecated = all_model_ids - my_model_ids
print(f"Models you can't use: {deprecated}")
```

### Step 2: Transition Period (Recommended)

Don't immediately remove deprecated models. Use a transition period to migrate traffic:

```python
from conduit.engines.router import Router
from conduit.core.models import Query, QueryConstraints

# Week 1-2: Route existing queries normally
# Collect data on which queries use deprecated model

deprecated_queries = []

async def route_with_tracking(query: Query):
    decision = await router.route(query)
    
    if decision.selected_model == "deprecated-model-id":
        deprecated_queries.append({
            "query": query.text,
            "features": decision.features,
            "timestamp": decision.created_at
        })
    
    return decision

# Week 3-4: Add constraint to discourage deprecated model
query_with_constraint = Query(
    text="Your query here",
    constraints=QueryConstraints(
        preferred_provider="openai"  # Avoid deprecated provider
    )
)

# Week 5: Remove deprecated model from registry
```

### Step 3: Remove Deprecated Model

```python
from conduit.engines.router import Router

# Recreate router with updated model list
from conduit.models import available_models

# This automatically excludes models without API keys
current_models = available_models()

# Create new router (fresh bandit state)
router = Router(models=current_models)
```

### Step 4: Handle Active Sessions

If you have persistent bandit state (saved to database):

```python
# Load existing bandit state
stats = router.bandit.get_stats()

# Remove deprecated arm data
deprecated_model_id = "deprecated-model"

if hasattr(router.bandit, 'arm_pulls'):
    # UCB1, Epsilon-Greedy
    del router.bandit.arm_pulls[deprecated_model_id]
    del router.bandit.mean_reward[deprecated_model_id]
    
if hasattr(router.bandit, 'A'):
    # LinUCB
    del router.bandit.A[deprecated_model_id]
    del router.bandit.b[deprecated_model_id]

if hasattr(router.bandit, 'alpha'):
    # Thompson Sampling
    del router.bandit.alpha[deprecated_model_id]
    del router.bandit.beta[deprecated_model_id]

# Update arm list
router.bandit.arms = {
    k: v for k, v in router.bandit.arms.items() 
    if k != deprecated_model_id
}
router.bandit.arm_list = list(router.bandit.arms.values())
router.bandit.n_arms = len(router.bandit.arm_list)
```

### Example: OpenAI GPT-3.5-Turbo Deprecation

```python
# Before deprecation (GPT-3.5-Turbo available)
from conduit.models import available_models

models_before = available_models(providers=["openai"])
# Returns: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.

# After OpenAI deprecates GPT-3.5-Turbo
# llm-prices.com automatically updates

models_after = available_models(providers=["openai"])
# Returns: gpt-4o, gpt-4o-mini (no gpt-3.5-turbo)

# Conduit automatically adapts - no code changes needed!
# Dynamic model discovery handles deprecation
```

### Preventing Routing to Deprecated Models

**Approach 1: Use Dynamic Model Discovery (RECOMMENDED)**
```python
from conduit.models import available_models

# Always use fresh model list
def create_router():
    current_models = available_models()
    return Router(models=current_models)

# Call this when initializing or after API key changes
router = create_router()
```

**Approach 2: Add Model Validation**
```python
from conduit.engines.router import Router
from conduit.core.models import Query

async def route_with_validation(router: Router, query: Query):
    decision = await router.route(query)
    
    # Check if model is still available
    from conduit.models import available_models
    current_models = available_models()
    current_ids = {m.model_id for m in current_models}
    
    if decision.selected_model not in current_ids:
        # Model deprecated mid-session
        # Fallback to best available model
        fallback = max(current_models, key=lambda m: m.expected_quality)
        
        print(f"Warning: {decision.selected_model} deprecated, using {fallback.model_id}")
        decision.selected_model = fallback.model_id
        decision.reasoning += f" (fallback due to deprecation)"
    
    return decision
```

**Approach 3: Automated Model Refresh**
```python
import asyncio
from datetime import datetime, timedelta

class AutoRefreshRouter:
    def __init__(self, refresh_interval_hours: int = 24):
        self.router = self._create_router()
        self.last_refresh = datetime.now()
        self.refresh_interval = timedelta(hours=refresh_interval_hours)
    
    def _create_router(self):
        from conduit.models import available_models
        return Router(models=available_models())
    
    async def route(self, query):
        # Check if refresh needed
        if datetime.now() - self.last_refresh > self.refresh_interval:
            print("Refreshing model registry...")
            self.router = self._create_router()
            self.last_refresh = datetime.now()
        
        return await self.router.route(query)
```

---

## Debugging Tools & Commands

### Command-Line Debugging

**Check Current Model Registry**:
```bash
# List all supported models
python -c "
from conduit.models import supported_models
models = supported_models()
for m in models[:10]:  # First 10
    print(f'{m.model_id}: ${m.cost_per_input_token:.6f}/1K input')
"

# Check YOUR available models
python -c "
from conduit.models import available_models
models = available_models()
print(f'You can use {len(models)} models:')
for m in models:
    print(f'  {m.model_id}')
"
```

**Check Bandit Statistics**:
```bash
# Create script: debug_bandit.py
cat > debug_bandit.py << 'EOF'
import asyncio
from conduit.engines.router import Router

async def main():
    router = Router()
    stats = router.bandit.get_stats()
    
    print(f"Algorithm: {stats['name']}")
    print(f"Total queries: {stats['total_queries']}")
    print("\nArm Statistics:")
    
    if 'arm_pulls' in stats:
        for model_id, pulls in stats['arm_pulls'].items():
            mean_reward = stats.get('arm_mean_rewards', {}).get(model_id, 0.0)
            print(f"  {model_id}:")
            print(f"    Pulls: {pulls}")
            print(f"    Mean reward: {mean_reward:.3f}")

asyncio.run(main())
EOF

python debug_bandit.py
```

**Test Routing Decision**:
```bash
# Create script: test_route.py
cat > test_route.py << 'EOF'
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

async def main():
    router = Router()
    
    # Test different query types
    queries = [
        "What is 2+2?",
        "Explain quantum entanglement in detail",
        "Write a Python function to implement quicksort"
    ]
    
    for query_text in queries:
        query = Query(text=query_text)
        decision = await router.route(query)
        
        print(f"\nQuery: {query_text}")
        print(f"Selected: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Reasoning: {decision.reasoning}")

asyncio.run(main())
EOF

python test_route.py
```

### Python REPL Debugging

**Interactive Exploration**:
```python
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

# Create router
router = Router()

# Check algorithm
print(f"Algorithm: {router.bandit.name}")

# Get statistics
stats = router.bandit.get_stats()
print(f"Total queries: {stats['total_queries']}")

# Test routing
async def test():
    query = Query(text="Explain machine learning")
    decision = await router.route(query)
    return decision

decision = asyncio.run(test())
print(f"Selected: {decision.selected_model}")
print(f"Confidence: {decision.confidence:.0%}")
```

**Compare Algorithms**:
```python
import asyncio
from conduit.engines.router import Router
from conduit.core.models import Query

algorithms = ["ucb1", "epsilon_greedy", "thompson_sampling", "linucb"]

async def compare():
    query = Query(text="What is machine learning?")
    
    for alg in algorithms:
        router = Router(algorithm=alg)
        decision = await router.route(query)
        
        print(f"\n{alg.upper()}:")
        print(f"  Model: {decision.selected_model}")
        print(f"  Confidence: {decision.confidence:.0%}")

asyncio.run(compare())
```

### Logging Configuration

**Enable Debug Logging**:
```python
# In your application
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or in .env
LOG_LEVEL=DEBUG
```

**Conduit-Specific Logging**:
```python
import logging

# Enable specific modules
logging.getLogger('conduit.engines.router').setLevel(logging.DEBUG)
logging.getLogger('conduit.engines.bandits').setLevel(logging.DEBUG)
logging.getLogger('conduit.engines.analyzer').setLevel(logging.INFO)
```

---

## Common Issues

### Issue: "No API key found for provider"

**Symptom**: `ValueError: No API key found for openai provider`

**Solution**:
```bash
# Check .env file
cat .env | grep API_KEY

# Should have at least one:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...

# If missing, add your API key
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### Issue: "Model not found in registry"

**Symptom**: `KeyError: 'model-id' not in arms`

**Solution**:
```python
# Check if model is supported
from conduit.models import supported_models

models = supported_models()
model_ids = [m.model_id for m in models]

if "your-model-id" not in model_ids:
    print(f"Model not supported. Available: {model_ids[:5]}")
    # Use a supported model instead
```

### Issue: Router Making Same Decision Every Time

**Symptom**: All queries routed to same model regardless of content

**Cause**: No exploration happening

**Solution**:
```python
# Check if exploration is enabled
stats = router.bandit.get_stats()

# For Epsilon-Greedy
if 'epsilon' in stats:
    if stats['epsilon'] < 0.01:
        # Too low - reset with higher epsilon
        from conduit.engines.bandits import EpsilonGreedyBandit
        router.bandit = EpsilonGreedyBandit(
            arms=router.bandit.arm_list,
            epsilon=0.10
        )

# For UCB1/LinUCB
if 'arm_pulls' in stats:
    min_pulls = min(stats['arm_pulls'].values())
    if min_pulls == 0:
        # Some arms never tried - still in exploration
        # This is normal for first K queries where K = number of arms
        pass
```

### Issue: High Costs Despite Cost Optimization

**Symptom**: Bills higher than expected even with cost-conscious routing

**Debug**:
```python
# Check reward weights
from conduit.core.defaults import DEFAULT_REWARD_WEIGHTS
print(f"Weights: {DEFAULT_REWARD_WEIGHTS}")

# Check which models being used
stats = router.bandit.get_stats()
for model_id, pulls in stats['arm_pulls'].items():
    arm = router.bandit.arms[model_id]
    avg_cost = (arm.cost_per_input_token + arm.cost_per_output_token) / 2
    usage_pct = pulls / stats['total_queries'] * 100
    
    print(f"{model_id}:")
    print(f"  Usage: {usage_pct:.1f}%")
    print(f"  Avg cost/1K tokens: ${avg_cost:.4f}")
```

**Solution**:
```python
# Increase cost weight in .env
REWARD_WEIGHT_QUALITY=0.50  # Down from 0.70
REWARD_WEIGHT_COST=0.40     # Up from 0.20
REWARD_WEIGHT_LATENCY=0.10

# Or use constraints
from conduit.core.models import Query, QueryConstraints

query = Query(
    text="Your query",
    constraints=QueryConstraints(
        max_cost=0.001  # Max $0.001 per query
    )
)
```

### Issue: Slow Query Analysis

**Symptom**: High latency before routing decision

**Cause**: Embedding generation is slow

**Debug**:
```python
import time
from conduit.engines.analyzer import QueryAnalyzer
from conduit.core.models import Query

analyzer = QueryAnalyzer()
query = Query(text="Test query for timing")

start = time.time()
features = await analyzer.analyze(query)
elapsed = time.time() - start

print(f"Analysis took {elapsed:.3f}s")
# Should be <0.5s for most queries
# >1s indicates embedding model issue
```

**Solution**:
```python
# Use smaller embedding model (in .env)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Faster, smaller

# Or cache embeddings for repeated queries
# (Conduit does this automatically with Redis)

# Check Redis
REDIS_URL=redis://localhost:6379  # Enable caching
```

### Issue: "Arm not found" After Model Update

**Symptom**: `KeyError: 'model-id'` after updating model registry

**Cause**: Bandit state has old model IDs

**Solution**:
```python
# Reset bandit when model list changes
router.bandit.reset()

# Or create new router
from conduit.models import available_models
router = Router(models=available_models())
```

---

## Getting Help

### Check Documentation
- **Architecture**: `docs/ARCHITECTURE.md`
- **Bandit Algorithms**: `docs/BANDIT_ALGORITHMS.md`
- **Cold Start**: `docs/COLD_START.md`
- **Hybrid Routing**: `docs/HYBRID_ROUTING.md`
- **Model Discovery**: `docs/MODEL_DISCOVERY.md`

### Run Examples
```bash
# Quick start
python examples/01_quickstart/hello_world.py

# Routing examples
python examples/02_routing/basic_routing.py
python examples/02_routing/hybrid_routing.py

# Optimization examples
python examples/03_optimization/caching.py
```

### Enable Verbose Logging
```bash
# In .env
LOG_LEVEL=DEBUG

# Run your application
python your_app.py 2>&1 | tee debug.log

# Search for errors
grep ERROR debug.log
grep WARNING debug.log
```

### Report Issues
- GitHub Issues: https://github.com/ashita-ai/conduit/issues
- Include:
  - Conduit version
  - Algorithm used
  - Number of queries processed
  - Error messages and stack traces
  - Minimal reproduction example
