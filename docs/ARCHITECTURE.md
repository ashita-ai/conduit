# Conduit - System Architecture

**Version**: 0.1.0
**Last Updated**: 2025-11-27

---

## Quick Reference: How Components Work Together

This section explains how Conduit's key components interact.

### Component Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CONDUIT ROUTING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   QUERY INPUT                                                                    │
│        │                                                                         │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         QUERY ANALYZER                                   │   │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │   │
│   │  │  Embedding  │───▶│  PCA        │───▶│  Feature    │                  │   │
│   │  │  Provider   │    │ (optional)  │    │  Vector     │                  │   │
│   │  │  384-1536d  │    │  64d        │    │  66-386d    │                  │   │
│   │  └─────────────┘    └─────────────┘    └─────────────┘                  │   │
│   │                                              │                           │   │
│   │  + token_count, complexity_score                │                           │   │
│   └──────────────────────────────────────────────┼──────────────────────────┘   │
│                                                  │                               │
│                                                  ▼                               │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      BANDIT ALGORITHM (Router)                          │   │
│   │                                                                          │   │
│   │   Default: Thompson Sampling        Optional: LinUCB (contextual)       │   │
│   │   ┌─────────────────┐             ┌─────────────────┐                   │   │
│   │   │    Thompson     │             │     LinUCB      │                   │   │
│   │   │   Sampling      │     OR      │  (uses features) │                   │   │
│   │   │  (Bayesian)     │             │  Smart routing   │                   │   │
│   │   └─────────────────┘             └─────────────────┘                   │   │
│   │                                                                          │   │
│   └──────────────────────────────────────────────┬──────────────────────────┘   │
│                                                  │                               │
│                                                  ▼                               │
│                                         SELECTED MODEL                           │
│                                         (gpt-4o-mini, etc.)                     │
│                                                  │                               │
│                                                  ▼                               │
│                                          LLM RESPONSE                            │
│                                                  │                               │
│                    ┌─────────────────────────────┼─────────────────────────┐    │
│                    │                             │                         │    │
│                    ▼                             ▼                         ▼    │
│            ┌─────────────┐              ┌─────────────┐           ┌───────────┐ │
│            │  Explicit   │              │  Implicit   │           │  Arbiter  │ │
│            │  Feedback   │              │  Feedback   │           │  (async)  │ │
│            │  (user)     │              │  (system)   │           │  LLM-judge│ │
│            └─────────────┘              └─────────────┘           └───────────┘ │
│                    │                             │                         │    │
│                    └─────────────────────────────┼─────────────────────────┘    │
│                                                  │                               │
│                                                  ▼                               │
│                                         ┌─────────────┐                         │
│                                         │   REWARD    │                         │
│                                         │ CALCULATION │                         │
│                                         │ 0.7q+0.2c+  │                         │
│                                         │   0.1l      │                         │
│                                         └─────────────┘                         │
│                                                  │                               │
│                                                  ▼                               │
│                                         ┌─────────────┐                         │
│                                         │   BANDIT    │                         │
│                                         │   UPDATE    │◀──── Learning Loop      │
│                                         └─────────────┘                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Component Interactions

| Component | Purpose | Inputs | Outputs |
|-----------|---------|--------|---------|
| **Embedding Provider** | Convert text to vectors | Query text | 384-1536 dim vector |
| **PCA** | Compress embeddings | Raw embedding | 64 dim embedding |
| **Query Analyzer** | Extract routing features | Query text | QueryFeatures (embedding + metadata) |
| **Thompson Sampling** | Default routing (Bayesian) | None (non-contextual) | Model selection |
| **LinUCB Bandit** | Contextual routing | QueryFeatures | Model selection |
| **Router** | Algorithm management | Query | RoutingDecision |
| **Arbiter** | Quality evaluation | Query + Response | Quality score (0-1) |
| **Feedback Loop** | Learning signal | Feedback sources | Reward → Bandit update |

### When Each Component Is Used

```
Algorithm Selection (Router initialization):

┌─────────────────────────────────────────────────────────────────────┐
│  Thompson Sampling (DEFAULT)     │  LinUCB (contextual)            │
├─────────────────────────────────────────────────────────────────────┤
│  Embedding: ❌ Not needed         │  Embedding: ✅ Required         │
│  PCA:       ❌ Not needed         │  PCA:       Optional (386→66)  │
│  Best for:  Cold start, simple   │  Best for:  Query-specific      │
│  Cost:      Lowest               │  Cost:      Higher (embeddings) │
└─────────────────────────────────────────────────────────────────────┘

# Default (Thompson Sampling) - no feature extraction needed
router = Router()  # Uses thompson_sampling

# Contextual (LinUCB) - uses query features
router = Router(algorithm="linucb")

Arbiter:        ✅ Always available (async, sampled at 10%)
```

### Arbiter vs Bandit (Common Confusion)

| Aspect | Bandit (Thompson/LinUCB) | Arbiter |
|--------|--------------------------|---------|
| **Purpose** | SELECT which model to use | EVALUATE response quality |
| **When** | Before LLM call | After LLM call |
| **Blocking** | Yes (must select model) | No (async, background) |
| **Cost** | Free (math only) | LLM call (~$0.001/eval) |
| **Output** | Model ID | Quality score (0-1) |

### PCA: When and Why

**Without PCA (default):**
```
Query → Embedding (384d) → Feature Vector (386d) → LinUCB
                                                   386x386 matrices
                                                   ~150KB per arm
```

**With PCA:**
```
Query → Embedding (384d) → PCA (64d) → Feature Vector (66d) → LinUCB
                                                              66x66 matrices
                                                              ~4KB per arm
```

**Trade-offs:**
- PCA requires fitting on training data first
- Reduces memory ~40x, speeds up matrix operations
- Slight information loss (typically retains 95%+ variance)
- Most beneficial with high-dim embeddings (OpenAI 1536d → 64d)

---

## Feedback Loop Closure: How Learning Works

**This is the CORE VALUE PROPOSITION of Conduit.** The system learns from feedback to make better routing decisions over time.

### Complete Feedback Loop Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FEEDBACK LOOP CLOSURE                                  │
│                                                                              │
│  1. ROUTE QUERY                                                              │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ Query → Router.route() → HybridRouter.route()                 │      │
│     │                                                                 │      │
│     │ Phase 1 (queries < 2000): Thompson Sampling                    │      │
│     │   → UCB1.select_arm() [no features, fast exploration]          │      │
│     │                                                                 │      │
│     │ Phase 2 (queries ≥ 2000): LinUCB                               │      │
│     │   → LinUCB.select_arm(features)                                │      │
│     │   → Compute UCB scores for each arm:                           │      │
│     │      UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)         │      │
│     │      where theta = A_inv @ b (ridge regression coefficients)   │      │
│     │   → Return arm with highest UCB                                │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                             │                                                │
│                             ▼                                                │
│  2. EXECUTE                                                                  │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ Execute LLM call with selected model                           │      │
│     │ Track: cost, quality, latency                                  │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                             │                                                │
│                             ▼                                                │
│  3. COLLECT FEEDBACK                                                         │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ Router.update(                                                 │      │
│     │   model_id=selected_model,                                     │      │
│     │   cost=0.001,              # Actual execution cost             │      │
│     │   quality_score=0.95,      # User rating or arbiter eval       │      │
│     │   latency=0.5,             # Response time                     │      │
│     │   features=decision.features  # CRITICAL: real query features  │      │
│     │ )                                                               │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                             │                                                │
│                             ▼                                                │
│  4. CALCULATE REWARD                                                         │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ BanditFeedback.calculate_reward():                             │      │
│     │                                                                 │      │
│     │ reward = quality_weight * quality                              │      │
│     │        + cost_weight * (1 / (1 + cost))                        │      │
│     │        + latency_weight * (1 / (1 + latency))                  │      │
│     │                                                                 │      │
│     │ Default weights:                                                │      │
│     │   - quality: 70%   (normalize_quality: clamp to [0,1])         │      │
│     │   - cost: 20%      (normalize_cost: 1/(1+cost), asymptotic)    │      │
│     │   - latency: 10%   (normalize_latency: 1/(1+latency))          │      │
│     │                                                                 │      │
│     │ Example:                                                        │      │
│     │   quality=0.95, cost=0.001, latency=0.5                        │      │
│     │   → reward = 0.7*0.95 + 0.2*0.999 + 0.1*0.667 = 0.932          │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                             │                                                │
│                             ▼                                                │
│  5. UPDATE WEIGHTS (LinUCB with Sherman-Morrison)                            │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ HybridRouter.update(feedback, features)                        │      │
│     │   → LinUCB.update(feedback, features)                          │      │
│     │                                                                 │      │
│     │ Extract feature vector:                                         │      │
│     │   x = [embedding, token_count, complexity]                     │      │
│     │   Dims: 386 (FastEmbed 384d) or 1538 (OpenAI 1536d)           │      │
│     │   With PCA: 66 (FastEmbed) or 130 (OpenAI)                    │      │
│     │                                                                 │      │
│     │ Incremental ridge regression update:                           │      │
│     │   A[model] += x @ x^T     # Add outer product (d×d)            │      │
│     │   b[model] += reward * x  # Add weighted features (d×1)        │      │
│     │                                                                 │      │
│     │ Sherman-Morrison formula (O(d²) instead of O(d³)):              │      │
│     │   A_inv_new = A_inv - (A_inv @ x @ x^T @ A_inv) /              │      │
│     │                       (1 + x^T @ A_inv @ x)                     │      │
│     │                                                                 │      │
│     │ This keeps A_inv updated without expensive inversion!          │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                             │                                                │
│                             ▼                                                │
│  6. IMPROVED ROUTING (next query)                                            │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │ LinUCB now has updated A_inv and b for this model              │      │
│     │                                                                 │      │
│     │ Next select_arm() call computes:                               │      │
│     │   theta_new = A_inv_new @ b_new  # Better estimate!            │      │
│     │   UCB = theta_new^T @ x + alpha * uncertainty                  │      │
│     │                                                                 │      │
│     │ Models with better reward history (higher theta) get           │      │
│     │ selected more often → LEARNING COMPLETE                        │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Details: LinUCB Learning

**Ridge Regression Formulation:**

For each model arm, LinUCB maintains:
- **A**: d×d matrix (sum of outer products: `Σ x_i @ x_i^T`)
- **b**: d×1 vector (sum of reward-weighted features: `Σ r_i * x_i`)
- **A_inv**: d×d inverse matrix (computed via Sherman-Morrison)

Where d = feature dimension (provider-dependent):
- **Without PCA**: 386 dims (384 embedding + 2 metadata) for FastEmbed, 1538 dims for OpenAI
- **With PCA**: 66 dims (64 PCA + 2 metadata) for FastEmbed, 130 dims (128 PCA + 2 metadata) for OpenAI

**Coefficients (theta):**
```
theta = A_inv @ b
```
This represents the learned reward prediction for this model.

**Upper Confidence Bound (UCB):**
```
UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)
       ︸───────────︸   ︸──────────────────────────︸
       exploitation          exploration
       (learned reward)      (uncertainty bonus)
```

**Update Rule (after receiving feedback):**
```python
# Incremental update (O(d²) using Sherman-Morrison)
A += x @ x^T              # Add information from this query
b += reward * x           # Add reward signal

# Sherman-Morrison: Update A_inv without full inversion
a_inv_x = A_inv @ x
denominator = 1.0 + (x^T @ a_inv_x)
A_inv -= (a_inv_x @ a_inv_x^T) / denominator
```

**Why Sherman-Morrison Matters:**

FastEmbed (384d embeddings, default):
- Full matrix inversion: O(d³) = O(386³) ≈ 57M operations
- Sherman-Morrison update: O(d²) = O(386²) ≈ 149K operations
- **~380x speedup** for each update!

OpenAI (1536d embeddings):
- Full matrix inversion: O(d³) = O(1538³) ≈ 3.6B operations
- Sherman-Morrison update: O(d²) = O(1538²) ≈ 2.4M operations
- **~1500x speedup** for each update!

With PCA (64 dims for FastEmbed, 128 for OpenAI):
- FastEmbed: O(66³) vs O(66²) = 287K vs 4.4K ops (**~65x speedup**)
- OpenAI: O(130³) vs O(130²) = 2.2M vs 17K ops (**~130x speedup**)

### Learning Convergence Example

From integration test `test_feedback_loop_improves_routing`:

```
Training Phase (40 queries with immediate feedback):
┌──────────────┬──────────┬──────────┬────────────────┐
│ Model        │ Cost     │ Quality  │ Reward         │
├──────────────┼──────────┼──────────┼────────────────┤
│ gpt-4o-mini  │ $0.001   │ 0.95     │ 0.932 (HIGH)   │
│ gpt-4o       │ $0.10    │ 0.95     │ 0.903 (LOWER)  │
└──────────────┴──────────┴──────────┴────────────────┘

After Learning (50 test queries):
┌──────────────┬────────────────────┬─────────────────┐
│ Model        │ Selection Rate     │ Learning Status │
├──────────────┼────────────────────┼─────────────────┤
│ gpt-4o-mini  │ >70% (verified ✅) │ Learned cheaper │
│ gpt-4o       │ <30%               │ Deprioritized   │
└──────────────┴────────────────────┴─────────────────┘

Result: System learned to prefer cheaper model with same quality
```

### Exploration vs Exploitation (alpha parameter)

**Default alpha = 1.0** (balanced exploration):
- Good for: Production environments, diverse queries
- Behavior: Continues exploring even after learning
- Trade-off: Slower convergence, better long-term adaptation

**Low alpha = 0.1** (exploitation-focused):
- Good for: Testing, stable workloads
- Behavior: Quickly exploit learned knowledge
- Trade-off: Faster convergence, less exploration of alternatives

**Formula impact:**
```
UCB = mean_reward + alpha * uncertainty

alpha = 1.0:  UCB = 0.9 + 1.0 * 0.2 = 1.1  (explore more)
alpha = 0.1:  UCB = 0.9 + 0.1 * 0.2 = 0.92 (exploit learned reward)
```

### Hybrid Routing (Optional/Legacy)

> **Note**: As of PR #169, the default algorithm is pure Thompson Sampling. Hybrid routing is available but no longer the default.

**When to use hybrid routing:**
- When you want contextual routing (different models for different query types) after a warm-up period
- Use `algorithm="hybrid_thompson_linucb"` to enable

**How hybrid works:**
```
Phase 1 (0-2000 queries): Thompson Sampling
  - No embedding computation
  - Fast Bayesian exploration
  - Builds quality priors for each model

Phase 2 (2000+ queries): LinUCB
  - Uses query embeddings for context-aware routing
  - Knowledge transferred from Thompson Sampling priors
```

**Default (Thompson Sampling only):**
```python
router = Router()  # Pure Thompson Sampling, no phase transition
```

**Hybrid (optional):**
```python
router = Router(algorithm="hybrid_thompson_linucb")  # Phase transition at 2000 queries
```

See `docs/HYBRID_ROUTING_ALGORITHMS.md` for detailed hybrid routing documentation.

### Code Paths (for debugging)

**Route → Update → Learn:**
```python
# 1. Route query
decision = await router.route(query)
# → Router.route() [conduit/engines/router.py:392]
#   → HybridRouter.route() [conduit/engines/hybrid_router.py:373]
#     → LinUCB.select_arm(features) [conduit/engines/bandits/linucb.py:146]
#       → Compute UCB scores using A_inv and b
#       → Return arm with highest UCB

# 2. Execute LLM call
response = await execute_llm(decision.selected_model, query)

# 3. Provide feedback
await router.update(
    model_id=decision.selected_model,
    cost=response.cost,
    quality_score=0.95,
    latency=response.latency,
    features=decision.features  # CRITICAL: pass real features
)
# → Router.update() [conduit/engines/router.py:517]
#   → HybridRouter.update() [conduit/engines/hybrid_router.py:510]
#     → LinUCB.update(feedback, features) [conduit/engines/bandits/linucb.py:195]
#       → Calculate composite reward
#       → Update A += x @ x^T
#       → Update b += reward * x
#       → Sherman-Morrison update A_inv
```

**Key files for feedback loop:**
- `conduit/engines/router.py:392-596` - Router.route() and Router.update()
- `conduit/engines/hybrid_router.py:373-510` - Phase management and delegation
- `conduit/engines/bandits/linucb.py:146-195` - LinUCB selection (UCB formula)
- `conduit/engines/bandits/linucb.py:195-250` - LinUCB update (Sherman-Morrison)
- `conduit/core/reward_calculation.py` - Composite reward calculation

**Integration tests:**
- `tests/integration/test_feedback_loop.py` - Proves feedback loop closure works
  - `test_feedback_loop_improves_routing` - Cost-based learning
  - `test_hybrid_routing_feedback_loop` - Learning across phase transition

---

## Overview

This document provides detailed technical architecture for Conduit's ML-powered LLM routing system.

## System Context

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Client    │────────▶│   Conduit   │────────▶│ LLM Provider│
│ Application │         │   Router    │         │ (OpenAI/etc)│
└─────────────┘         └─────────────┘         └─────────────┘
                              │
                              ├─────────▶ PostgreSQL
                              └─────────▶ Redis Cache
```

---

## Component Architecture

### Layer 1: API Layer (FastAPI)

**Responsibilities**:
- HTTP request/response handling
- Input validation (Pydantic)
- Rate limiting
- Authentication
- Response serialization

**Endpoints**:
- `POST /v1/complete` - Route and execute LLM query
- `POST /v1/feedback` - Submit quality feedback
- `GET /v1/stats` - Analytics and metrics
- `POST /v1/experiment` - A/B testing setup
- `GET /v1/models` - List available models
- `GET /health/live` - Liveness probe (Kubernetes-compatible)
- `GET /health/ready` - Readiness probe (Kubernetes-compatible)
- `GET /health/startup` - Startup probe (Kubernetes-compatible)

**Key Files**:
```
conduit/api/
├── routes.py         # FastAPI route handlers
├── middleware.py     # Auth, rate limiting, CORS
└── validation.py     # Request/response schemas
```

---

### Layer 2: Routing Engine

**Responsibilities**:
- Query analysis and feature extraction
- ML model selection (Thompson Sampling)
- Constraint satisfaction
- Routing decision logging

**Components**:

**QueryAnalyzer** (`conduit/engines/analyzer.py`):
```python
class QueryAnalyzer:
    """Extract features from query for routing decision."""

    def __init__(
        self,
        embedding_provider_type: str = "huggingface",
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
    ):
        # Uses configurable embedding provider (HuggingFace API default)
        self.embedding_provider = create_embedding_provider(
            provider=embedding_provider_type,
            model=embedding_model,
            api_key=embedding_api_key,
        )

    async def analyze(self, query: str) -> QueryFeatures:
        """
        Extract features from query.

        Returns:
            QueryFeatures with embedding, complexity
        """
        # Generate embedding using provider (HuggingFace/OpenAI/Cohere/sentence-transformers)
        embedding = await self.embedding_provider.embed(query)
        complexity = self.compute_complexity(query)

        return QueryFeatures(
            embedding=embedding,
            token_count=len(query.split()),
            complexity_score=complexity
        )
```

**Embedding Providers** (`conduit/engines/embeddings/`):
- **HuggingFace API** (default): Free, no API key required
- **OpenAI**: High-quality embeddings, requires API key
- **Cohere**: Optimized for semantic search, requires API key
- **sentence-transformers**: Local embeddings, optional dependency

See `docs/EMBEDDING_PROVIDERS.md` for detailed provider documentation.

**RoutingEngine** (`conduit/engines/router.py`):
```python
class RoutingEngine:
    """ML-powered model selection engine."""

    def __init__(self, bandit: ContextualBandit, models: list[str]):
        self.bandit = bandit
        self.models = models

    async def select_model(
        self,
        features: QueryFeatures,
        constraints: QueryConstraints | None = None
    ) -> RoutingDecision:
        """
        Select optimal model using Thompson Sampling with fallback strategy.

        Args:
            features: Extracted query features
            constraints: Optional cost/latency/quality constraints

        Returns:
            RoutingDecision with selected model and metadata

        Fallback Strategy:
            1. Filter models by constraints
            2. If no eligible models: relax constraints by 20% and retry
            3. If still none: use default model (gpt-4o-mini)
            4. Thompson Sampling on eligible models
            5. Check circuit breaker for selected model
            6. If circuit OPEN: exclude and reselect (max 2 retries)
            7. Return selection with metadata flags
        """
        # Filter models by constraints
        eligible_models = self._filter_by_constraints(
            self.models, constraints
        )
        constraints_relaxed = False

        # Fallback 1: Relax constraints if no eligible models
        if not eligible_models and constraints:
            logger.warning(f"No models satisfy constraints, relaxing by 20%")
            relaxed_constraints = self._relax_constraints(constraints, factor=0.2)
            eligible_models = self._filter_by_constraints(
                self.models, relaxed_constraints
            )
            constraints_relaxed = True

        # Fallback 2: Use default model if still no matches
        if not eligible_models:
            logger.error(f"No models after constraint relaxation, using default")
            return RoutingDecision(
                selected_model="gpt-4o-mini",
                confidence=0.0,
                features=features,
                reasoning="Default fallback - no models satisfied constraints",
                metadata={"constraints_relaxed": True, "fallback": "default"}
            )

        # Thompson Sampling selection with circuit breaker retry
        max_retries = 2
        for attempt in range(max_retries + 1):
            selected_model = self.bandit.select_model(
                features=features,
                models=eligible_models
            )

            # Check circuit breaker
            if not self._is_circuit_open(selected_model):
                return RoutingDecision(
                    selected_model=selected_model,
                    confidence=self.bandit.get_confidence(selected_model),
                    features=features,
                    reasoning=self._explain_selection(selected_model, features),
                    metadata={"constraints_relaxed": constraints_relaxed, "attempt": attempt}
                )

            # Circuit open, exclude and retry
            logger.warning(f"Circuit breaker OPEN for {selected_model}, retrying")
            eligible_models = [m for m in eligible_models if m != selected_model]
            if not eligible_models:
                break

        # Fallback 3: All models circuit broken or exhausted retries
        logger.error(f"All models failed circuit breaker checks, using default")
        return RoutingDecision(
            selected_model="gpt-4o-mini",
            confidence=0.0,
            features=features,
            reasoning="Default fallback - circuit breakers open",
            metadata={"fallback": "circuit_breaker"}
        )
```

**Key Files**:
```
conduit/engines/
├── analyzer.py       # QueryAnalyzer - feature extraction
├── router.py         # RoutingEngine - model selection
└── bandit.py         # ContextualBandit - Thompson Sampling
```

---

### Layer 3: ML Core (Contextual Bandit)

**Responsibilities**:
- Thompson Sampling implementation
- Beta distribution management
- Reward computation
- Online learning updates

**ContextualBandit** (`conduit/engines/bandit.py`):
```python
class ContextualBandit:
    """Thompson Sampling for model selection."""

    def __init__(self, models: list[str], db: Database):
        self.models = models
        self.db = db
        self.model_states = self._load_states()

    def select_model(
        self,
        features: QueryFeatures,
        models: list[str]
    ) -> str:
        """
        Thompson Sampling: sample from Beta distributions.

        Args:
            features: Query features for reward prediction
            models: Eligible models (after constraint filtering)

        Returns:
            Selected model ID
        """
        samples = {}
        for model in models:
            state = self.model_states[model]

            # Sample from Beta(α, β)
            theta = np.random.beta(state.alpha, state.beta)

            # Weight by predicted reward
            predicted_reward = self._predict_reward(model, features)
            samples[model] = theta * predicted_reward

        return max(samples, key=samples.get)

    def update(
        self,
        model: str,
        reward: float,
        query_id: str,
        success_threshold: float = 0.7
    ):
        """
        Update model's Beta distribution using Thompson Sampling.

        Args:
            model: Model that was used
            reward: Computed reward (0.0-1.0)
            query_id: For audit trail
            success_threshold: Reward threshold to count as success (default 0.7)

        Note:
            Thompson Sampling requires binary success/failure updates.
            We convert continuous reward to binary outcome using threshold.
        """
        state = self.model_states[model]

        # Bayesian update with binary success/failure
        if reward >= success_threshold:
            state.alpha += 1.0  # Count success
        else:
            state.beta += 1.0  # Count failure

        state.total_requests += 1

        # Persist to database
        self.db.update_model_state(state)

        logger.info(
            f"Updated {model}: α={state.alpha:.2f}, β={state.beta:.2f}, "
            f"reward={reward:.2f}, success={reward >= success_threshold}, query={query_id}"
        )

    def _predict_reward(
        self,
        model: str,
        features: QueryFeatures
    ) -> float:
        """
        Predict expected reward for model on this query.

        Phase 1: Simple heuristic based on complexity.
        Phase 2+: Neural network or linear model.
        """
        # Model cost tiers (approximate)
        cost_tier = {
            "gpt-4o-mini": 0.1,
            "gpt-4o": 0.5,
            "claude-opus-4": 1.0,
            "claude-sonnet-4": 0.3,
        }.get(model, 0.5)

        # Simple matching: cheap models for simple queries
        if features.complexity_score < 0.3:
            return 1.0 - cost_tier  # Prefer cheap
        elif features.complexity_score < 0.7:
            return 1.0 - abs(0.5 - cost_tier)  # Prefer mid-tier
        else:
            return cost_tier  # Prefer expensive (high quality)
```

**Key Files**:
```
conduit/engines/
└── bandit.py         # ContextualBandit - Thompson Sampling logic
```

---

### Layer 4: Model Execution (PydanticAI)

**Responsibilities**:
- LLM API calls via unified interface
- Automatic retry logic
- Interaction tracking
- Structured output parsing

**ModelExecutor** (`conduit/engines/executor.py`):
```python
class ModelExecutor:
    """Execute LLM calls via PydanticAI."""

    def __init__(self):
        self.clients = {}  # Cached PydanticAI agents

    async def execute(
        self,
        model: str,
        prompt: str,
        result_type: Type[BaseModel],
        query_id: str,
        timeout: float = 60.0
    ) -> Response:
        """
        Execute LLM call with selected model and timeout.

        Args:
            model: Model ID (e.g., "gpt-4o-mini")
            prompt: User query
            result_type: Pydantic model for structured output
            query_id: For tracking
            timeout: Maximum execution time in seconds (default 60s)

        Returns:
            Response with result and metadata

        Raises:
            ExecutionError: If model call fails or times out
            asyncio.TimeoutError: If execution exceeds timeout

        Timeout Strategy:
            - Default: 60s for all models
            - Fast models (mini): 30s recommended
            - Premium models (opus): 90s recommended
            - Configurable per-query via constraints.max_latency
        """
        start_time = time.time()

        # Get or create PydanticAI agent
        agent = self._get_agent(model, result_type)

        # Execute with timeout and automatic retry
        try:
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout
            )
            latency = time.time() - start_time

            # Extract cost from interaction
            cost = self._compute_cost(result.usage(), model)

            return Response(
                id=generate_id(),
                query_id=query_id,
                model=model,
                text=result.data.model_dump_json(),
                cost=cost,
                latency=latency,
                tokens=result.usage().total_tokens
            )

        except asyncio.TimeoutError:
            latency = time.time() - start_time
            logger.error(f"Execution timeout for {model} after {latency:.2f}s (limit: {timeout}s)")
            raise ExecutionError(
                f"Model {model} exceeded timeout of {timeout}s",
                details={"latency": latency, "timeout": timeout}
            )
        except Exception as e:
            logger.error(f"Execution failed for {model}: {e}")
            raise ExecutionError(f"Model {model} failed: {e}")

    def _get_agent(
        self,
        model: str,
        result_type: Type[BaseModel]
    ) -> Agent:
        """Get cached or create new PydanticAI agent."""
        cache_key = f"{model}_{result_type.__name__}"

        if cache_key not in self.clients:
            self.clients[cache_key] = Agent(
                model=model,
                result_type=result_type
            )

        return self.clients[cache_key]
```

**Key Files**:
```
conduit/engines/
└── executor.py       # ModelExecutor - PydanticAI interface
```

---

### Layer 5: Feedback System

**Responsibilities**:
- Collect explicit feedback (user ratings)
- Compute implicit feedback (quality scores)
- Calculate reward signal
- Trigger model updates

**FeedbackProcessor** (`conduit/engines/feedback.py`):
```python
class FeedbackProcessor:
    """Process feedback and update routing model."""

    def __init__(self, bandit: ContextualBandit, db: Database):
        self.bandit = bandit
        self.db = db

    async def process_feedback(
        self,
        response_id: str,
        feedback: Feedback
    ):
        """
        Process user feedback and update model.

        Args:
            response_id: Response being rated
            feedback: User feedback data
        """
        # Get response metadata
        response = await self.db.get_response(response_id)
        routing = await self.db.get_routing_decision(response.query_id)

        # Compute reward signal
        reward = self._compute_reward(
            feedback=feedback,
            response=response
        )

        # Update model's Beta distribution
        self.bandit.update(
            model=routing.selected_model,
            reward=reward,
            query_id=response.query_id
        )

        # Persist feedback
        await self.db.save_feedback(feedback)

        logger.info(
            f"Processed feedback for {response.model}: "
            f"reward={reward:.2f}, quality={feedback.quality_score:.2f}"
        )

    def _compute_reward(
        self,
        feedback: Feedback,
        response: Response
    ) -> float:
        """
        Compute reward signal from feedback and response.

        Weighted combination:
        - 50% quality score
        - 30% cost efficiency
        - 20% latency performance
        """
        quality_component = feedback.quality_score

        # Normalize cost (assume $0.01 target)
        cost_component = 1.0 - min(response.cost / 0.01, 1.0)

        # Normalize latency (assume 3s target)
        latency_component = 1.0 - min(response.latency / 3.0, 1.0)

        reward = (
            0.5 * quality_component +
            0.3 * cost_component +
            0.2 * latency_component
        )

        return max(0.0, min(1.0, reward))
```

**Key Files**:
```
conduit/engines/
└── feedback.py       # FeedbackProcessor - reward computation
```

---

### Layer 6: Data Persistence

**Responsibilities**:
- Query/response/feedback storage
- Model state persistence
- Analytics aggregation
- Connection pooling

**Database** (`conduit/core/database.py`):
```python
class Database:
    """PostgreSQL interface with transaction management.

    Transaction Boundaries:
        - Single row inserts: Auto-commit (no explicit transaction)
        - Feedback loop updates: Transaction (query → routing → response → feedback)
        - Batch operations: Transaction for consistency
        - Model state updates: Auto-commit (last-write-wins acceptable)
        - Circuit breaker state: Auto-commit (eventual consistency)

    Isolation Level: READ COMMITTED (PostgreSQL default)
    Retry Strategy: Exponential backoff on deadlock/serialization failure
    """

    def __init__(self, connection_string: str):
        self.pool = asyncpg.create_pool(connection_string, max_size=20)

    async def save_query(self, query: Query) -> str:
        """Save query and return ID.

        Transaction: None (single INSERT, auto-commit)
        """
        async with self.pool.acquire() as conn:
            query_id = await conn.fetchval(
                """
                INSERT INTO queries (id, text, user_id, context, constraints, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                query.id, query.text, query.user_id,
                json.dumps(query.context), json.dumps(query.constraints),
                query.created_at
            )
        return query_id

    async def save_complete_interaction(
        self,
        routing: RoutingDecision,
        response: Response,
        feedback: Feedback | None = None
    ):
        """Save routing decision, response, and optional feedback atomically.

        Transaction: REQUIRED (ensures consistency of related records)
        Rollback: On any failure, all records rolled back
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Save routing decision
                await conn.execute(
                    """
                    INSERT INTO routing_decisions (id, query_id, selected_model, confidence, features, reasoning, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    routing.id, routing.query_id, routing.selected_model,
                    routing.confidence, json.dumps(routing.features.model_dump()),
                    routing.reasoning, routing.created_at
                )

                # Save response
                await conn.execute(
                    """
                    INSERT INTO responses (id, query_id, model, text, cost, latency, tokens, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    response.id, response.query_id, response.model, response.text,
                    response.cost, response.latency, response.tokens, response.created_at
                )

                # Save feedback if provided
                if feedback:
                    await conn.execute(
                        """
                        INSERT INTO feedback (id, response_id, quality_score, user_rating, met_expectations, comments, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        feedback.id, feedback.response_id, feedback.quality_score,
                        feedback.user_rating, feedback.met_expectations,
                        feedback.comments, feedback.created_at
                    )

    async def update_model_state(self, state: ModelState):
        """Update model's Beta parameters.

        Transaction: None (UPSERT with ON CONFLICT, auto-commit)
        Concurrency: Last-write-wins acceptable for ML updates
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_states (model_id, alpha, beta, total_requests, total_cost, avg_quality, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (model_id) DO UPDATE
                SET alpha = $2, beta = $3, total_requests = $4,
                    total_cost = $5, avg_quality = $6, updated_at = $7
                """,
                state.model_id, state.alpha, state.beta,
                state.total_requests, state.total_cost,
                state.avg_quality, state.updated_at
            )

    async def get_model_states(self) -> dict[str, ModelState]:
        """Load all model states.

        Transaction: None (single SELECT, read-only)
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM model_states")

        return {
            row["model_id"]: ModelState(
                model_id=row["model_id"],
                alpha=row["alpha"],
                beta=row["beta"],
                total_requests=row["total_requests"],
                total_cost=row["total_cost"],
                avg_quality=row["avg_quality"],
                updated_at=row["updated_at"]
            )
            for row in rows
        }
```

**Key Files**:
```
conduit/core/
└── database.py       # Database - PostgreSQL interface
```

---

## Data Flow

### Complete Request Flow

```
1. Client Request
   POST /v1/complete
   {
     "prompt": "What is photosynthesis?",
     "result_type": "AnalysisResult",
     "constraints": {"max_cost": 0.01}
   }

2. API Layer (FastAPI)
   - Validate request
   - Check rate limits
   - Extract user context

3. Query Analysis
   QueryAnalyzer.analyze()
   - Compute embedding (384-dim)
   - Calculate complexity score

4. Routing Decision
   RoutingEngine.select_model()
   - Filter models by constraints
   - Thompson Sampling → "gpt-4o-mini"
   - Log decision to DB

5. Model Execution
   ModelExecutor.execute()
   - Call PydanticAI agent
   - Parse structured response
   - Track cost/latency

6. Response
   {
     "id": "resp_abc123",
     "model": "gpt-4o-mini",
     "data": {...},
     "metadata": {
       "cost": 0.0002,
       "latency": 1.2,
       "routing_confidence": 0.89
     }
   }

7. Feedback (async)
   Client submits rating → FeedbackProcessor
   - Compute reward signal
   - Update model's Beta(α, β)
   - Persist to DB
```

---

## Configuration Management

**Config** (`conduit/core/config.py`):
```python
class Settings(BaseModel):
    """Application configuration from environment."""

    # Database
    database_url: str
    database_pool_size: int = 20

    # Redis
    redis_url: str
    redis_ttl: int = 3600  # 1 hour

    # LLM Providers
    openai_api_key: str
    anthropic_api_key: str
    google_api_key: str

    # ML Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    default_models: list[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-sonnet-4",
        "claude-opus-4"
    ]

    # Routing
    exploration_rate: float = 0.1
    reward_weights: dict[str, float] = {
        "quality": 0.5,
        "cost": 0.3,
        "latency": 0.2
    }

    # API
    rate_limit: int = 100  # requests/minute
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

---

## Error Handling

### Exception Hierarchy

```python
class ConduitError(Exception):
    """Base exception for all Conduit errors."""

class AnalysisError(ConduitError):
    """Query analysis failed."""

class RoutingError(ConduitError):
    """Model selection failed."""

class ExecutionError(ConduitError):
    """LLM execution failed."""

class DatabaseError(ConduitError):
    """Database operation failed."""

class ValidationError(ConduitError):
    """Input validation failed."""
```

### Retry Strategy

**Circuit Breaker** (`conduit/resilience/circuit_breaker.py`):
```python
class CircuitBreaker:
    """Prevent cascading failures for LLM calls."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Reset on successful call."""
        self.failures = 0
        if self.state == "half_open":
            self.state = "closed"

    def on_failure(self):
        """Track failure and open if threshold reached."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
```

---

## Monitoring and Observability

### Metrics

**Key Metrics to Track**:
- Routing latency (p50, p95, p99)
- Model selection distribution
- Cost per query (by model)
- Quality scores (by model)
- Error rates (by type)
- Beta parameter evolution (α, β trends)

**Logging** (`conduit/core/logging.py`):
```python
import structlog

logger = structlog.get_logger()

# Usage in code
logger.info(
    "routing_decision",
    query_id=query_id,
    model=selected_model,
    confidence=confidence,
    features=features.dict()
)
```

---

## Security Considerations

### API Key Management

- Store in environment variables or secure secret manager
- Load from environment variables only
- Never log API keys
- Rotate regularly

### Rate Limiting

- 100 requests/minute per user (configurable)
- Redis-backed rate limiter
- Exponential backoff on violations

### Data Privacy

- No PII logged without consent
- User data encrypted in transit (HTTPS)
- Database encryption at rest (provider-specific)
- GDPR-compliant data retention

---

**Last Updated**: 2025-11-27
**Status**: Phase 3 complete, Thompson Sampling default (PR #169)
