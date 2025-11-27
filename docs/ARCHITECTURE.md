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
│   │                         HYBRID ROUTER                                    │   │
│   │                                                                          │   │
│   │   Phase 1 (0-2000 queries)        Phase 2 (2000+ queries)               │   │
│   │   ┌─────────────────┐             ┌─────────────────┐                   │   │
│   │   │      UCB1       │──transition─▶│     LinUCB      │                   │   │
│   │   │  (no features)  │             │  (uses features) │                   │   │
│   │   │  Fast explore   │             │  Smart routing   │                   │   │
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
│                                         │ 0.5q+0.3c+  │                         │
│                                         │   0.2l      │                         │
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
| **UCB1 Bandit** | Cold start exploration | None (non-contextual) | Model selection |
| **LinUCB Bandit** | Contextual routing | QueryFeatures | Model selection |
| **Hybrid Router** | Phase management | Query | RoutingDecision |
| **Arbiter** | Quality evaluation | Query + Response | Quality score (0-1) |
| **Feedback Loop** | Learning signal | Feedback sources | Reward → Bandit update |

### When Each Component Is Used

```
Query Count:    0        500      1000      1500      2000      5000
                │         │         │         │         │         │
Embedding:      ❌ ─────────────────────────────╳ ✅ ──────────────────▶
                Not needed (UCB1)               Required (LinUCB)

PCA:            ❌ ─────────────────────────────╳ Optional ────────────▶
                Not needed (UCB1)               Reduces 386→66 dims

UCB1:           ✅ ════════════════════════════╗
                Active, exploring              ║ Transition
                                               ╚═══════════════════════▶ ❌

LinUCB:         ❌                              ╔═══════════════════════▶ ✅
                Waiting                        Active, contextual

Arbiter:        ✅ ─────────────────────────────────────────────────────▶
                Always available (async, sampled at 10%)
```

### Arbiter vs Bandit (Common Confusion)

| Aspect | Bandit (UCB1/LinUCB) | Arbiter |
|--------|---------------------|---------|
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

**Last Updated**: 2025-11-25
**Status**: Phase 3 complete, performance optimizations shipped
