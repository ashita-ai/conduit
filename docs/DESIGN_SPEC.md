# Conduit - Design Specification

**Version**: 0.1.0
**Date**: 2025-11-18
**Status**: Initial Design

---

## Executive Summary

Conduit is an ML-powered LLM routing system that learns optimal model selection based on cost, latency, and quality trade-offs. Unlike static rule-based routers (LiteLLM, PortKey), Conduit uses contextual bandits with Thompson Sampling to continuously improve routing decisions through feedback loops.

**Core Value Proposition**: 30-80% cost reduction through intelligent, adaptive routing that learns from your specific workload patterns.

---

## Problem Statement

### Current State

**Enterprises waste 30-80% of LLM costs** due to:
- Static routing rules that don't adapt to workload patterns
- Manual tuning requiring constant maintenance
- No learning from actual usage and quality feedback
- One-size-fits-all model selection ignoring query complexity

**Example Waste**:
- FAQ queries ("What's your return policy?") routed to GPT-4 instead of GPT-4o-mini
- Simple code tasks using expensive models
- Legal queries going to wrong model for domain

### Target Users

**Primary**: Engineering teams at companies with >$100K/month LLM spend
**Secondary**: Mid-market companies ($10K-100K/month spend) seeking cost optimization

### Success Metrics

- **Cost Reduction**: 40%+ average vs baseline (static routing)
- **Quality Maintenance**: >95% user satisfaction vs always using premium models
- **Latency Overhead**: <100ms routing decision time
- **Accuracy**: 80%+ correct model selection for query type

---

## Solution Overview

### Core Innovation

**ML-Driven Routing**: Contextual bandit algorithm learns optimal model selection from usage patterns and feedback, adapting to your specific workload characteristics.

### Key Capabilities

**1. Intelligent Query Analysis**
- Semantic embedding (sentence-transformers)
- Complexity scoring (length, structure)
- Domain classification (code, legal, medical, general)
- Historical pattern matching

**2. Adaptive Model Selection**
- Thompson Sampling for exploration/exploitation balance
- Multi-objective optimization (cost/latency/quality)
- Constraint satisfaction (max cost, min quality thresholds)
- Continuous learning from feedback

**3. Provider-Agnostic Execution**
- Unified interface via PydanticAI
- Support for OpenAI, Anthropic, Google, Groq
- Automatic retry and error handling
- Interaction tracking (cost, latency, tokens)

**4. Feedback-Driven Improvement**
- Explicit user ratings (1-5 stars, thumbs up/down)
- Implicit signals (latency, token efficiency)
- Quality scoring (semantic similarity to references)
- Model parameter updates (online learning)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conduit Router                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Query Analysis Layer                                 │ │
│  │                                                        │ │
│  │   - Embedding: sentence-transformers/all-MiniLM-L6-v2 │ │
│  │   - Complexity: token count, structure analysis       │ │
│  │   - Domain: keyword + embedding-based classification  │ │
│  │   - Features: [embedding(384), length, domain_id]     │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Routing Intelligence (ML Core)                       │ │
│  │                                                        │ │
│  │   Algorithm: Contextual Bandit (Thompson Sampling)    │ │
│  │   State: Beta distributions per model (α, β)          │ │
│  │   Features: query_embedding, length, domain, history  │ │
│  │   Reward: f(cost, quality, latency)                   │ │
│  │   Update: Online learning after each interaction      │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Model Execution Layer (PydanticAI)                   │ │
│  │                                                        │ │
│  │   - Provider abstraction (OpenAI, Anthropic, Google)  │ │
│  │   - Structured outputs (Pydantic models)              │ │
│  │   - Automatic retry logic                             │ │
│  │   - Interaction tracking (cost, tokens, latency)      │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Feedback & Learning Loop                             │ │
│  │                                                        │ │
│  │   - User ratings collection (explicit)                │ │
│  │   - Quality scoring (semantic similarity)             │ │
│  │   - Model updates (Thompson Sampling β updates)       │ │
│  │   - History persistence (PostgreSQL)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
           ↓              ↓              ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Analytics │   │   API    │   │Dashboard │
    │ Database │   │ Service  │   │   (Web)  │
    └──────────┘   └──────────┘   └──────────┘
```

### Component Responsibilities

**Query Analysis Layer**:
- Extract semantic features via embedding model
- Compute complexity metrics (token count, nesting depth)
- Classify domain (code, legal, medical, general)
- Output: Feature vector for ML model

**Routing Intelligence**:
- Maintain Beta distributions for each model's performance
- Sample from distributions (Thompson Sampling)
- Select model balancing exploration/exploitation
- Apply constraints (max cost, min quality, max latency)

**Model Execution Layer**:
- Call selected LLM via PydanticAI unified interface
- Handle retries and errors
- Track interaction metadata (cost, tokens, latency)
- Return structured response

**Feedback Loop**:
- Collect explicit feedback (user ratings)
- Compute implicit feedback (quality scores)
- Calculate reward signal
- Update model's Beta distribution parameters
- Persist to database for analytics

---

## Data Models

### Core Data Models

**Query**:
```python
class Query(BaseModel):
    id: str  # UUID
    text: str
    user_id: str | None
    context: dict[str, Any] | None
    constraints: QueryConstraints | None
    created_at: datetime
```

**QueryConstraints**:
```python
class QueryConstraints(BaseModel):
    max_cost: float | None  # Maximum cost in dollars
    max_latency: float | None  # Maximum latency in seconds
    min_quality: float | None  # Minimum quality score (0.0-1.0)
    preferred_provider: str | None  # "openai", "anthropic", etc.
```

**RoutingDecision**:
```python
class RoutingDecision(BaseModel):
    query_id: str
    selected_model: str  # "gpt-4o-mini", "claude-opus-4"
    confidence: float  # Thompson sampling confidence
    features: QueryFeatures
    reasoning: str  # Why this model was chosen
    timestamp: datetime
```

**QueryFeatures**:
```python
class QueryFeatures(BaseModel):
    embedding: list[float]  # 384-dim from sentence-transformers
    token_count: int
    complexity_score: float  # 0.0-1.0
    domain: str  # "code", "legal", "medical", "general"
    domain_confidence: float
```

**Response**:
```python
class Response(BaseModel):
    id: str
    query_id: str
    model: str
    text: str
    cost: float
    latency: float
    tokens: int
    created_at: datetime
```

**Feedback**:
```python
class Feedback(BaseModel):
    response_id: str
    quality_score: float  # 0.0-1.0 (explicit or computed)
    user_rating: int | None  # 1-5 stars
    met_expectations: bool
    comments: str | None
    created_at: datetime
```

### ML Model State

**ModelState** (persisted in database):
```python
class ModelState(BaseModel):
    model_id: str  # "gpt-4o-mini", "claude-opus-4"
    alpha: float  # Beta distribution α parameter (successes + 1)
    beta: float  # Beta distribution β parameter (failures + 1)
    total_requests: int
    total_cost: float
    avg_quality: float
    updated_at: datetime
```

---

## ML Algorithm: Contextual Bandit

### Thompson Sampling Overview

**Problem**: Choose which model to use for each query to maximize reward over time.

**Solution**: Maintain Beta distribution for each model's success rate. Sample from distributions and choose model with highest sample.

### Algorithm Pseudocode

```python
class ContextualBandit:
    def __init__(self, models: list[str]):
        # Initialize Beta(1, 1) for each model (uniform prior)
        self.model_params = {
            model: {"alpha": 1.0, "beta": 1.0}
            for model in models
        }

    def select_model(self, query_features: QueryFeatures) -> str:
        """Thompson Sampling: sample from each model's Beta distribution."""
        samples = {}
        for model, params in self.model_params.items():
            # Sample success probability from Beta(α, β)
            theta = np.random.beta(params["alpha"], params["beta"])
            # Weight by predicted reward for this query
            samples[model] = theta * self.predict_reward(model, query_features)

        return max(samples, key=samples.get)

    def predict_reward(self, model: str, features: QueryFeatures) -> float:
        """Estimate reward for this model on this query."""
        # Simple heuristic (Phase 1): based on query complexity
        if features.complexity_score < 0.3:
            # Simple queries: prefer cheap models
            return 1.0 if "mini" in model else 0.5
        elif features.complexity_score < 0.7:
            # Medium complexity: prefer balanced models
            return 1.0 if "gpt-4o" in model else 0.7
        else:
            # Complex queries: prefer premium models
            return 1.0 if "opus" in model or "gpt-4" in model else 0.3

    def update(self, model: str, reward: float, success_threshold: float = 0.7):
        """Update Beta distribution after observing reward.

        Args:
            model: Model identifier to update
            reward: Observed reward (0.0-1.0)
            success_threshold: Reward threshold to count as success (default 0.7)

        Note:
            Thompson Sampling requires binary success/failure updates.
            We convert continuous reward to binary outcome using threshold.
        """
        params = self.model_params[model]

        if reward >= success_threshold:
            params["alpha"] += 1.0  # Count success
        else:
            params["beta"] += 1.0  # Count failure
```

### Reward Function

**Objective**: Balance cost, quality, and latency.

```python
def compute_reward(
    cost: float,
    quality_score: float,
    latency: float,
    target_cost: float = 0.01,  # $0.01 target per query
    target_latency: float = 3.0  # 3s target
) -> float:
    """
    Compute reward signal for model update.

    Returns: 0.0-1.0 reward (higher is better)
    """
    # Normalize cost (lower is better)
    cost_score = 1.0 - min(cost / target_cost, 1.0)

    # Quality score already 0.0-1.0 (higher is better)
    quality_penalty = quality_score

    # Normalize latency (lower is better)
    latency_score = 1.0 - min(latency / target_latency, 1.0)

    # Weighted combination (tunable)
    reward = (
        0.5 * quality_penalty +  # Quality most important
        0.3 * cost_score +       # Cost second
        0.2 * latency_score      # Latency third
    )

    return max(0.0, min(1.0, reward))
```

---

## Database Schema

### PostgreSQL Tables

**queries**:
```sql
CREATE TABLE queries (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    user_id TEXT,
    context JSONB,
    constraints JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_created_at ON queries(created_at);
```

**routing_decisions**:
```sql
CREATE TABLE routing_decisions (
    id UUID PRIMARY KEY,
    query_id UUID REFERENCES queries(id),
    selected_model TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    features JSONB NOT NULL,
    reasoning TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_routing_decisions_query_id ON routing_decisions(query_id);
CREATE INDEX idx_routing_decisions_model ON routing_decisions(selected_model);
```

**responses**:
```sql
CREATE TABLE responses (
    id UUID PRIMARY KEY,
    query_id UUID REFERENCES queries(id),
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    cost FLOAT NOT NULL,
    latency FLOAT NOT NULL,
    tokens INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_responses_query_id ON responses(query_id);
CREATE INDEX idx_responses_model ON responses(model);
```

**feedback**:
```sql
CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    response_id UUID REFERENCES responses(id),
    quality_score FLOAT NOT NULL,
    user_rating INT,
    met_expectations BOOLEAN NOT NULL,
    comments TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_response_id ON feedback(response_id);
```

**model_states**:
```sql
CREATE TABLE model_states (
    model_id TEXT PRIMARY KEY,
    alpha FLOAT NOT NULL DEFAULT 1.0,
    beta FLOAT NOT NULL DEFAULT 1.0,
    total_requests INT NOT NULL DEFAULT 0,
    total_cost FLOAT NOT NULL DEFAULT 0.0,
    avg_quality FLOAT NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### Transaction Boundaries

**Isolation Level**: READ COMMITTED (PostgreSQL default)

**Transaction Policies**:

1. **Single Row Operations** (Auto-commit):
   - `INSERT INTO queries` - Single query save
   - `SELECT * FROM model_states` - Model state reads
   - No explicit transaction needed

2. **Complete Interaction** (Transactional):
   - Atomic write of: routing_decision + response + feedback (optional)
   - All-or-nothing guarantee for related records
   - Rollback on any failure to maintain referential integrity
   ```sql
   BEGIN;
     INSERT INTO routing_decisions (...);
     INSERT INTO responses (...);
     INSERT INTO feedback (...);  -- Optional
   COMMIT;
   ```

3. **Model State Updates** (Auto-commit):
   - `UPSERT INTO model_states` with ON CONFLICT
   - Last-write-wins acceptable (ML updates are idempotent)
   - No transaction needed (eventual consistency)

4. **Circuit Breaker State** (Auto-commit):
   - In-memory state with periodic database sync
   - Eventual consistency acceptable
   - No strong ACID requirements

**Concurrency Control**:
- Optimistic locking not required (no user-facing conflicts)
- Database-level UNIQUE constraints prevent duplicates
- Retry logic handles transient deadlocks (exponential backoff)

---

## API Design

### REST API Endpoints

**POST /v1/complete**
```json
Request:
{
    "prompt": "What is the capital of France?",
    "result_type": "AnalysisResult",  // Pydantic model name
    "constraints": {
        "max_cost": 0.01,
        "max_latency": 2.0,
        "min_quality": 0.7
    }
}

Response:
{
    "id": "resp_123abc",
    "query_id": "query_456def",
    "model": "gpt-4o-mini",
    "data": {
        "answer": "Paris",
        "confidence": 0.99
    },
    "metadata": {
        "cost": 0.0001,
        "latency": 0.8,
        "tokens": 15,
        "routing_confidence": 0.87
    }
}
```

**POST /v1/feedback**
```json
Request:
{
    "response_id": "resp_123abc",
    "quality_score": 0.95,
    "user_rating": 5,
    "met_expectations": true,
    "comments": "Perfect answer"
}

Response:
{
    "status": "success",
    "model_updated": true
}
```

**GET /v1/stats**
```json
Response:
{
    "total_queries": 10000,
    "total_cost": 125.50,
    "avg_cost_per_query": 0.0125,
    "cost_savings_vs_baseline": 0.68,  // 68% savings
    "model_distribution": {
        "gpt-4o-mini": 0.65,
        "gpt-4o": 0.25,
        "claude-opus-4": 0.10
    },
    "avg_quality_score": 0.87
}
```

**GET /health/live**
```json
Response: 200 OK
{
    "status": "healthy",
    "timestamp": "2025-11-18T12:00:00Z"
}

Response: 503 Service Unavailable
{
    "status": "unhealthy",
    "timestamp": "2025-11-18T12:00:00Z",
    "error": "Database connection failed"
}
```

**GET /health/ready**
```json
Response: 200 OK
{
    "status": "ready",
    "timestamp": "2025-11-18T12:00:00Z",
    "checks": {
        "database": "ok",
        "redis": "ok",
        "model_states_loaded": true,
        "llm_providers": {
            "openai": "ok",
            "anthropic": "ok"
        }
    }
}

Response: 503 Service Unavailable
{
    "status": "not_ready",
    "timestamp": "2025-11-18T12:00:00Z",
    "checks": {
        "database": "ok",
        "redis": "degraded",
        "model_states_loaded": false,
        "llm_providers": {
            "openai": "ok",
            "anthropic": "unavailable"
        }
    }
}
```

**GET /health/startup**
```json
Response: 200 OK
{
    "status": "started",
    "timestamp": "2025-11-18T12:00:00Z",
    "startup_duration_ms": 1234
}

Response: 503 Service Unavailable
{
    "status": "starting",
    "timestamp": "2025-11-18T12:00:00Z",
    "startup_duration_ms": 500,
    "pending": ["loading_model_states", "initializing_redis"]
}
```

### Health Check Semantics

**Liveness Probe** (`/health/live`):
- Purpose: Detect if process is alive
- Checks: Minimal (process responding)
- Failure action: Restart container/process
- Timeout: 1s
- Frequency: Every 10s

**Readiness Probe** (`/health/ready`):
- Purpose: Detect if ready to serve traffic
- Checks: Database connectivity, Redis availability, model states loaded, LLM provider API keys valid
- Failure action: Remove from load balancer
- Timeout: 5s
- Frequency: Every 5s

**Startup Probe** (`/health/startup`):
- Purpose: Detect when app has fully initialized
- Checks: All initialization tasks complete
- Failure action: Mark deployment failed
- Timeout: 30s
- Frequency: Every 5s during startup

---

## Implementation Phases

### Phase 0: Foundation (Weeks 1-4)

**Deliverables**:
- Core data models (Pydantic)
- Database schema and migrations (Supabase)
- Configuration management
- Exception hierarchy

### Phase 1: Rule-Based Router (Weeks 5-6)

**Deliverables**:
- Query analysis (length, basic complexity)
- Static routing rules (baseline)
- PydanticAI integration
- Basic cost tracking

### Phase 2: ML Integration (Weeks 7-10)

**Deliverables**:
- Sentence-transformers embedding
- Thompson Sampling implementation
- Feedback collection API
- Online learning loop

### Phase 3: Production Features (Weeks 11-14)

**Deliverables**:
- FastAPI REST endpoints
- Supabase connection pooling
- Redis caching
- Monitoring and metrics

### Phase 4: Advanced Optimization (Weeks 15-16)

**Deliverables**:
- Multi-objective optimization
- Cost prediction
- A/B testing framework
- Advanced analytics

---

## Non-Functional Requirements

### Performance

- **Routing Latency**: <100ms p95
- **Total Latency Overhead**: <10% of LLM call time
- **Throughput**: 100+ queries/second
- **Database**: <50ms p95 query time

### Timeouts

- **Default LLM Timeout**: 60s (configurable via `LLM_TIMEOUT_DEFAULT`)
- **Fast Models (mini)**: 30s recommended (configurable via `LLM_TIMEOUT_FAST`)
- **Premium Models (opus)**: 90s recommended (configurable via `LLM_TIMEOUT_PREMIUM`)
- **Per-Query Override**: Timeout can be reduced via `QueryConstraints.max_latency`
- **Timeout Behavior**:
  - LLM call wrapped in `asyncio.wait_for()`
  - On timeout: ExecutionError raised with latency details
  - Timeout event logged for monitoring
  - Falls back to default model (see Reliability section)

### Scalability

- **Concurrent Users**: 1000+
- **Queries/Day**: 1M+
- **Model States**: 50+ models supported
- **History Retention**: 90 days

### Reliability

- **Uptime**: 99.9% (3 nines)
- **Error Rate**: <0.1%
- **Data Durability**: Supabase handles backup/recovery

### Fallback Strategies

**Constraint Violation Handling**:
- If no models satisfy QueryConstraints (max_cost, max_latency, min_quality):
  1. Relax constraints by 20% and retry model selection
  2. If still no match: Use cheapest available model (gpt-4o-mini)
  3. Log constraint violation event for monitoring
  4. Return response with metadata flag: `constraints_relaxed: true`

**Model Failure Handling**:
- Circuit breaker pattern (5 failures → OPEN for 60s)
- On model failure or timeout:
  1. Check circuit breaker state for failed model
  2. If OPEN: Exclude from selection pool
  3. Retry with next-best model from Thompson Sampling
  4. Max 2 retries before returning error to user
  5. Log failure chain for debugging

**Default Model Fallback**:
- Ultimate fallback: `gpt-4o-mini` (fast, cheap, reliable)
- Used when:
  - All models fail circuit breaker checks
  - All constraint relaxations exhausted
  - System in degraded mode (database unreachable)
- Default model bypass routing logic (direct execution)

**Graceful Degradation**:
- If database unreachable: Use in-memory model states (ephemeral)
- If embedding service fails: Use rule-based routing (complexity heuristic)
- If all LLM providers fail: Return 503 Service Unavailable with retry-after header

### Security

- **API Keys**: Stored in Supabase secrets (encrypted at rest)
- **User Data**: No PII logged without consent
- **Rate Limiting**: 100 req/min per user
- **HTTPS Only**: All API communication encrypted

---

## Success Criteria

### Technical Success

- 80%+ test coverage
- <100ms routing overhead (p95)
- Zero data loss (Supabase durability)
- Type-safe API (Pydantic validation)

### Business Success

- 40%+ cost reduction vs baseline
- >95% user satisfaction with quality
- 80%+ model selection accuracy
- <10% user churn

### Learning Success

- Model parameters converge within 1000 queries
- Continuous improvement visible in metrics
- Handles concept drift (workload changes)
- Adapts to new models automatically

---

## Risks and Mitigations

### Risk 1: ML Doesn't Beat Rules

**Mitigation**:
- Start with strong rule-based baseline
- Extensive A/B testing before full rollout
- Fall back to rules if ML underperforms
- Hire ML consultant if needed

### Risk 2: Cold Start Problem

**Mitigation**:
- Warm-start with rule-based routing
- Transfer learning from similar workloads
- Offer manual configuration during ramp-up

### Risk 3: Concept Drift

**Mitigation**:
- Detect distribution shifts
- Periodic model retraining
- Decay old interaction weights
- Alert on anomalous patterns

### Risk 4: Latency Overhead

**Mitigation**:
- Cache embeddings for repeated queries
- Pre-compute features where possible
- Use fast embedding models
- Profile and optimize hot paths

---

## Future Enhancements

### Phase 5+

- **Deep Learning**: Replace Thompson Sampling with neural bandit
- **Multi-Armed Context**: Incorporate user history, session context
- **Reinforcement Learning**: Full RL with state transitions
- **Federated Learning**: Learn across multiple organizations
- **Prompt Optimization**: Auto-tune prompts per model
- **Model Fine-Tuning**: Custom models for specific domains

---

**Last Updated**: 2025-11-18
**Next Review**: After Phase 1 completion
