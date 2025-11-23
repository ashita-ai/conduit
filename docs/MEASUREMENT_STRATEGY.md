# Conduit Measurement Strategy

**Purpose**: Systematic quality assessment and performance tracking for ML-powered routing.

**Last Updated**: 2025-11-23

---

## Overview

Conduit uses a **three-tier measurement strategy** to ensure routing quality, track performance, and guide algorithm improvements:

1. **Tier 1: Core Metrics** (Issue #42) - Regret calculation, quality trends, cost efficiency
2. **Tier 2: Automated Evaluation** (Issues #42, #43) - LLM-as-judge, embeddings database
3. **Tier 3: Advanced Analysis** (Issues #44, #45, #46) - Batch pipelines, clustering, dashboards

---

## Tier 1: Core Evaluation Metrics

**Status**: ✅ Implemented (Migration `d330b3ea662d`)

**Database Table**: `evaluation_metrics`

### Key Metrics

#### Regret Metrics
Measure distance from optimal routing:

```python
# regret_vs_oracle: Distance from perfect routing
# 0.0 = Oracle performance (always choose best model in hindsight)
# Lower is better
regret_oracle = oracle_reward - actual_reward

# regret_vs_random: Distance from random routing
# Negative = worse than random (exploration phase)
# Positive = better than random (exploitation phase)
regret_random = actual_reward - random_reward

# regret_vs_always_best: Distance from always using historically best model
# Shows value of contextual routing over static routing
regret_always_best = always_best_reward - actual_reward
```

#### Quality Trend
7-day moving average of quality scores:
```python
quality_trend = {
    "last_hour": 0.92,   # Recent performance
    "last_day": 0.90,    # Daily average
    "last_week": 0.88,   # Weekly trend
}
```

#### Cost Efficiency
Quality achieved per dollar spent:
```python
cost_efficiency = avg_quality_score / avg_cost_per_query
# Higher = better quality/cost ratio
```

#### Convergence Rate
How quickly bandit approaches optimal performance:
```python
convergence_rate = (initial_regret - current_regret) / queries_processed
# Measures learning speed
```

### Computing Metrics

```python
# conduit/evaluation/metrics.py
from conduit.engines.bandits.baselines import OracleBaseline, RandomBaseline

async def compute_regret_metrics(time_window: str = "last_day"):
    """Compute regret vs baselines for time window."""

    # Fetch routing decisions and outcomes
    decisions = await db.fetch_routing_decisions(time_window)

    # Oracle: Perfect hindsight (always best model)
    oracle = OracleBaseline(decisions)
    oracle_reward = oracle.compute_total_reward()

    # Random: Random model selection
    random = RandomBaseline(decisions)
    random_reward = random.compute_total_reward()

    # Actual: What Conduit chose
    actual_reward = sum(d.reward for d in decisions)

    # Compute regret
    regret_oracle = (oracle_reward - actual_reward) / len(decisions)
    regret_random = (actual_reward - random_reward) / len(decisions)

    # Store in evaluation_metrics table
    await db.store_metric(
        metric_name="regret_vs_oracle",
        metric_value=regret_oracle,
        time_window=time_window
    )
    await db.store_metric(
        metric_name="regret_vs_random",
        metric_value=regret_random,
        time_window=time_window
    )
```

### Scheduled Computation

```python
# Run hourly via cron or background task
# 0 * * * * python -m conduit.evaluation.compute_metrics

async def compute_all_metrics():
    """Compute all evaluation metrics for multiple time windows."""
    for window in ["last_hour", "last_day", "last_week"]:
        await compute_regret_metrics(window)
        await compute_quality_trend(window)
        await compute_cost_efficiency(window)
        await compute_convergence_rate(window)
```

---

## Tier 2: Automated Quality Evaluation

### LLM-as-Judge with Arbiter (Issue #42)

**Status**: 📋 Planned

**Framework**: [Arbiter](https://github.com/MisfitIdeas/arbiter) - Production-grade LLM evaluation

#### Why Arbiter?

Arbiter is our LLM evaluation framework with:
- **Multiple Evaluators**: Semantic, CustomCriteria, Factuality, Groundedness, Relevance
- **Automatic Cost Tracking**: Every evaluation logs LLM usage costs
- **PydanticAI Integration**: Same provider abstraction as Conduit
- **Production Quality**: 95% test coverage, strict type safety
- **Same Author**: Maintained alongside Conduit for consistency

#### Integration Pattern

```python
# conduit/evaluation/arbiter_evaluator.py
from arbiter import evaluate, SemanticEvaluator, FactualityEvaluator
from conduit.core.models import Response, Query

async def evaluate_response_quality(
    query: Query,
    response: Response,
    sample_rate: float = 0.1  # Evaluate 10% of responses
) -> float:
    """Evaluate response quality using Arbiter.

    Runs async in background - doesn't block routing.
    """
    if random.random() > sample_rate:
        return None  # Skip this evaluation

    # Evaluate with multiple criteria
    result = await evaluate(
        output=response.text,
        reference=query.text,
        evaluators=[
            SemanticEvaluator(),      # Semantic similarity
            FactualityEvaluator(),    # Factual correctness
        ]
    )

    # Store in feedback table
    await db.store_feedback(
        response_id=response.id,
        quality_score=result.overall_score,
        comments=f"Arbiter eval: {result.interactions[0].cost_usd:.4f} USD"
    )

    return result.overall_score
```

#### Cost Management

```python
# Track evaluation costs separately
evaluation_budget = 10.00  # $10/day evaluation budget
current_spend = await db.sum_evaluation_costs(time_range="today")

if current_spend < evaluation_budget:
    # Continue evaluations
    sample_rate = 0.1  # 10% sampling
else:
    # Budget exhausted, reduce sampling
    sample_rate = 0.01  # 1% sampling
```

### Embeddings Database with pgvector (Issue #43)

**Status**: 📋 Planned

**Technology**: PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)

#### Use Cases

1. **Fast Retry Detection**: Semantic similarity search with pgvector indexing
2. **Query Clustering**: Identify common query patterns
3. **Semantic Routing**: Route similar queries consistently

#### Schema

```sql
-- Add vector extension and embedding column
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE queries
ADD COLUMN embedding vector(384);  -- sentence-transformer dimension

-- Fast similarity search index
CREATE INDEX ON queries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### Similarity Search

```python
# Find similar queries in < 50ms
similar = await db.fetch(
    """
    SELECT q.*, (embedding <=> $1) as distance
    FROM queries q
    WHERE (embedding <=> $1) < $2  -- cosine distance threshold
    ORDER BY distance
    LIMIT 10
    """,
    query_embedding,
    0.2  # similarity threshold
)
```

---

## Tier 3: Advanced Analysis & Orchestration

### Batch Evaluation with Loom (Issue #44)

**Status**: 📋 Planned

**Framework**: [Loom](https://github.com/MisfitIdeas/loom) - AI pipeline orchestration

#### Why Loom?

Loom is our pipeline orchestration framework with:
- **Extract → Transform → Evaluate → Load** pattern
- **Hard Dependency on Arbiter**: Built-in evaluation infrastructure
- **Async/Await**: Matches Conduit's concurrency patterns
- **Experiment Tracking**: Version control for configurations
- **Same Author**: Designed to work with Arbiter and Conduit

#### Integration Pattern

```python
# conduit/evaluation/pipeline.py
from loom import Pipeline
from arbiter import SemanticEvaluator

class ConduitEvaluationPipeline(Pipeline):
    """Batch evaluation for routing strategies."""

    async def extract(self) -> list[Query]:
        """Extract historical queries."""
        return await db.fetch_queries(time_range="last_24h")

    async def transform(self, queries: list[Query]) -> list[Response]:
        """Route queries and collect responses."""
        responses = []
        for query in queries:
            response = await router.route(query)
            responses.append(response)
        return responses

    async def evaluate(self, responses: list[Response]) -> EvaluationResult:
        """Evaluate with Arbiter."""
        results = []
        for response in responses:
            result = await arbiter.evaluate(
                output=response.text,
                evaluators=[SemanticEvaluator()]
            )
            results.append(result)
        return aggregate_results(results)

    async def load(self, results: EvaluationResult) -> None:
        """Store metrics in evaluation_metrics table."""
        await db.store_evaluation_metrics(results)
```

#### A/B Testing

```python
# Compare routing strategies offline
pipeline = ConduitEvaluationPipeline()

results = await pipeline.ab_test(
    control=RouterConfig(algorithm="linucb", pca_dims=67),
    variants=[
        RouterConfig(algorithm="ucb1"),           # Non-contextual
        RouterConfig(algorithm="linucb", pca_dims=128),  # More dims
        RouterConfig(algorithm="thompson"),        # Bayesian
    ],
    queries=historical_queries[-1000:],
    evaluators=[SemanticEvaluator(), FactualityEvaluator()]
)

print(f"Winner: {results.winner}")
print(f"Regret improvement: {results.regret_reduction:.2%}")
print(f"Statistical significance: p={results.p_value:.4f}")
```

### Query Pattern Clustering (Issue #45)

**Status**: 📋 Planned

**Technology**: scikit-learn DBSCAN + pgvector

```python
from sklearn.cluster import DBSCAN

# Cluster queries by embedding similarity
embeddings = await db.fetch_embeddings(time_range="last_week")
clustering = DBSCAN(eps=0.15, min_samples=10, metric='cosine')
labels = clustering.fit_predict(embeddings)

# Analyze cluster performance
for cluster_id in set(labels):
    cluster = await analyze_cluster(cluster_id, labels)
    print(f"Cluster {cluster_id}: {cluster.size} queries")
    print(f"  Best model: {cluster.best_model}")
    print(f"  Avg quality: {cluster.avg_quality:.2f}")
    print(f"  Regret: {cluster.regret_vs_oracle:.3f}")
```

### Real-Time Dashboard (Issue #46)

**Status**: 📋 Planned

**Technology**: FastAPI + WebSocket + Plotly

```python
# WebSocket real-time metrics
@app.websocket("/ws/metrics")
async def metrics_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        metrics = await get_current_metrics()
        await websocket.send_json({
            "regret_oracle": metrics.regret_vs_oracle,
            "quality_trend": metrics.quality_trend,
            "convergence": metrics.converged,
            "model_distribution": metrics.model_counts
        })
        await asyncio.sleep(5)
```

---

## Measurement Workflow

### Phase 1: Initial Deployment (0-2K queries)
- **Algorithm**: UCB1 (non-contextual exploration)
- **Metrics**: Basic regret vs random
- **Frequency**: Compute hourly
- **Goal**: Gather diverse model performance data

### Phase 2: Learning (2K-10K queries)
- **Algorithm**: LinUCB (contextual exploitation)
- **Metrics**: Regret vs oracle, quality trends
- **Frequency**: Compute hourly + daily summaries
- **Arbiter**: 10% sampling for automated evaluation
- **Goal**: Converge to near-optimal routing

### Phase 3: Optimization (10K+ queries)
- **Algorithm**: Hybrid UCB1→LinUCB with PCA
- **Metrics**: All metrics + convergence status
- **Frequency**: Real-time + daily batch analysis
- **Loom**: Weekly A/B tests for improvements
- **Clustering**: Identify routing patterns
- **Goal**: Maintain 5% regret vs oracle

### Phase 4: Production Monitoring
- **Dashboard**: Real-time WebSocket updates
- **Alerts**: Regret > 10% threshold
- **Reports**: Daily quality summaries
- **Experiments**: Continuous improvement via Loom
- **Goal**: Sustained optimal performance

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Conduit Router                        │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │   Query     │──▶│  LinUCB      │──▶│   Response     │  │
│  │  Analysis   │   │  Routing     │   │   Execution    │  │
│  └─────────────┘   └──────────────┘   └────────────────┘  │
│         │                  │                    │           │
└─────────┼──────────────────┼────────────────────┼───────────┘
          │                  │                    │
          ▼                  ▼                    ▼
  ┌───────────────┐  ┌──────────────┐    ┌──────────────┐
  │   pgvector    │  │ evaluation_  │    │    Arbiter   │
  │  (embeddings) │  │   metrics    │    │ (LLM-as-     │
  │               │  │   (regret)   │    │  judge)      │
  └───────────────┘  └──────────────┘    └──────────────┘
          │                  │                    │
          └──────────────────┼────────────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │     Loom     │
                     │  (Batch      │
                     │   Pipeline)  │
                     └──────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │   Dashboard      │
                   │  (Real-time      │
                   │   Metrics)       │
                   └──────────────────┘
```

---

## Success Metrics

### Immediate (Tier 1 - Week 1)
- ✅ Regret calculation implemented
- ✅ Quality trends tracked
- ✅ Cost efficiency computed
- ✅ Hourly metric computation scheduled

### Short-Term (Tier 2 - Month 1)
- 📋 Arbiter integration complete
- 📋 10% of responses auto-evaluated
- 📋 Embeddings database operational
- 📋 Fast similarity search (< 50ms)

### Long-Term (Tier 3 - Quarter 1)
- 📋 Loom pipelines for A/B testing
- 📋 Query clustering analysis
- 📋 Real-time dashboard deployed
- 📋 Continuous improvement workflow established

### Performance Targets
- **Regret vs Oracle**: < 5% (near-optimal routing)
- **Regret vs Random**: > 15% (significant improvement)
- **Quality Trend**: Stable or improving over time
- **Cost Efficiency**: Maximize quality/cost ratio
- **Convergence**: < 10K queries to < 5% regret

---

## Cost Analysis

### Operational Costs (Per Day)
- **Routing Queries**: 1,000 queries × $0.001 avg = **$1.00**
- **Arbiter Evaluation**: 100 evals (10%) × $0.01 avg = **$1.00**
- **Loom Batch Analysis**: 1 daily run × $0.50 = **$0.50**
- **Total**: **$2.50/day** ($75/month)

### Development Costs (One-Time)
- **Tier 1 (Core Metrics)**: ✅ Complete
- **Tier 2 (Arbiter + pgvector)**: ~20 hours
- **Tier 3 (Loom + Dashboard)**: ~40 hours
- **Total**: ~60 hours development

### ROI Calculation
Assuming 30% cost reduction from optimal routing:
- **Baseline**: 1,000 queries/day × $0.01 avg = $10/day
- **With Conduit**: 30% reduction = **$3/day saved**
- **Net Benefit**: $3.00 - $2.50 = **$0.50/day** ($15/month)
- **Plus**: Quality improvements, latency optimization, automation value

---

## Related Issues

- **#42**: Add evaluation_metrics table (Tier 1) - ✅ Implemented
- **#42**: LLM-as-Judge with Arbiter (Tier 2) - 📋 Planned
- **#43**: Embeddings database with pgvector (Tier 2) - 📋 Planned
- **#44**: Batch evaluation with Loom (Tier 3) - 📋 Planned
- **#45**: Query clustering analysis (Tier 3) - 📋 Planned
- **#46**: Real-time metrics dashboard (Tier 3) - 📋 Planned

---

## References

- **Arbiter**: https://github.com/MisfitIdeas/arbiter
- **Loom**: https://github.com/MisfitIdeas/loom
- **pgvector**: https://github.com/pgvector/pgvector
- **Regret Analysis**: See `docs/BANDIT_ALGORITHMS.md`
- **Baseline Algorithms**: `conduit/engines/bandits/baselines.py`
