# Implicit Feedback System

**Purpose**: Learn from behavioral signals without explicit user ratings
**Status**: Phase 2 Complete (2025-11-19)
**Coverage**: 98-100% (76 comprehensive tests)

## Overview

The implicit feedback system implements the "Observability Trinity" - learning from errors, latency patterns, and retry behavior to improve routing decisions without requiring explicit user ratings.

### Key Principle

**Behavioral signals reveal user satisfaction more reliably than explicit ratings:**
- Users who retry queries are dissatisfied (even if they don't rate it)
- Fast responses with no errors suggest satisfaction
- Model errors clearly indicate routing failures

### Weighted Integration

Implicit feedback is weighted at 30% vs explicit feedback at 70%:
- Explicit ratings (quality_score, user_rating) are more reliable when available
- Implicit signals fill gaps when users don't provide ratings
- Combined approach maximizes learning signal

## The Observability Trinity

### 1. Error Detection

**Purpose**: Identify model failures and routing mistakes

**Signal Types**:
- **Hard Errors**: Execution failures (status="error")
- **Empty Responses**: Response text < 10 characters
- **Content Errors**: Model refusals ("I cannot", "I apologize, but I")

**Implementation**: `conduit/feedback/signals.py:SignalDetector.detect_error()`

```python
error_patterns = [
    "I apologize, but I",
    "I cannot",
    "Error:",
    "Exception:"
]

for pattern in error_patterns:
    if pattern.lower() in response_text.lower():
        return ErrorSignal(occurred=True, error_type="model_refusal_or_error")
```

**Reward Mapping**: Error occurred → 0.0 reward (always failure)

### 2. Latency Tracking

**Purpose**: Measure user patience and response time tolerance

**Tolerance Categories**:
- **High** (< 10s): Fast, user satisfied with speed
- **Medium** (10-30s): Acceptable, but borderline
- **Low** (> 30s): Too slow, user likely frustrated

**Implementation**: `conduit/feedback/signals.py:LatencySignal.categorize_tolerance()`

```python
def categorize_tolerance(self) -> None:
    if self.actual_latency_seconds > 30:
        self.tolerance_level = "low"
    elif self.actual_latency_seconds > 10:
        self.tolerance_level = "medium"
    else:
        self.tolerance_level = "high"
```

**Reward Mapping**:
- High tolerance (< 10s) → 0.9 reward
- Medium tolerance (10-30s) → 0.7 reward
- Low tolerance (> 30s) → 0.5 reward

### 3. Retry Detection

**Purpose**: Identify when users retry queries due to dissatisfaction

**Detection Method**: Semantic similarity using cosine similarity on query embeddings

**Parameters**:
- **Similarity Threshold**: 0.85 (highly similar queries)
- **Time Window**: 5 minutes (300 seconds)
- **Storage**: Redis with automatic expiration

**Implementation**: `conduit/feedback/history.py:QueryHistoryTracker.find_similar_query()`

```python
async def find_similar_query(
    self,
    current_embedding: list[float],
    user_id: str,
    similarity_threshold: float = 0.85
) -> QueryHistoryEntry | None:
    recent_queries = await self.get_recent_queries(user_id, limit=10)

    best_similarity = 0.0
    best_match = None

    for entry in recent_queries:
        similarity = self._cosine_similarity(current_embedding, entry.embedding)
        if similarity >= similarity_threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = entry

    return best_match
```

**Reward Mapping**: Retry detected → 0.3 reward (strong negative signal)

## Architecture

### Component Hierarchy

```
ImplicitFeedbackDetector (orchestrator)
├── SignalDetector (stateless detector)
│   ├── detect_error() → ErrorSignal
│   └── detect_latency() → LatencySignal
├── QueryHistoryTracker (Redis storage)
│   ├── add_query() → Store for retry detection
│   └── find_similar_query() → Detect retries
└── ImplicitFeedback (result model)
```

### Data Flow

```
1. Query Execution
   ├── request_start_time (timestamp)
   ├── Execute LLM call
   ├── response_complete_time (timestamp)
   └── Collect response_text, execution_status, execution_error

2. Signal Detection (ImplicitFeedbackDetector.detect())
   ├── Error Detection
   │   └── Check execution_status, response_text patterns
   ├── Latency Detection
   │   └── Calculate latency, categorize tolerance
   └── Retry Detection
       ├── Check recent query history (Redis)
       └── Cosine similarity on embeddings

3. Feedback Integration (FeedbackIntegrator)
   ├── Convert signals to reward (0.0-1.0)
   ├── Weight by implicit_weight (0.3)
   └── Update Thompson Sampling (bandit.update())

4. History Storage
   └── Store current query in Redis (5-min TTL)
```

### Key Classes

#### QueryHistoryTracker
**File**: `conduit/feedback/history.py`
**Purpose**: Redis-based query history with automatic expiration

```python
class QueryHistoryTracker:
    def __init__(self, redis: Redis | None = None, ttl_seconds: int = 300):
        self.redis = redis
        self.ttl = ttl_seconds
        self.enabled = redis is not None
```

**Methods**:
- `add_query()`: Store query with embedding and metadata (5-min TTL)
- `find_similar_query()`: Find most similar recent query above threshold
- `get_recent_queries()`: Retrieve last N queries for user
- `clear_user_history()`: Remove all history for user (testing/privacy)

#### ImplicitFeedbackDetector
**File**: `conduit/feedback/detector.py`
**Purpose**: Orchestrate all signal detection

```python
class ImplicitFeedbackDetector:
    def __init__(self, history: QueryHistoryTracker):
        self.history = history
```

**Methods**:
- `detect()`: Main detection orchestrator, returns ImplicitFeedback
- `_detect_retry()`: Internal retry detection using history tracker

#### FeedbackIntegrator
**File**: `conduit/feedback/integration.py`
**Purpose**: Convert feedback signals to Thompson Sampling rewards

```python
class FeedbackIntegrator:
    def __init__(
        self,
        bandit: ContextualBandit,
        explicit_weight: float = 0.7,
        implicit_weight: float = 0.3
    ):
        self.bandit = bandit
        self.explicit_weight = explicit_weight
        self.implicit_weight = implicit_weight
```

**Methods**:
- `update_from_explicit()`: Process user ratings (quality_score, user_rating, met_expectations)
- `update_from_implicit()`: Process behavioral signals (errors, latency, retries)
- `_explicit_to_reward()`: Convert explicit feedback to 0-1 reward
- `_implicit_to_reward()`: Convert implicit signals to 0-1 reward (priority order)

### Reward Calculation Priority

**Priority Order** (first match wins):
1. **Error occurred** → 0.0 reward (hard failure)
2. **Retry detected** → 0.3 reward (strong negative signal)
3. **Latency tolerance** → 0.5-0.9 reward (based on speed)

```python
def _implicit_to_reward(self, feedback: ImplicitFeedback) -> float:
    # Priority 1: Error occurred
    if feedback.error_occurred:
        return 0.0

    # Priority 2: Retry detected
    if feedback.retry_detected:
        return 0.3

    # Priority 3: Latency tolerance
    latency_rewards = {
        "high": 0.9,   # < 10s
        "medium": 0.7, # 10-30s
        "low": 0.5     # > 30s
    }
    return latency_rewards.get(feedback.latency_tolerance, 0.7)
```

## Database Schema

**Migration**: `migrations/002_implicit_feedback.sql`

```sql
CREATE TABLE implicit_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Error signals
    error_occurred BOOLEAN NOT NULL DEFAULT FALSE,
    error_type TEXT,

    -- Latency signals
    latency_seconds FLOAT NOT NULL CHECK (latency_seconds >= 0.0),
    latency_accepted BOOLEAN NOT NULL DEFAULT TRUE,
    latency_tolerance TEXT CHECK (latency_tolerance IN ('high', 'medium', 'low')),

    -- Retry signals
    retry_detected BOOLEAN NOT NULL DEFAULT FALSE,
    retry_delay_seconds FLOAT CHECK (retry_delay_seconds IS NULL OR retry_delay_seconds >= 0.0),
    similarity_score FLOAT CHECK (similarity_score IS NULL OR (similarity_score >= 0.0 AND similarity_score <= 1.0)),
    original_query_id UUID REFERENCES queries(id),

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_implicit_feedback_query ON implicit_feedback(query_id);
CREATE INDEX idx_implicit_feedback_model ON implicit_feedback(model_id);
CREATE INDEX idx_implicit_feedback_timestamp ON implicit_feedback(timestamp);
```

## Redis Data Structures

### Query History Entry
**Key Pattern**: `conduit:history:{user_id}:{query_id}`
**TTL**: 300 seconds (5 minutes)
**Format**: JSON

```json
{
    "query_id": "uuid",
    "query_text": "What is Python?",
    "embedding": [0.123, -0.456, ...],  // 384-dimensional vector
    "timestamp": 1700000000.0,
    "model_used": "gpt-4o-mini"
}
```

### User History Index
**Key Pattern**: `conduit:history:{user_id}:index`
**Type**: Sorted Set
**Score**: Timestamp (for chronological ordering)

```
ZADD conduit:history:user_abc:index 1700000000.0 "query_id_1"
ZADD conduit:history:user_abc:index 1700000100.0 "query_id_2"
```

## Graceful Degradation

**Without Redis**:
- Core routing still works
- Error detection still works (no Redis required)
- Latency tracking still works (no Redis required)
- Retry detection **disabled** (requires history storage)
- System logs warning and continues

**Detection**:
```python
if not self.history.enabled:
    # Skip retry detection
    retry_signal = RetrySignal(detected=False)
```

## Testing

**Test Files**:
- `tests/unit/test_feedback_signals.py` (28 tests)
- `tests/unit/test_query_history.py` (27 tests)
- `tests/unit/test_feedback_integration.py` (21 tests)

**Total**: 76 tests, 98-100% coverage

**Test Categories**:
1. **Signal Detection**
   - Error patterns (empty response, model refusal, execution errors)
   - Latency categorization (high/medium/low tolerance)
   - Edge cases (zero latency, very high latency)

2. **Query History**
   - Redis storage and retrieval
   - TTL expiration
   - Similarity calculations
   - Concurrent queries
   - History clearing

3. **Feedback Integration**
   - Reward calculations (priority order)
   - Thompson Sampling updates
   - Weighted feedback (70/30 split)
   - Edge cases (simultaneous signals)

## Usage Examples

### Basic Implicit Feedback Detection

```python
from redis.asyncio import Redis
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.feedback import (
    ImplicitFeedbackDetector,
    QueryHistoryTracker,
    FeedbackIntegrator
)

# Setup
redis = Redis.from_url("redis://localhost:6379")
analyzer = QueryAnalyzer()
bandit = ContextualBandit(models=["gpt-4o-mini", "gpt-4o"])
history = QueryHistoryTracker(redis=redis)
detector = ImplicitFeedbackDetector(history)
integrator = FeedbackIntegrator(bandit)

# Analyze query
query = Query(text="What is Python?")
features = await analyzer.analyze(query.text)

# Execute and time
start = time.time()
# ... execute LLM call ...
end = time.time()

# Detect signals
feedback = await detector.detect(
    query=query.text,
    query_id=query.id,
    features=features,
    response_text="Python is a programming language...",
    model_id="gpt-4o-mini",
    execution_status="success",
    execution_error=None,
    request_start_time=start,
    response_complete_time=end,
    user_id="user_abc"
)

# Update bandit
integrator.update_from_implicit("gpt-4o-mini", features, feedback)
```

### Combined Explicit + Implicit Feedback

```python
# Explicit feedback (user rating)
explicit = Feedback(
    response_id="resp_001",
    quality_score=0.95,  # 0-1 scale
    user_rating=5,       # 1-5 stars
    met_expectations=True
)

# Implicit feedback (behavioral signals)
implicit = await detector.detect(...)

# Update with both (weighted 70% explicit, 30% implicit)
integrator.update_from_explicit("gpt-4o-mini", features, explicit)
integrator.update_from_implicit("gpt-4o-mini", features, implicit)
```

See `examples/03_optimization/combined_feedback.py` for complete working example.

## Configuration

### Tunable Parameters

**Retry Detection**:
```python
QueryHistoryTracker(
    redis=redis,
    ttl_seconds=300,  # 5-minute window (default)
)

await tracker.find_similar_query(
    current_embedding=embedding,
    user_id=user_id,
    similarity_threshold=0.85  # 85% similarity required (default)
)
```

**Latency Thresholds**:
```python
# In LatencySignal.categorize_tolerance()
if latency_seconds > 30:      # Low tolerance threshold
    tolerance_level = "low"
elif latency_seconds > 10:    # Medium tolerance threshold
    tolerance_level = "medium"
else:
    tolerance_level = "high"
```

**Feedback Weights**:
```python
FeedbackIntegrator(
    bandit=bandit,
    explicit_weight=0.7,  # User ratings weight (default)
    implicit_weight=0.3   # Behavioral signals weight (default)
)
```

## Performance Characteristics

**Latency**:
- Error detection: < 1ms (pattern matching)
- Latency categorization: < 1ms (simple arithmetic)
- Retry detection: 5-10ms (Redis lookup + cosine similarity)
- Total overhead: < 15ms per query

**Storage**:
- Redis memory: ~2KB per query (embedding + metadata)
- Automatic cleanup: 5-minute TTL
- Estimated capacity: 100,000 queries = ~200MB RAM

**Scalability**:
- Redis handles 10,000+ queries/sec
- Cosine similarity is O(n) where n = embedding dimensions (384)
- History lookup is O(log n) where n = queries in window

## Design Decisions

### Why 5-Minute TTL?

**Rationale**: Balance between retry detection accuracy and memory usage
- Most retries happen within 1-2 minutes
- 5 minutes captures virtually all retry patterns
- Automatic cleanup prevents unbounded memory growth

### Why 0.85 Similarity Threshold?

**Rationale**: Tested empirically to balance false positives/negatives
- 0.80: Too many false positives (different queries flagged as retries)
- 0.85: Good balance (catches paraphrases, avoids false matches)
- 0.90: Too strict (misses legitimate retries with slight wording changes)

### Why 70/30 Explicit/Implicit Split?

**Rationale**: Explicit ratings are more reliable when available
- Users who rate are consciously evaluating quality
- Behavioral signals can be noisy (retries for exploration, etc.)
- 70/30 weights explicit higher while still learning from behavior

### Why Priority Order (Error > Retry > Latency)?

**Rationale**: Signals have different reliability levels
- Errors are unambiguous failures
- Retries are strong negative signals
- Latency is contextual (some queries naturally take longer)

## Future Enhancements

**Phase 3+**:
1. **Adaptive Thresholds**: Learn optimal similarity threshold per user
2. **Context-Aware Latency**: Different thresholds for simple vs complex queries
3. **Retry Intent Classification**: Distinguish dissatisfaction retries from exploratory retries
4. **Temporal Patterns**: Learn user-specific patience patterns (morning vs evening)
5. **A/B Testing**: Experiment with different weight combinations

## References

- **Implementation**: `conduit/feedback/` (detector.py, history.py, integration.py, signals.py)
- **Tests**: `tests/unit/test_feedback_*.py` (76 tests, 98-100% coverage)
- **Examples**: `examples/03_optimization/implicit_feedback.py`, `combined_feedback.py`
- **Migration**: `migrations/002_implicit_feedback.sql`
- **Models**: `conduit/core/models.py` (ImplicitFeedback, ErrorSignal, LatencySignal, RetrySignal)
