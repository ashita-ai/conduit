# Conduit Bootstrap Strategy

## Problem: Cold Start

When Conduit first starts, it has:
- **No cost data** for models (doesn't know pricing)
- **No performance data** (doesn't know quality, latency)
- **No usage history** (contextual bandits need training data)

How do we make intelligent routing decisions from day one?

## Solution: 3-Tier Fallback System

### Tier 1: YAML Configuration (Primary)

**Location**: `pricing.yaml` and `conduit.yaml`

**Purpose**: Provide curated defaults based on public pricing and community knowledge.

**Pricing Data** (`pricing.yaml`):
```yaml
pricing:
  o4-mini:
    input: 1.10   # $ per 1M tokens
    output: 4.40
  gpt-5.1:
    input: 2.00
    output: 8.00
  # ... 30+ models
```

**Algorithm Parameters** (`conduit.yaml`):
```yaml
algorithms:
  linucb:
    alpha: 1.0                # Exploration parameter
    success_threshold: 0.85   # Quality threshold

quality_estimation:
  base_quality: 0.9          # Assume good quality initially
  penalties:
    short_response: 0.15     # Penalize very short responses
```

### Tier 2: Environment Variables (Overrides)

**Purpose**: Allow runtime configuration without editing YAML files.

**Examples**:
```bash
export ALGORITHM_LINUCB_ALPHA=2.0
export PRICING_O4_MINI_INPUT=1.50
export ROUTING_DEFAULT_OPTIMIZATION=cost
```

**Priority**: Environment variables override YAML configuration.

### Tier 3: Hardcoded Defaults (Ultimate Fallback)

**Purpose**: System works even without any configuration files.

**Location**: Embedded directly in `conduit/core/config.py` loader functions.

**Example**:
```python
def load_pricing_config() -> dict[str, dict[str, float]]:
    """Load pricing with 3-tier fallback."""
    defaults = {
        "o4-mini": {"input": 1.10, "output": 4.40},
        "gpt-5.1": {"input": 2.00, "output": 8.00},
        # ... ultimate fallback if YAML missing
    }
    # Try YAML → env vars → hardcoded defaults
```

## Bootstrap Phases

### Phase 1: First 100 Queries (UCB1)

**Algorithm**: UCB1 (non-contextual bandit)
- Simple exploration/exploitation
- Only needs model names, not query features
- Fast convergence for initial learning

**Data Sources**:
- Pricing: `pricing.yaml` or hardcoded defaults
- Quality: Estimated from response content (length, repetition, keyword overlap)
- Latency: Measured from API calls

**Learning**: Builds initial model performance statistics.

### Phase 2: After 100 Queries (LinUCB)

**Algorithm**: LinUCB (contextual bandit)
- Uses query features (embedding, token count, complexity)
- Learns which models are best for which types of queries
- More sophisticated routing decisions

**Data Sources**:
- Pricing: Still from YAML/defaults
- Quality: Improved estimates + optional LLM-as-judge (Arbiter)
- Latency: Measured averages per model
- Context: 387-dim feature vectors (384 embedding + 3 metadata)

**Learning**: Continuously refines routing based on query similarity.

### Phase 3: Production (Continuous Learning)

**Ongoing**:
- Bandits adapt to changing model performance
- New models can be added dynamically
- User preferences override ML decisions when specified

## Cost & Performance Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                   Query Arrives                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├─> Extract features (embedding, tokens, complexity)
                     │
                     ├─> Route through bandit (UCB1 or LinUCB)
                     │   │
                     │   ├─> Check pricing.yaml for costs
                     │   ├─> Use quality estimation defaults
                     │   └─> Select optimal model
                     │
                     ├─> Execute query with selected model
                     │   │
                     │   ├─> Measure actual latency
                     │   ├─> Capture actual cost (from LiteLLM)
                     │   └─> Estimate quality (response analysis)
                     │
                     └─> Provide feedback to bandit
                         │
                         ├─> Update model statistics
                         └─> Improve future routing decisions
```

## Updating Pricing Data

### Option 1: Manual YAML Updates (Current)

**Pros**:
- Simple, version-controlled
- Easy to review changes
- No database dependencies

**Cons**:
- Requires manual updates when prices change
- No automatic price discovery

**Process**:
```bash
# Edit pricing.yaml
vi pricing.yaml

# Update model prices
pricing:
  o4-mini:
    input: 1.25  # Updated from 1.10
    output: 4.50 # Updated from 4.40

# Restart application (hot-reload not implemented)
```

**Update Frequency**: Check monthly at:
- https://openai.com/api/pricing
- https://anthropic.com/pricing
- https://ai.google.dev/pricing

### Option 2: Database-Backed Pricing (Recommended for Production)

**Pros**:
- Automatic price discovery from actual API costs
- Historical price tracking
- No restarts needed for updates
- Can detect price changes automatically

**Cons**:
- Requires PostgreSQL setup
- More complex configuration

**Implementation**:
```python
# In conduit/core/pricing.py
def get_model_price(model_id: str, token_type: str) -> float:
    """Get price with DB → YAML → hardcoded fallback."""

    # Tier 1: Check database (learned from actual costs)
    if db_price := await db.get_model_price(model_id, token_type):
        return db_price

    # Tier 2: Check YAML configuration
    pricing_config = load_pricing_config()
    if model_id in pricing_config:
        return pricing_config[model_id][token_type]

    # Tier 3: Hardcoded fallback
    return DEFAULT_PRICES.get(model_id, {}).get(token_type, 1.0)
```

**Price Learning**:
```python
# After each API call (in ConduitFeedbackLogger)
actual_cost = response._hidden_params['response_cost']
tokens_used = response.usage.total_tokens

# Calculate actual price per million tokens
price_per_million = (actual_cost / tokens_used) * 1_000_000

# Store in database
await db.save_model_price(
    model_id=model_id,
    token_type="combined",  # or separate input/output
    price=price_per_million,
    timestamp=datetime.now()
)
```

### Option 3: Hybrid Approach (Best of Both)

**Strategy**:
1. Start with `pricing.yaml` defaults (bootstrap)
2. Learn actual costs from API responses (production)
3. Update `pricing.yaml` periodically from learned data (continuous improvement)

**Configuration**:
```yaml
# conduit.yaml
pricing:
  source: hybrid            # yaml | database | hybrid
  learning_enabled: true    # Learn from actual costs
  update_threshold: 100     # Update after N samples
  confidence_threshold: 0.9 # Only update if confident
```

## Quality Estimation Bootstrap

### Without LLM-as-Judge

**Method**: Heuristic analysis of response content

**Signals**:
- Response length (too short = low quality)
- Repetition detection (repeated phrases = low quality)
- Keyword overlap with query (no overlap = off-topic)
- Error patterns (apology phrases = refusal/error)

**Configuration** (`conduit.yaml`):
```yaml
quality_estimation:
  base_quality: 0.9
  min_response_chars: 50
  penalties:
    short_response: 0.15
    repetition: 0.30
    no_keyword_overlap: 0.10
  thresholds:
    keyword_overlap_very_low: 0.05
    repetition_min_length: 20
```

### With LLM-as-Judge (Arbiter)

**Method**: Sample responses evaluated by cheap LLM (o4-mini)

**Configuration**:
```yaml
arbiter:
  sample_rate: 0.1          # Evaluate 10% of responses
  daily_budget: 10.0        # Max $10/day on evaluation
  model: o4-mini            # Cheap, fast evaluator
  evaluators:               # Types of evaluation
    - semantic              # Answer quality
    - factuality            # Correctness
```

**Usage**:
```python
from conduit.evaluation import ArbiterEvaluator

evaluator = ArbiterEvaluator(
    db=database,
    sample_rate=0.1,
    daily_budget=10.0
)

strategy = ConduitRoutingStrategy(
    evaluator=evaluator  # Enable LLM-as-judge
)
```

## Recommendations

### For Development/Testing:
- Use YAML configuration (simplest)
- Manual pricing updates (fine for small scale)
- Heuristic quality estimation (no extra costs)

### For Production:
- **Pricing**: Hybrid (YAML bootstrap + database learning)
- **Quality**: LLM-as-judge with sampling (10% @ $10/day = ~$300/month)
- **Updates**: Automated database learning with manual YAML review monthly

### Price Update Schedule:
- **Daily**: Database learning from actual costs (automatic)
- **Weekly**: Review database-learned prices for anomalies
- **Monthly**: Update `pricing.yaml` from verified database averages
- **Quarterly**: Audit against public pricing pages

## Performance Considerations

### Memory:
- YAML configs loaded once at startup (~10KB)
- Database pricing cache (optional, ~1MB for 1000 models)
- Bandit state (grows with usage, ~100KB per 1000 queries)

### Latency:
- YAML pricing lookup: ~1μs (in-memory dict)
- Database pricing lookup: ~5ms (PostgreSQL query with caching)
- Quality estimation: ~10ms (text analysis)
- LLM-as-judge: ~500ms (only for sampled 10%)

### Cost:
- YAML configuration: $0
- Database storage: <$1/month (tiny table)
- Quality estimation: $0
- LLM-as-judge: ~$300/month @ 10% sampling, 100K queries/day

## Migration Path

### Current State:
✅ YAML configuration implemented
✅ Hardcoded defaults as fallback
✅ Environment variable overrides
✅ Heuristic quality estimation

### Next Steps (Optional):
1. Database-backed pricing (PR #X)
2. Automated price learning (PR #Y)
3. Hot-reload for YAML changes (PR #Z)
4. Price anomaly detection (alert if learned price deviates >20% from YAML)

## FAQ

**Q: Do I need to set up a database?**
A: No - Conduit works with just YAML files. Database is optional for advanced production deployments.

**Q: How often should I update pricing.yaml?**
A: Monthly checks are sufficient. Prices rarely change more than once per quarter.

**Q: What if a new model isn't in pricing.yaml?**
A: The system will use a conservative default (currently $1.00 per 1M tokens) until you add it.

**Q: Can I override pricing per-customer?**
A: Yes - use environment variables or create customer-specific YAML files loaded at runtime.

**Q: Does Conduit work offline?**
A: Routing logic works offline (uses local models for embeddings). API calls require internet.
