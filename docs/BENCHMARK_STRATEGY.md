# Benchmark Strategy

**Goal**: Demonstrate Conduit's 30-50% cost savings claim with empirical evidence

## Multi-Baseline Approach

### Why Multiple Baselines?

Different users have different comparison points. We need to show value against:

1. **"Always Premium"** - Users who prioritize quality over cost
2. **"Manual Routing"** - Users who use smart heuristics today
3. **"Random"** - Worst case scenario (sanity check)

This gives us multiple value propositions:
- "50% cheaper than always-GPT-4o"
- "20-30% better than manual routing"
- "Beats human judgment with ML"

## Baseline Definitions

### Baseline A: Always Premium (GPT-4o)

**Strategy**: Route every query to GPT-4o

**Rationale**: What users do when they want guaranteed quality

**Expected Cost** (per 1000 queries):
```python
# Assumptions
avg_input_tokens = 200
avg_output_tokens = 400
gpt4o_input_price = $2.50 / 1M tokens
gpt4o_output_price = $10.00 / 1M tokens

cost = 1000 * (
    (200 * $2.50 / 1M) +  # Input: $0.50
    (400 * $10.00 / 1M)   # Output: $4.00
) = $4.50 per 1000 queries
```

**Expected Performance**:
- Quality: 90%+ success rate
- Latency: ~2-3 seconds average
- Error rate: <5%

**Value Prop**: "Conduit saves 50% vs always-premium while maintaining quality"

### Baseline B: Manual Routing (Heuristics)

**Strategy**: Simple IF/ELSE rules based on complexity

```python
def manual_route(query: str) -> str:
    # Count tokens as complexity proxy
    token_count = len(query.split())

    if token_count < 20:
        return "gpt-4o-mini"  # Short query → cheap
    elif "code" in query.lower() or "debug" in query.lower():
        return "gpt-4o"  # Code query → premium
    elif token_count > 100:
        return "gpt-4o"  # Long query → premium
    else:
        return "gpt-4o-mini"  # Default → cheap
```

**Expected Distribution**:
- 60% gpt-4o-mini ($0.15/1M input, $0.60/1M output)
- 40% gpt-4o ($2.50/1M input, $10.00/1M output)

**Expected Cost** (per 1000 queries):
```python
# gpt-4o-mini (60%)
mini_cost = 600 * ((200 * $0.15 / 1M) + (400 * $0.60 / 1M))
         = 600 * ($0.03 + $0.24) = $162

# gpt-4o (40%)
gpt4_cost = 400 * ((200 * $2.50 / 1M) + (400 * $10.00 / 1M))
          = 400 * ($0.50 + $4.00) = $1800

total = $162 + $1800 = $1962 = $1.96 per 1000 queries
```

**Expected Performance**:
- Quality: 80-85% success rate (some misrouting)
- Latency: ~2 seconds average
- Error rate: 10-15% (budget model failures on complex queries)

**Value Prop**: "Conduit saves 20-30% vs manual routing and catches edge cases humans miss"

### Baseline C: Random (Sanity Check)

**Strategy**: Random selection among all models

**Expected Distribution**:
- 25% each model (gpt-4o-mini, gpt-4o, claude-sonnet-4, claude-haiku)

**Expected Cost**: $2.30 per 1000 queries (weighted average)

**Expected Performance**:
- Quality: 70-75% (many mismatches)
- Error rate: 20-25%

**Value Prop**: Sanity check only - proves intelligent routing matters

## Conduit Strategy

### Model Pool

```python
models = [
    "gpt-4o-mini",      # $0.15/1M in, $0.60/1M out - Budget
    "gpt-4o",           # $2.50/1M in, $10.00/1M out - Premium OpenAI
    "claude-sonnet-4",  # $3.00/1M in, $15.00/1M out - Premium Anthropic
    "claude-haiku",     # $0.25/1M in, $1.25/1M out - Ultra-budget
]
```

### Learning Phases

**Phase 1: Cold Start (Queries 1-100)**
- Use informed priors + contextual heuristics
- High exploration to gather data
- Expected cost: $2.50 per 1000 (higher than optimal, learning phase)

**Phase 2: Learning (Queries 100-500)**
- Patterns emerge, confidence grows
- Balanced exploration/exploitation
- Expected cost: $2.00 per 1000 (improving)

**Phase 3: Converged (Queries 500-1000)**
- Stable routing decisions
- Minimal exploration, mostly exploitation
- Expected cost: $1.50-1.80 per 1000 (optimized for this workload)

### Expected Final Distribution

Based on typical mixed workload:

```python
conduit_distribution = {
    "gpt-4o-mini": 0.50,      # 50% simple queries
    "gpt-4o": 0.30,           # 30% complex queries
    "claude-sonnet-4": 0.15,  # 15% specific use cases
    "claude-haiku": 0.05      # 5% ultra-simple queries
}
```

**Target Cost**: $1.50-2.00 per 1000 queries (after convergence)

## Workload Design

### Requirements

**Size**: 1,000 diverse queries

**Domains**:
- Customer Support (30%): FAQ, troubleshooting, account questions
- Code Generation (25%): Write functions, debug code, explain algorithms
- Content Writing (25%): Blog posts, social media, creative writing
- Data Analysis (20%): Summarize data, explain trends, create reports

**Complexity Distribution**:
- Simple (40%): < 50 tokens, straightforward
- Medium (40%): 50-200 tokens, moderate complexity
- Complex (20%): > 200 tokens, multi-step reasoning

### Sample Queries

**Simple (should route to budget)**:
```
- "What are your business hours?"
- "How do I reset my password?"
- "What is 15% of 80?"
- "Define 'machine learning' in one sentence"
```

**Medium (could route to either)**:
```
- "Write a function to calculate fibonacci numbers"
- "Explain the difference between REST and GraphQL"
- "Summarize this customer feedback: [200 word review]"
- "Create an email template for new user onboarding"
```

**Complex (should route to premium)**:
```
- "Debug this React component that's causing infinite re-renders: [code]"
- "Design a microservices architecture for an e-commerce platform"
- "Analyze these sales trends and predict next quarter: [data]"
- "Write a comprehensive blog post on AI ethics (1000 words)"
```

### Workload Sources

1. **Real Production Data** (if available, anonymized)
2. **Public Datasets**:
   - ShareGPT conversations
   - StackOverflow questions
   - GitHub issues
3. **Synthetic Generation**:
   - Use GPT-4o to generate diverse queries
   - Ensure balanced distribution across domains/complexity

## Benchmark Execution

### Setup

```python
# Baseline A: Always Premium
baseline_a = AlwaysPremiumRouter(model="gpt-4o")

# Baseline B: Manual Routing
baseline_b = ManualRouter(
    simple_model="gpt-4o-mini",
    complex_model="gpt-4o",
    complexity_threshold=20  # tokens
)

# Conduit: ML Routing
conduit = Router(
    models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-haiku"],
    cold_start_mode="informed+heuristic"
)
```

### Execution Flow

```python
results = {
    "baseline_a": [],
    "baseline_b": [],
    "conduit": []
}

for i, query in enumerate(workload):
    # Run all three approaches
    for approach in ["baseline_a", "baseline_b", "conduit"]:
        start_time = time.time()

        # Execute
        response = await approach.route_and_execute(query)

        # Collect metrics
        result = {
            "query_id": query.id,
            "query": query.text,
            "model_used": response.model,
            "cost": response.cost,
            "latency": time.time() - start_time,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "error": response.error,
            "response_text": response.text
        }

        results[approach].append(result)

        # Update Conduit with feedback (only for Conduit, not baselines)
        if approach == "conduit":
            feedback = await detect_implicit_feedback(response)
            conduit.update(response.model, feedback)
```

### Quality Evaluation

**Automated Metrics**:
```python
def evaluate_quality(response) -> float:
    """Automated quality assessment."""
    score = 0.0

    # Error detection
    if not response.error:
        score += 0.3

    # Length appropriateness
    if 50 < len(response.text.split()) < 1000:
        score += 0.2

    # Latency tolerance
    if response.latency < 5:
        score += 0.2

    # Content patterns (basic heuristics)
    if not any(pattern in response.text.lower() for pattern in
               ["i apologize", "i cannot", "error", "sorry"]):
        score += 0.3

    return score
```

**Human Evaluation** (sample):
```python
# Randomly sample 100 queries for human rating
sample_queries = random.sample(workload, 100)

for query in sample_queries:
    # Show all 3 responses
    print(f"Query: {query.text}")
    print(f"Baseline A (GPT-4o): {baseline_a_response}")
    print(f"Baseline B (Manual): {baseline_b_response}")
    print(f"Conduit (Learned): {conduit_response}")

    # Human rates each on 1-5 scale
    ratings = get_human_rating(query, responses)
```

## Analysis & Reporting

### Cost Comparison

```python
cost_analysis = {
    "baseline_a": {
        "total_cost": $4500,  # For 1000 queries
        "cost_per_query": $4.50,
        "model_distribution": {"gpt-4o": 1.0}
    },
    "baseline_b": {
        "total_cost": $1960,
        "cost_per_query": $1.96,
        "model_distribution": {
            "gpt-4o-mini": 0.60,
            "gpt-4o": 0.40
        }
    },
    "conduit": {
        "total_cost": $1650,  # After learning
        "cost_per_query": $1.65,
        "model_distribution": {
            "gpt-4o-mini": 0.50,
            "gpt-4o": 0.30,
            "claude-sonnet-4": 0.15,
            "claude-haiku": 0.05
        },
        "learning_phases": {
            "phase_1 (1-100)": $2.50,
            "phase_2 (100-500)": $2.00,
            "phase_3 (500-1000)": $1.65
        }
    }
}

# Calculate savings
savings_vs_a = (4.50 - 1.65) / 4.50 * 100  # 63% savings!
savings_vs_b = (1.96 - 1.65) / 1.96 * 100  # 16% savings
```

### Quality Comparison

```python
quality_analysis = {
    "baseline_a": {
        "success_rate": 0.92,
        "avg_quality_score": 0.88,
        "error_rate": 0.05,
        "avg_latency": 2.3
    },
    "baseline_b": {
        "success_rate": 0.82,
        "avg_quality_score": 0.78,
        "error_rate": 0.12,
        "avg_latency": 1.9
    },
    "conduit": {
        "success_rate": 0.90,  # Close to premium!
        "avg_quality_score": 0.85,
        "error_rate": 0.06,
        "avg_latency": 2.1,
        "quality_maintained": True  # 95%+ of baseline A
    }
}
```

### Report Generation

```markdown
# Conduit Benchmark Results

## Executive Summary

Conduit achieves **63% cost savings** vs always-premium baseline while
maintaining **95%+ quality**.

### Key Findings

1. **Cost Efficiency**
   - 63% cheaper than GPT-4o-only ($1.65 vs $4.50 per 1K queries)
   - 16% cheaper than manual routing ($1.65 vs $1.96)
   - Converges to optimal routing within 500 queries

2. **Quality Maintained**
   - 90% success rate (vs 92% premium baseline)
   - 6% error rate (vs 5% premium baseline)
   - Matches premium quality on 85% of queries

3. **Learning Speed**
   - Meaningful improvements by query 100
   - Near-optimal routing by query 500
   - Continues learning and adapting

## Model Distribution

**Baseline A (Always Premium)**:
- gpt-4o: 100%

**Baseline B (Manual Routing)**:
- gpt-4o-mini: 60%
- gpt-4o: 40%

**Conduit (Learned)**:
- gpt-4o-mini: 50% (simple queries, saves money)
- gpt-4o: 30% (complex queries, ensures quality)
- claude-sonnet-4: 15% (specific use cases)
- claude-haiku: 5% (ultra-simple queries)

## Cost Breakdown

| Approach | Total Cost | Cost/Query | vs Premium | vs Manual |
|----------|-----------|------------|-----------|----------|
| Baseline A (Premium) | $4,500 | $4.50 | - | - |
| Baseline B (Manual) | $1,960 | $1.96 | -56% | - |
| **Conduit (Learned)** | **$1,650** | **$1.65** | **-63%** | **-16%** |

## Quality Metrics

| Metric | Baseline A | Baseline B | Conduit | vs Premium |
|--------|-----------|-----------|---------|-----------|
| Success Rate | 92% | 82% | 90% | -2% ✓ |
| Error Rate | 5% | 12% | 6% | +1% ✓ |
| Avg Latency | 2.3s | 1.9s | 2.1s | -0.2s ✓ |
| Quality Score | 0.88 | 0.78 | 0.85 | 96% ✓ |

**Quality Maintained**: ✓ 95%+ of premium baseline

## Learning Curve

Queries 1-100: $2.50/query (cold start)
Queries 100-500: $2.00/query (learning)
Queries 500-1000: $1.65/query (converged)

## Conclusion

Conduit demonstrates:
1. ✓ 60%+ cost savings vs premium baseline
2. ✓ 95%+ quality maintenance
3. ✓ Fast convergence (500 queries)
4. ✓ Beats human-designed routing heuristics
```

## Success Criteria

**Must Demonstrate**:
- ✓ 30-50% cost savings vs reasonable baseline
- ✓ 95%+ quality vs premium baseline
- ✓ Convergence within 1000 queries
- ✓ p99 latency < 200ms routing overhead

**Marketing Claims** (if achieved):
- "Saves 30-50% on LLM costs"
- "Maintains 95%+ quality"
- "Gets smarter with use"
- "Beats manual routing by 20%"

## Implementation Checklist

- [ ] Create 1000-query workload (diverse domains/complexity)
- [ ] Implement Baseline A (always-premium)
- [ ] Implement Baseline B (manual heuristics)
- [ ] Run Conduit with learning enabled
- [ ] Collect cost, latency, quality metrics
- [ ] Human evaluation on 100-query sample
- [ ] Generate comparison report
- [ ] Validate 30-50% savings claim
- [ ] Document methodology and results

## References

- **Benchmark Repo**: `conduit-benchmark/` (private)
- **Bandit Training**: `docs/BANDIT_TRAINING.md`
- **Cold Start**: `docs/COLD_START.md`
- **Implementation**: `conduit/engines/bandit.py`
