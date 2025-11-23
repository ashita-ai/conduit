# Implement PILOT-Style Preference-Prior Informed Routing

## Priority: Advanced / Future
## Labels: `enhancement`, `algorithm`, `research`, `phase-6`, `offline-learning`

## Overview

Implement **PILOT** (Preference-Prior Informed LinUCB for Adaptive LLM Routing) - an advanced approach that combines **offline preference data** with **online bandit learning**. This addresses the cold start problem by using historical preference data to warm-start the routing algorithm, then refining with online feedback.

## Motivation

**Current Problem**: Cold start requires thousands of queries before optimal routing
- LinUCB needs ~10,000 queries (or 1,500-2,500 with hybrid routing)
- New customers/users start from scratch
- No way to leverage existing preference data

**PILOT Solution**:
- Use offline preference data (human evaluations, A/B tests, etc.) to initialize routing
- Shared embedding space for queries and LLMs (learned from offline data)
- Online bandit learning refines initial preferences
- **Result**: Faster convergence, better cold start performance

**Key Innovation**:
- Aligns query embeddings and LLM embeddings in shared space
- Reflects compatibility between queries and models
- Enables generalization to unseen models

## Research Papers

1. **Primary Paper**: "PILOT: Preference-Prior Informed LinUCB for Adaptive LLM Routing" (arxiv:2508.21141)
   - Link: https://arxiv.org/abs/2508.21141
   - Key Algorithm: LinUCB with preference priors + online cost policy
   - Includes Category-Calibrated Fine-Tuning (CCFT) for embeddings

2. **Related**: "LLM Routing with Dueling Feedback" (arxiv:2510.00841)
   - CCFT embedding method (can reuse for PILOT)
   - Link: https://arxiv.org/abs/2510.00841

3. **Background**: "Learning to Rank" literature
   - Learning-to-rank algorithms for preference modeling
   - Embedding alignment techniques

## Implementation Plan

### Phase 1: Offline Preference Data Pipeline

**File**: `conduit/engines/bandits/offline_learning.py` (new)

**Components**:
1. **Preference Data Loader**:
   - Load offline preference data (CSV, JSON, database)
   - Format: (query_text, model_1, model_2, preference)
   - Support multiple preference sources (human eval, A/B tests, etc.)

2. **Embedding Alignment**:
   - Learn shared embedding space for queries and LLMs
   - Use CCFT (Category-Calibrated Fine-Tuning) or simpler contrastive learning
   - Align embeddings to reflect query-model compatibility

3. **Prior Initialization**:
   - Convert offline preferences to LinUCB priors (A matrix, b vector)
   - Initialize bandit algorithm with prior knowledge

**Interface**:
```python
class OfflinePreferenceLearner:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_ccft: bool = True,  # Use CCFT or simpler contrastive learning
    ) -> None:
        """Initialize offline preference learner."""
        pass

    async def learn_from_preferences(
        self,
        preferences: list[PreferenceExample],  # (query, winner, loser)
        models: list[str],
    ) -> PreferencePrior:
        """Learn query-model embeddings from offline preferences."""
        pass

    def initialize_bandit(
        self,
        bandit: LinUCBBandit,
        prior: PreferencePrior,
    ) -> None:
        """Initialize bandit algorithm with learned priors."""
        pass
```

### Phase 2: PILOT Algorithm

**File**: `conduit/engines/bandits/pilot.py` (new)

**Key Components**:
1. **PILOT Bandit**: Extends LinUCB with:
   - Prior initialization from offline data
   - Online cost policy for budget-aware routing
   - Reward-to-cost ratio optimization

2. **Cost Policy**:
   - Track budget utilization
   - Dynamically adjust model selection based on budget
   - Prefer high reward-to-cost ratio models when budget is tight

**Interface**:
```python
class PILOTBandit(LinUCBBandit):
    def __init__(
        self,
        arms: list[ModelArm],
        preference_prior: PreferencePrior,  # From offline learning
        budget: float | None = None,  # Optional budget constraint
        alpha: float = 1.0,
        feature_dim: int = 387,
    ) -> None:
        """Initialize PILOT with preference priors."""
        super().__init__(arms, alpha, feature_dim)
        # Initialize A and b matrices from prior
        pass

    async def select_arm(
        self,
        features: QueryFeatures,
        budget_remaining: float | None = None,
    ) -> ModelArm:
        """Select model using PILOT policy with cost awareness."""
        # Use LinUCB selection, but adjust for cost if budget constrained
        pass
```

### Phase 3: Integration with Existing Systems

**Database Schema** (add to migrations):
```sql
-- Store offline preference data
CREATE TABLE offline_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_embedding JSONB,  -- Cached embedding
    winner_model_id VARCHAR(255) NOT NULL,
    loser_model_id VARCHAR(255) NOT NULL,
    preference_score FLOAT,  -- Optional: strength of preference
    source VARCHAR(50),  -- 'human_eval', 'ab_test', 'synthetic', etc.
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store learned embeddings
CREATE TABLE model_embeddings (
    model_id VARCHAR(255) PRIMARY KEY,
    embedding JSONB NOT NULL,  -- Learned model embedding
    learned_at TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1
);
```

**CLI Command**: Add to `conduit/cli/main.py`
```python
@cli.command()
def learn_offline(
    preferences_file: str,
    output_dir: str = "models/pilot",
) -> None:
    """Learn preference priors from offline data."""
    # Load preferences, learn embeddings, save priors
    pass
```

## Integration Points

### Existing Code References

1. **LinUCB**: Extend `conduit/engines/bandits/linucb.py`
   - PILOT is essentially LinUCB with prior initialization
   - Can add PILOT as subclass or separate class using LinUCB internals

2. **Analyzer**: Use `conduit/engines/analyzer.py`
   - Reuse query embedding extraction
   - Add model embedding extraction (for CCFT)

3. **Router**: Add to `conduit/engines/router.py`
   - Support PILOT initialization from offline data
   - Add budget-aware routing option

4. **Examples**: Add to `examples/05_advanced/` (new folder)
   - `pilot_offline_learning.py` - Learn from offline preferences
   - `pilot_routing.py` - Use PILOT for routing
   - `budget_aware_routing.py` - Show budget constraints

## Acceptance Criteria

- [ ] Implement `OfflinePreferenceLearner` class
- [ ] Implement preference data loading (CSV, JSON formats)
- [ ] Implement embedding alignment (CCFT or contrastive learning)
- [ ] Implement `PILOTBandit` class extending LinUCB
- [ ] Add prior initialization from offline preferences
- [ ] Add budget-aware cost policy (optional)
- [ ] Add database schema for offline preferences and embeddings
- [ ] Add CLI command for offline learning
- [ ] Write comprehensive tests (>90% coverage)
  - Unit tests for offline learning
  - Integration tests for PILOT routing
  - Test cold start improvement vs standard LinUCB
- [ ] Add examples showing offline learning and PILOT usage
- [ ] Update `docs/BANDIT_ALGORITHMS.md` with PILOT section
- [ ] Benchmark cold start performance:
  - Queries to convergence: PILOT vs LinUCB vs Hybrid
  - Final regret after convergence
  - Generalization to unseen models

## Testing Strategy

1. **Unit Tests**: `tests/unit/test_bandits_pilot.py`
   - Test offline preference loading
   - Test embedding alignment
   - Test prior initialization
   - Test PILOT selection with priors

2. **Integration Tests**: `tests/integration/test_pilot_routing.py`
   - End-to-end: Load preferences → Learn embeddings → Initialize PILOT → Route queries
   - Compare cold start vs standard LinUCB

3. **Benchmark**: Use `conduit-bench` framework
   - Measure cold start improvement (queries to convergence)
   - Test generalization to unseen models
   - Test budget-aware routing

## Data Requirements

**Offline Preference Data Format**:
```json
[
  {
    "query": "Explain quantum physics simply",
    "winner": "gpt-4o",
    "loser": "gpt-4o-mini",
    "preference_score": 0.8,  // Optional: strength
    "source": "human_eval"
  },
  ...
]
```

**Minimum Data**:
- 100+ preference pairs for basic initialization
- 1000+ pairs for good performance
- More diverse queries → better generalization

## Performance Considerations

- **Offline Learning**: One-time cost (can be done offline)
- **Embedding Alignment**: Moderate cost (minutes to hours depending on data size)
- **Online Routing**: Same cost as LinUCB (prior initialization is free)
- **Cold Start**: Should reduce queries to convergence by 50-80%

## Architecture Decisions

1. **Embedding Method**:
   - Option A: CCFT (Category-Calibrated Fine-Tuning) - more accurate, complex
   - Option B: Simple contrastive learning - simpler, faster
   - **Recommendation**: Start with Option B, upgrade to Option A if needed

2. **Prior Initialization**:
   - Option A: Initialize A and b matrices directly from preferences
   - Option B: Learn preference function, then convert to LinUCB format
   - **Recommendation**: Option A is simpler, Option B more flexible

3. **Budget Policy**:
   - Option A: Hard budget constraint (reject queries if over budget)
   - Option B: Soft constraint (prefer cheaper models when budget tight)
   - **Recommendation**: Start with Option B, add Option A as enhancement

## Future Enhancements

- [ ] Active learning: Select most informative preference pairs to collect
- [ ] Multi-task learning: Learn from preferences across multiple customers
- [ ] Transfer learning: Transfer preferences from similar domains
- [ ] Synthetic preference generation: Use LLM to generate preferences for cold start

## Related Issues

- #XX (Dueling Bandits) - Can use dueling bandits for preference collection
- #XX (NeuralUCB) - Can use neural networks for preference modeling
- Hybrid routing - Can combine PILOT with hybrid routing

## References

- PILOT Paper: https://arxiv.org/abs/2508.21141
- CCFT Method: https://arxiv.org/abs/2510.00841 (Dueling Bandits paper)
- Learning to Rank: https://en.wikipedia.org/wiki/Learning_to_rank
- Contrastive Learning: https://arxiv.org/abs/2004.04906

