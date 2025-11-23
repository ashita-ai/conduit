# Implement Contextual Dueling Bandits for LLM Routing

## Priority: High
## Labels: `enhancement`, `algorithm`, `research`, `phase-4`

## Overview

Implement **Contextual Dueling Bandits** (specifically Feel-Good Thompson Sampling for Contextual Dueling Bandits - FGTS.CDB) as an alternative to current absolute-score bandit algorithms. This approach learns from **pairwise preference feedback** (which model is better?) rather than requiring absolute quality scores, making it easier to collect feedback in production.

## Motivation

Current bandit algorithms (LinUCB, Thompson Sampling, etc.) require absolute quality scores (0.0-1.0) for each query. In practice:
- Users rarely provide explicit quality scores
- Pairwise comparisons ("Model A is better than Model B") are easier to collect
- Can leverage implicit signals (user corrections, retries, etc.) as preference indicators

**Dueling Bandits** solve this by learning from pairwise comparisons, which is more natural for LLM routing scenarios.

## Research Papers

1. **Primary Paper**: "LLM Routing with Dueling Feedback" (arxiv:2510.00841)
   - Link: https://arxiv.org/abs/2510.00841
   - Key Algorithm: Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB)
   - Includes Category-Calibrated Fine-Tuning (CCFT) for model embeddings

2. **Background**: "Dueling Bandits" (general theory)
   - Contextual dueling bandits extend standard dueling bandits with query features
   - Learn preference functions: P(model_i > model_j | query_features)

## Implementation Plan

### Phase 1: Core Dueling Bandit Algorithm

**File**: `conduit/engines/bandits/dueling_bandits.py`

**Key Components**:
1. **Preference Model**: Learn P(model_i > model_j | features) using logistic regression or neural network
2. **FGTS.CDB Algorithm**:
   - Maintain posterior distributions over preference parameters
   - Sample from posterior to select model pairs for comparison
   - Update posterior based on pairwise feedback
3. **Model Selection**: Use learned preferences to select best model for new queries

**Interface**:
```python
class DuelingBandit(BanditAlgorithm):
    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select model using dueling bandit policy."""
        pass

    async def update_pairwise(
        self,
        winner: str,
        loser: str,
        features: QueryFeatures
    ) -> None:
        """Update preference model with pairwise comparison."""
        pass

    async def update(
        self,
        feedback: BanditFeedback,
        features: QueryFeatures
    ) -> None:
        """Convert absolute feedback to pairwise (if needed)."""
        # Can convert absolute scores to pairwise by comparing to mean
        pass
```

### Phase 2: Preference Feedback Integration

**File**: `conduit/feedback/preference.py` (new)

**Features**:
- Detect pairwise preferences from implicit signals:
  - User corrections → preferred model wins
  - Retries → original model loses
  - Explicit comparisons → direct preference
- Convert absolute scores to pairwise when available
- Store preference history in database

**Database Schema** (add to migrations):
```sql
CREATE TABLE preference_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id),
    winner_model_id VARCHAR(255) NOT NULL,
    loser_model_id VARCHAR(255) NOT NULL,
    features JSONB,
    source VARCHAR(50), -- 'explicit', 'implicit', 'converted'
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Phase 3: CCFT Model Embeddings (Advanced)

**File**: `conduit/engines/bandits/ccft.py` (new)

**Category-Calibrated Fine-Tuning**:
- Derive model embeddings from offline preference data
- Use contrastive fine-tuning with categorical weighting
- Enables better generalization to unseen models

**Note**: Can start with simpler embedding approach, add CCFT later if needed.

## Integration Points

### Existing Code References

1. **Base Class**: Extend `conduit/engines/bandits/base.py::BanditAlgorithm`
   - Follow same pattern as `LinUCBBandit` and `ContextualThompsonSamplingBandit`
   - Implement `select_arm()` and `update()` methods

2. **Feedback System**: Integrate with `conduit/feedback/`
   - Add preference detection to `conduit/feedback/detector.py`
   - Store preferences via `conduit/feedback/history.py`

3. **Router Integration**: Add to `conduit/engines/router.py`
   - Support preference-based feedback in `RoutingEngine`
   - Add `update_pairwise()` method to router interface

4. **Examples**: Add to `examples/03_optimization/`
   - `preference_feedback.py` - Show pairwise feedback collection
   - Update existing examples to show both absolute and pairwise modes

## Acceptance Criteria

- [ ] Implement `DuelingBandit` class extending `BanditAlgorithm`
- [ ] Implement FGTS.CDB algorithm with preference model
- [ ] Add `update_pairwise()` method for explicit pairwise feedback
- [ ] Convert absolute scores to pairwise comparisons when available
- [ ] Integrate with feedback system for implicit preference detection
- [ ] Add database schema for preference feedback storage
- [ ] Write comprehensive tests (>90% coverage)
  - Unit tests for preference model learning
  - Integration tests with Router
  - Test pairwise vs absolute feedback conversion
- [ ] Add example showing pairwise feedback collection
- [ ] Update `docs/BANDIT_ALGORITHMS.md` with dueling bandits section
- [ ] Benchmark against LinUCB on same dataset (regret, convergence speed)

## Testing Strategy

1. **Unit Tests**: `tests/unit/test_bandits_dueling.py`
   - Test preference model learning from pairwise comparisons
   - Test model selection using learned preferences
   - Test conversion from absolute to pairwise feedback

2. **Integration Tests**: `tests/integration/test_dueling_routing.py`
   - End-to-end routing with pairwise feedback
   - Compare performance vs LinUCB on synthetic dataset

3. **Benchmark**: Use `conduit-bench` framework
   - Compare regret curves vs LinUCB, Thompson Sampling
   - Measure convergence speed (queries to optimal routing)

## Performance Considerations

- **Preference Model**: Logistic regression is fast, neural network slower but more expressive
- **Pairwise Comparisons**: O(n²) comparisons for n models (can optimize with active learning)
- **Cold Start**: Need initial pairwise comparisons (can bootstrap from absolute scores)

## Future Enhancements

- [ ] CCFT model embeddings for better generalization
- [ ] Active learning for efficient pairwise comparison collection
- [ ] Multi-armed dueling (compare multiple models simultaneously)
- [ ] Preference aggregation from multiple users

## Related Issues

- None currently (this is the first dueling bandits implementation)

## References

- Paper: https://arxiv.org/abs/2510.00841
- Dueling Bandits Survey: https://arxiv.org/abs/1309.6869
- Contextual Bandits: https://arxiv.org/abs/1003.0146 (LinUCB paper)

