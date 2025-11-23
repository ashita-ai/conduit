# Implement NeuralUCB for Non-Linear Reward Modeling

## Priority: Medium
## Labels: `enhancement`, `algorithm`, `research`, `phase-5`

## Overview

Implement **NeuralUCB** (Neural Upper Confidence Bound) as an advanced contextual bandit algorithm that uses neural networks instead of linear models to capture **non-linear relationships** between query features and model performance. This addresses limitations of linear LinUCB when feature interactions are complex.

## Motivation

Current **LinUCB** algorithm assumes linear relationships:
- Reward = θ^T @ features (linear combination)
- Works well when features are independent or have simple interactions
- **Limitation**: Cannot capture complex non-linear patterns

**NeuralUCB** uses neural networks to model:
- Complex feature interactions (e.g., "complex queries + code domain → prefer GPT-4")
- Non-linear reward surfaces
- Better accuracy when linear models plateau

**When to use**:
- After LinUCB performance plateaus
- When query features have complex interactions
- When you have sufficient data (neural networks need more samples)

## Research Papers

1. **Primary Paper**: "Neural Contextual Bandits with Deep Representation Learning" (Zhou et al., 2020)
   - Link: https://arxiv.org/abs/2012.01780
   - Key Algorithm: NeuralUCB with neural network reward model
   - Theoretical guarantees: O(√T) regret bound

2. **Alternative**: "Neural Thompson Sampling" (Zhou et al., 2020)
   - Link: https://arxiv.org/abs/1911.04462
   - Bayesian neural network approach (alternative to UCB)

3. **Recent Improvements**: "Neural Bandits with Thompson Sampling" (Riquelme et al., 2018)
   - Link: https://arxiv.org/abs/1707.02038

## Implementation Plan

### Phase 1: Core NeuralUCB Algorithm

**File**: `conduit/engines/bandits/neural_ucb.py`

**Key Components**:
1. **Neural Network Reward Model**:
   - Input: QueryFeatures (387 dims or PCA-reduced)
   - Hidden layers: 2-3 layers with ReLU activation
   - Output: Single reward prediction per model
   - Architecture: Separate network per model OR shared representation + model-specific heads

2. **UCB Calculation**:
   - Mean reward: Neural network prediction
   - Uncertainty: Use neural network gradient norm or ensemble variance
   - UCB = mean + alpha * uncertainty

3. **Training**:
   - Online learning: Update network after each feedback
   - Use Adam optimizer with learning rate decay
   - Regularization: L2 weight decay to prevent overfitting

**Interface**:
```python
class NeuralUCBBandit(BanditAlgorithm):
    def __init__(
        self,
        arms: list[ModelArm],
        hidden_dims: list[int] = [256, 128],  # Neural network architecture
        alpha: float = 1.0,  # Exploration parameter
        learning_rate: float = 0.001,
        use_pca: bool = True,  # Use PCA-reduced features
        pca_dimensions: int = 67,
    ) -> None:
        """Initialize NeuralUCB with neural network reward models."""
        pass

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select model using NeuralUCB policy."""
        pass

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update neural network with new feedback."""
        pass
```

### Phase 2: Neural Network Infrastructure

**Dependencies**: Add to `pyproject.toml`
```toml
[project.optional-dependencies]
neural = [
    "torch>=2.0.0",  # PyTorch for neural networks
    "torchvision>=0.15.0",  # Optional, for utilities
]
```

**File**: `conduit/engines/bandits/neural_models.py` (new)

**Components**:
- `RewardNetwork`: PyTorch neural network for reward prediction
- `UncertaintyEstimator`: Calculate UCB uncertainty (gradient norm, ensemble, etc.)
- Model checkpointing: Save/load trained networks

### Phase 3: Integration with Existing Systems

**PCA Integration**:
- Use PCA-reduced features (67 dims) instead of full 387 dims
- Reduces neural network size and training time
- Leverage existing `conduit/engines/analyzer.py` PCA support

**Hybrid Routing**:
- Can use NeuralUCB in hybrid router (replace LinUCB in Phase 2)
- Or use as standalone algorithm

**Caching**:
- Cache neural network predictions (same as LinUCB feature caching)
- Cache model checkpoints to disk

## Integration Points

### Existing Code References

1. **Base Class**: Extend `conduit/engines/bandits/base.py::BanditAlgorithm`
   - Follow pattern from `LinUCBBandit`
   - Reuse `_extract_features()` pattern for PCA support

2. **Analyzer**: Use `conduit/engines/analyzer.py`
   - Leverage existing PCA support (`use_pca`, `pca_dimensions`)
   - Use same feature extraction pipeline

3. **Router**: Add to `conduit/engines/router.py`
   - Add NeuralUCB as algorithm option
   - Support in hybrid router (optional)

4. **Examples**: Add to `examples/02_routing/`
   - `neural_ucb_routing.py` - Show NeuralUCB usage
   - Compare performance vs LinUCB

## Acceptance Criteria

- [ ] Implement `NeuralUCBBandit` class extending `BanditAlgorithm`
- [ ] Implement neural network reward model (PyTorch)
- [ ] Implement UCB uncertainty calculation (gradient norm or ensemble)
- [ ] Support PCA-reduced features (integrate with existing PCA)
- [ ] Online learning: Update network after each feedback
- [ ] Model checkpointing: Save/load trained networks
- [ ] Write comprehensive tests (>90% coverage)
  - Unit tests for neural network training
  - Integration tests with Router
  - Test convergence vs LinUCB
- [ ] Add example showing NeuralUCB usage
- [ ] Update `docs/BANDIT_ALGORITHMS.md` with NeuralUCB section
- [ ] Benchmark against LinUCB:
  - Convergence speed (queries to optimal)
  - Final regret (after convergence)
  - Training time per query

## Testing Strategy

1. **Unit Tests**: `tests/unit/test_bandits_neural_ucb.py`
   - Test neural network forward pass
   - Test UCB calculation
   - Test online learning updates
   - Test checkpoint save/load

2. **Integration Tests**: `tests/integration/test_neural_routing.py`
   - End-to-end routing with NeuralUCB
   - Compare performance vs LinUCB on same dataset

3. **Benchmark**: Use `conduit-bench` framework
   - Compare regret curves vs LinUCB
   - Measure training time overhead
   - Test with and without PCA

## Performance Considerations

- **Training Time**: Neural networks slower than linear models (10-100x)
  - Mitigation: Use PCA-reduced features, smaller networks, batch updates
- **Memory**: Neural networks require more memory
  - Mitigation: Use model checkpointing, clear old gradients
- **Cold Start**: Need more initial samples than linear models
  - Mitigation: Warm start with LinUCB, then switch to NeuralUCB

## Architecture Decisions

1. **Network Architecture**:
   - Option A: Separate network per model (more parameters, better accuracy)
   - Option B: Shared representation + model-specific heads (fewer parameters, faster)
   - **Recommendation**: Start with Option B, can upgrade to Option A if needed

2. **Uncertainty Estimation**:
   - Option A: Gradient norm (fast, approximate)
   - Option B: Ensemble of networks (accurate, slower)
   - Option C: Bayesian neural network (most accurate, slowest)
   - **Recommendation**: Start with Option A, add Option B as enhancement

3. **Training Strategy**:
   - Option A: Online (update after each query) - simpler, slower
   - Option B: Batch (update every N queries) - faster, more complex
   - **Recommendation**: Start with Option A, optimize to Option B if needed

## Future Enhancements

- [ ] Neural Thompson Sampling (Bayesian alternative)
- [ ] Transfer learning: Pre-train on offline data, fine-tune online
- [ ] Architecture search: Auto-tune network size/hyperparameters
- [ ] Distributed training: Scale to multiple workers

## Related Issues

- #XX (Dueling Bandits) - Can combine with NeuralUCB for neural preference models
- Hybrid routing - Can use NeuralUCB instead of LinUCB in Phase 2

## References

- NeuralUCB Paper: https://arxiv.org/abs/2012.01780
- Neural Thompson Sampling: https://arxiv.org/abs/1911.04462
- Neural Bandits Survey: https://arxiv.org/abs/1707.02038
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

