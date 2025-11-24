# LinUCB Sherman-Morrison Optimization

**Date**: 2025-11-24
**Issue**: #[issue_number]
**Implementation**: Sherman-Morrison incremental matrix update

## Problem

LinUCB computed full matrix inversion (`A_inv = np.linalg.inv(A)`) on every query in `select_arm()`:
- **Complexity**: O(d³) per query where d=387 (or 67 with PCA)
- **Operations**: ~58 million operations per query for standard features
- **Impact**: Performance bottleneck at high query volumes (>1000 QPS)

## Solution

Cache `A_inv` and update incrementally using the Sherman-Morrison formula.

### Sherman-Morrison Formula

For rank-1 update (A + xx^T)^-1:
```
(A + xx^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)
```

### Implementation Strategy

**Two modes based on `window_size`:**

1. **Non-sliding window (window_size=0)**:
   - Use Sherman-Morrison incremental update
   - Update A_inv without full matrix inversion
   - O(d²) complexity per update

2. **Sliding window (window_size>0)**:
   - Rebuild A and b from observation history when window slides
   - Recompute A_inv after rebuilding (observations drop out)
   - Still faster than per-query inversion

### Code Changes

**1. Initialize A_inv**
```python
self.A_inv = {
    arm.model_id: np.identity(feature_dim) for arm in arms
}
```

**2. Update select_arm() - use cached A_inv**
```python
# Before (O(d³)):
A_inv = np.linalg.inv(self.A[model_id])
theta = A_inv @ self.b[model_id]

# After (O(d²)):
theta = self.A_inv[model_id] @ self.b[model_id]
```

**3. Update update() - Sherman-Morrison**
```python
if self.window_size > 0:
    # Sliding window: rebuild from history
    self.A[model_id] = np.identity(self.feature_dim)
    self.b[model_id] = np.zeros((self.feature_dim, 1))
    for obs_x, obs_r in self.observation_history[model_id]:
        self.A[model_id] += obs_x @ obs_x.T
        self.b[model_id] += obs_r * obs_x
    self.A_inv[model_id] = np.linalg.inv(self.A[model_id])
else:
    # No window: incremental Sherman-Morrison
    self.A[model_id] += x @ x.T
    self.b[model_id] += reward * x
    
    # Sherman-Morrison update
    a_inv_x = self.A_inv[model_id] @ x
    denominator = 1.0 + float((x.T @ a_inv_x)[0, 0])
    
    if denominator > 1e-10:
        self.A_inv[model_id] -= (a_inv_x @ a_inv_x.T) / denominator
    else:
        # Numerical stability fallback
        self.A_inv[model_id] = np.linalg.inv(self.A[model_id])
```

**4. Update reset()**
```python
self.A_inv = {
    arm.model_id: np.identity(self.feature_dim) for arm in self.arm_list
}
```

**5. Update get_stats()**
```python
# Before:
A_inv = np.linalg.inv(self.A[model_id])
theta = A_inv @ self.b[model_id]

# After:
theta = self.A_inv[model_id] @ self.b[model_id]
```

## Performance Results

### Test Suite
- **Before**: 17.89s
- **After**: 7.62s
- **Speedup**: 2.3x

### Benchmark (1000 queries)

**Standard features (387 dimensions)**:
- Selection: 3,033 QPS (0.33ms latency)
- Updates: 1,020 UPS (0.98ms latency)

**PCA features (67 dimensions)**:
- Selection: 11,540 QPS (0.087ms latency)
- Updates: 9,797 UPS (0.102ms latency)

### Theoretical Analysis

**Before optimization**:
- select_arm: O(d³) = 387³ ≈ 58M operations
- PCA: O(d³) = 67³ ≈ 300K operations

**After optimization**:
- select_arm: O(d²) = 387² ≈ 150K operations (387x speedup)
- update: O(d²) = 387² ≈ 150K operations
- PCA select_arm: O(d²) = 67² ≈ 4.5K operations (67x speedup)

## Numerical Stability

**Challenge**: Sherman-Morrison can have numerical issues if denominator approaches zero.

**Solution**: 
1. Check denominator > 1e-10
2. Fallback to full inversion if check fails
3. In practice, denominator = 1 + x^T A_inv x is always > 1 for positive definite A

**Verification**:
- Test correctness: A @ A_inv ≈ I (identity matrix)
- All tests verify this property with tolerance 1e-10

## Testing

**New tests (6 added)**:
1. `test_sherman_morrison_incremental_update`: Verify A_inv updates correctly
2. `test_sherman_morrison_vs_direct_inversion`: Compare to direct inversion
3. `test_sliding_window_recalculates_a_inv`: Verify sliding window mode
4. `test_a_inv_initialization`: Check initial state
5. `test_a_inv_reset`: Check reset restores identity
6. `test_numerical_stability_fallback`: Verify fallback path

**Coverage**: 96% for linucb.py (18/18 tests passing)

## Edge Cases

1. **Initial state**: A_inv = I (identity), no inversion needed
2. **Sliding window full**: Rebuild A and A_inv from windowed history
3. **Numerical instability**: Automatic fallback to full inversion
4. **Multiple arms**: Each arm maintains separate A_inv

## Future Optimizations

1. **Woodbury identity** for batch removal in sliding window:
   - Currently: full recalculation when window slides
   - Future: Incremental update for batch removal
   - Would make sliding window O(d²) as well

2. **Condition number monitoring**:
   - Track matrix condition number
   - Alert on potential numerical issues
   - Periodic full inversion to prevent drift

## References

- [Sherman-Morrison Formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
- [LinUCB Paper](https://arxiv.org/abs/1003.0146) - Li et al. 2010
- [Tutorial](https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/)
- Benchmark: `benchmarks/linucb_sherman_morrison_benchmark.py`

## Files Changed

1. `conduit/engines/bandits/linucb.py` - Core implementation
2. `tests/unit/test_bandits_linucb.py` - Comprehensive tests
3. `benchmarks/linucb_sherman_morrison_benchmark.py` - Performance benchmark
4. `docs/BANDIT_ALGORITHMS.md` - Documentation update
