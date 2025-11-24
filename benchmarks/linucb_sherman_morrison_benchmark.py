"""Benchmark LinUCB Sherman-Morrison optimization.

Measures performance improvement from caching A_inv and using incremental updates
instead of recomputing matrix inversion on every query.
"""

import asyncio
import time
import numpy as np
from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.core.models import QueryFeatures


def create_test_arms():
    """Create test model arms."""
    return [
        ModelArm(
            model_id="gpt-4o-mini",
            model_name="gpt-4o-mini",
            provider="openai",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="gpt-4o",
            model_name="gpt-4o",
            provider="openai",
            cost_per_input_token=0.0025,
            cost_per_output_token=0.010,
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-3-haiku",
            model_name="claude-3-haiku",
            provider="anthropic",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
            expected_quality=0.80,
        ),
    ]


def create_random_features(feature_dim=387):
    """Create random query features.

    Args:
        feature_dim: Target feature dimension (default 387).
                     For 387: 384 embedding + 3 metadata
                     For 67: 64 embedding + 3 metadata (PCA)
    """
    embedding_dim = feature_dim - 3  # Reserve 3 for metadata
    return QueryFeatures(
        embedding=[np.random.rand() for _ in range(embedding_dim)],
        token_count=np.random.randint(10, 500),
        complexity_score=np.random.rand(),
        domain="general",
        domain_confidence=np.random.rand(),
    )


async def benchmark_selection(num_queries=1000, feature_dim=387):
    """Benchmark arm selection performance.

    This tests the benefit of caching A_inv in select_arm().
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmarking ARM SELECTION (feature_dim={feature_dim})")
    print(f"{'=' * 80}")

    arms = create_test_arms()
    bandit = LinUCBBandit(arms, feature_dim=feature_dim, window_size=0)

    # Pre-populate with some updates to get non-trivial A matrices
    print(f"Pre-populating with 100 updates...")
    for _ in range(100):
        features = create_random_features(feature_dim)
        arm = await bandit.select_arm(features)
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.85 + np.random.rand() * 0.1,
            latency=1.0,
        )
        await bandit.update(feedback, features)

    # Benchmark selection
    print(f"Running {num_queries} arm selections...")
    start = time.perf_counter()
    for _ in range(num_queries):
        features = create_random_features(feature_dim)
        arm = await bandit.select_arm(features)
    end = time.perf_counter()

    elapsed = end - start
    qps = num_queries / elapsed
    latency_ms = (elapsed / num_queries) * 1000

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Queries per second: {qps:.2f} QPS")
    print(f"  Average latency: {latency_ms:.3f} ms")
    print(f"  Feature dimension: {feature_dim}")

    return elapsed, qps, latency_ms


async def benchmark_update(num_updates=1000, feature_dim=387):
    """Benchmark update performance with Sherman-Morrison.

    This tests the Sherman-Morrison incremental update in update().
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmarking UPDATES with Sherman-Morrison (feature_dim={feature_dim})")
    print(f"{'=' * 80}")

    arms = create_test_arms()
    bandit = LinUCBBandit(arms, feature_dim=feature_dim, window_size=0)

    print(f"Running {num_updates} updates...")
    start = time.perf_counter()
    for i in range(num_updates):
        features = create_random_features(feature_dim)
        arm = await bandit.select_arm(features)
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001 * (i % 10 + 1),
            quality_score=0.85 + np.random.rand() * 0.1,
            latency=1.0 + np.random.rand(),
        )
        await bandit.update(feedback, features)
    end = time.perf_counter()

    elapsed = end - start
    ups = num_updates / elapsed
    latency_ms = (elapsed / num_updates) * 1000

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Updates per second: {ups:.2f} UPS")
    print(f"  Average latency: {latency_ms:.3f} ms")

    # Verify correctness: Check A_inv is truly the inverse of A
    print(f"\nVerifying correctness...")
    all_correct = True
    for model_id in bandit.arms:
        product = bandit.A[model_id] @ bandit.A_inv[model_id]
        is_identity = np.allclose(product, np.identity(feature_dim), atol=1e-8)
        if not is_identity:
            all_correct = False
            max_error = np.max(np.abs(product - np.identity(feature_dim)))
            print(f"  {model_id}: FAILED (max error: {max_error:.2e})")
        else:
            print(f"  {model_id}: OK (A @ A_inv = I)")

    if all_correct:
        print(f"\n✓ All A_inv matrices are correct inverses!")
    else:
        print(f"\n✗ Some A_inv matrices have numerical errors!")

    return elapsed, ups, latency_ms


async def benchmark_comparison():
    """Compare performance across different feature dimensions."""
    print(f"\n{'#' * 80}")
    print(f"# LinUCB Sherman-Morrison Optimization Benchmark")
    print(f"{'#' * 80}")

    # Test standard features (387 dims)
    print("\n" + "=" * 80)
    print("STANDARD FEATURES (387 dimensions)")
    print("=" * 80)
    await benchmark_selection(num_queries=1000, feature_dim=387)
    await benchmark_update(num_updates=1000, feature_dim=387)

    # Test PCA features (67 dims)
    print("\n" + "=" * 80)
    print("PCA FEATURES (67 dimensions)")
    print("=" * 80)
    await benchmark_selection(num_queries=1000, feature_dim=67)
    await benchmark_update(num_updates=1000, feature_dim=67)

    # Theoretical complexity analysis
    print(f"\n{'=' * 80}")
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print(f"{'=' * 80}")

    print("\nBefore optimization (matrix inversion on every select_arm):")
    print(f"  387³ = {387**3:,} operations per query")
    print(f"  67³ = {67**3:,} operations per query (PCA)")

    print("\nAfter optimization (cached A_inv, Sherman-Morrison updates):")
    print(f"  select_arm: O(d²) ≈ {387**2:,} operations for standard features")
    print(f"  select_arm: O(d²) ≈ {67**2:,} operations for PCA features")
    print(f"  update: O(d²) ≈ {387**2:,} operations for standard features")
    print(f"  update: O(d²) ≈ {67**2:,} operations for PCA features")

    print("\nExpected speedup:")
    print(f"  Standard features: {387**3 / 387**2:.0f}x (d³/d² = d)")
    print(f"  PCA features: {67**3 / 67**2:.0f}x (d³/d² = d)")

    print(f"\n{'#' * 80}")
    print("# Benchmark Complete")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    asyncio.run(benchmark_comparison())
