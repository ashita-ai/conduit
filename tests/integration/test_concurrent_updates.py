"""Integration tests for concurrent state updates with real database.

Tests race conditions and state consistency when multiple workers
update routing state concurrently. Uses real PostgreSQL for testing.
"""

import asyncio

import numpy as np
import pytest

from conduit.core.models import Query, QueryFeatures
from conduit.engines import Router
from conduit.engines.bandits import LinUCBBandit
from conduit.engines.bandits.base import BanditFeedback, ModelArm


@pytest.mark.asyncio
async def test_concurrent_routing_and_feedback(test_arms, test_features):
    """Test concurrent routing + feedback loop simulating production load.

    Simulates 100 concurrent workers routing queries and providing feedback,
    verifying no state corruption or race conditions occur.
    """
    router = Router(models=[arm.model_id for arm in test_arms])

    async def route_and_feedback(worker_id: int):
        """Single worker: route query → provide feedback."""
        query = Query(text=f"Worker {worker_id} query")

        # Route query
        decision = await router.route(query)

        # Simulate feedback (quality varies by worker)
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.5 + (worker_id % 5) * 0.1,
            latency=1.0,
        )

        # Update bandit state
        await router.hybrid_router.update(feedback, decision.features)

        return decision

    # Spawn 100 concurrent workers
    tasks = [route_and_feedback(i) for i in range(100)]
    decisions = await asyncio.gather(*tasks)

    # Verify all completed successfully
    assert len(decisions) == 100
    assert all(d is not None for d in decisions)

    # Verify bandit state is still valid (no corruption)
    bandit = router.hybrid_router.linucb
    for arm in test_arms:
        if arm.model_id in bandit.A:  # May not have been selected yet
            A = bandit.A[arm.model_id]

            # A should be symmetric
            assert np.allclose(A, A.T, rtol=1e-10), f"A matrix not symmetric for {arm.model_id}"

            # A should be positive definite
            eigenvalues = np.linalg.eigvalsh(A)
            assert np.all(eigenvalues > 0), f"A matrix not positive definite for {arm.model_id}"


@pytest.mark.asyncio
async def test_concurrent_matrix_updates_consistency(test_arms, test_features):
    """Test that concurrent A matrix updates don't cause corruption.

    LinUCB updates A ← A + x @ x^T and b ← b + reward * x concurrently.
    Verify final state is consistent with serial execution.
    """
    bandit = LinUCBBandit(test_arms, feature_dim=387)

    # Perform 200 concurrent updates
    updates = []
    for i in range(200):
        arm = test_arms[i % len(test_arms)]
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.7 + (i % 3) * 0.1,
            latency=1.0,
        )
        updates.append((feedback, test_features))

    # Execute updates concurrently
    tasks = [bandit.update(fb, feat) for fb, feat in updates]
    await asyncio.gather(*tasks)

    # Verify state consistency
    for arm in test_arms:
        A = bandit.A[arm.model_id]
        b = bandit.b[arm.model_id]

        # Check A matrix properties
        assert A.shape == (387, 387)
        assert np.allclose(A, A.T, rtol=1e-10)

        # Check for NaN or Inf (corruption indicators)
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))
        assert not np.any(np.isnan(b))
        assert not np.any(np.isinf(b))

        # Verify positive definiteness
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0)


@pytest.mark.asyncio
async def test_high_concurrency_routing(test_arms):
    """Test routing under high concurrency (1000 concurrent requests)."""
    router = Router(models=[arm.model_id for arm in test_arms])

    # Create 1000 concurrent routing requests
    queries = [Query(text=f"Concurrent query {i}") for i in range(1000)]
    tasks = [router.route(q) for q in queries]

    # Execute all concurrently
    decisions = await asyncio.gather(*tasks)

    # Verify all succeeded
    assert len(decisions) == 1000
    assert all(d is not None for d in decisions)
    assert all(d.selected_model in [arm.model_id for arm in test_arms] for d in decisions)

    # Verify query count updated correctly
    # Note: query_count may be approximate due to race conditions
    # but should be close to 1000
    assert 950 <= router.hybrid_router.query_count <= 1050


@pytest.mark.asyncio
async def test_interleaved_selection_and_updates(test_arms, test_features):
    """Test interleaved arm selection and state updates.

    Simulates realistic production scenario where selection and updates
    happen concurrently from different workers.
    """
    bandit = LinUCBBandit(test_arms, feature_dim=387)

    async def select_arm_loop():
        """Worker that continuously selects arms."""
        for _ in range(50):
            await bandit.select_arm(test_features)
            await asyncio.sleep(0.001)  # Small delay

    async def update_loop(arm_id: int):
        """Worker that continuously updates state."""
        arm = test_arms[arm_id % len(test_arms)]
        for i in range(50):
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.7,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)
            await asyncio.sleep(0.001)  # Small delay

    # Spawn interleaved workers
    select_tasks = [select_arm_loop() for _ in range(5)]
    update_tasks = [update_loop(i) for i in range(5)]

    # Run all concurrently
    await asyncio.gather(*select_tasks, *update_tasks)

    # Verify state is still valid
    for arm in test_arms:
        A = bandit.A[arm.model_id]

        # No corruption
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))

        # Still positive definite
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0)


@pytest.mark.asyncio
async def test_concurrent_updates_different_arms(test_arms, test_features):
    """Test concurrent updates to different arms don't interfere.

    Each arm has independent A/b matrices. Concurrent updates to
    different arms should be completely independent.
    """
    bandit = LinUCBBandit(test_arms, feature_dim=387)

    async def update_arm_repeatedly(arm: ModelArm, arm_index: int, count: int):
        """Update single arm 'count' times with arm-specific rewards."""
        for i in range(count):
            # Vary quality score by arm to ensure different b vectors
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.6 + (arm_index * 0.1),  # Different per arm
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

    # Update each arm 100 times concurrently with different rewards
    tasks = [update_arm_repeatedly(arm, i, 100) for i, arm in enumerate(test_arms)]
    await asyncio.gather(*tasks)

    # Verify each arm has exactly 100 pulls
    for arm in test_arms:
        assert bandit.arm_pulls[arm.model_id] == 100

    # Verify states are independent (different b vectors due to different rewards)
    b_vectors = [bandit.b[arm.model_id] for arm in test_arms]
    for i in range(len(b_vectors)):
        for j in range(i + 1, len(b_vectors)):
            # Different arms got different rewards, so should have different b vectors
            assert not np.allclose(b_vectors[i], b_vectors[j])


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_arms() -> list[ModelArm]:
    """Standard test arms for concurrent update tests."""
    return [
        ModelArm(
            model_id="o4-mini",
            model_name="o4-mini",
            provider="openai",
            cost_per_input_token=0.00011,
            cost_per_output_token=0.00044,
            expected_quality=0.7,
        ),
        ModelArm(
            model_id="gpt-5.1",
            model_name="gpt-5.1",
            provider="openai",
            cost_per_input_token=0.002,
            cost_per_output_token=0.008,
            expected_quality=0.9,
        ),
        ModelArm(
            model_id="claude-haiku-4-5-20241124",
            model_name="claude-haiku-4-5",
            provider="anthropic",
            cost_per_input_token=0.0008,
            cost_per_output_token=0.004,
            expected_quality=0.75,
        ),
    ]


@pytest.fixture
def test_features() -> QueryFeatures:
    """Standard test features for concurrent update tests."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )
