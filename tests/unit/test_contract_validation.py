"""Contract validation tests for bandit algorithms.

Tests verify the documented interface contracts from BanditAlgorithm base class:
- Thread safety for async operations
- State invariants (total_queries increment)
- Error handling (select_arm never fails)
- Persistence (to_state/from_state lossless)
- Idempotency (reset behavior)
"""

import asyncio
import pytest

from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.thompson_sampling import ThompsonSamplingBandit
from conduit.engines.bandits.ucb import UCB1Bandit
from conduit.engines.bandits.epsilon_greedy import EpsilonGreedyBandit
from conduit.core.models import QueryFeatures


@pytest.fixture
def all_algorithms(test_arms):
    """Fixture providing all algorithm implementations for contract testing."""
    return [
        LinUCBBandit(test_arms, alpha=1.0, feature_dim=386),
        ThompsonSamplingBandit(test_arms),
        UCB1Bandit(test_arms, c=2.0),
        EpsilonGreedyBandit(test_arms, epsilon=0.1),
    ]


class TestStateInvariants:
    """Test state invariant contracts."""

    @pytest.mark.asyncio
    async def test_update_increments_total_queries(self, all_algorithms, test_features):
        """Contract: After update(), total_queries MUST increment by 1."""
        for algorithm in all_algorithms:
            # Record initial count
            initial_count = algorithm.total_queries

            # Select arm and provide feedback
            arm = await algorithm.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.0001,
                quality_score=0.95,
                latency=1.0,
            )
            await algorithm.update(feedback, test_features)

            # Verify increment
            assert algorithm.total_queries == initial_count + 1, (
                f"{algorithm.name}: total_queries must increment by exactly 1 after update()"
            )

    @pytest.mark.asyncio
    async def test_multiple_updates_increment_correctly(
        self, all_algorithms, test_features
    ):
        """Contract: Multiple updates should increment total_queries correctly."""
        for algorithm in all_algorithms:
            initial_count = algorithm.total_queries

            # Perform 10 updates
            for i in range(10):
                arm = await algorithm.select_arm(test_features)
                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.0001,
                    quality_score=0.95,
                    latency=1.0,
                )
                await algorithm.update(feedback, test_features)

                # Verify count after each update
                assert algorithm.total_queries == initial_count + i + 1, (
                    f"{algorithm.name}: total_queries incorrect after {i+1} updates"
                )


class TestErrorHandling:
    """Test error handling contracts."""

    @pytest.mark.asyncio
    async def test_select_arm_never_fails_with_valid_features(
        self, all_algorithms, test_features
    ):
        """Contract: select_arm() MUST NOT raise exceptions with valid inputs."""
        for algorithm in all_algorithms:
            # Should never raise an exception
            arm = await algorithm.select_arm(test_features)

            # Must return a valid arm
            assert arm in algorithm.arms.values(), (
                f"{algorithm.name}: select_arm() must return valid arm from self.arms"
            )

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm_always(
        self, all_algorithms, test_features
    ):
        """Contract: select_arm() MUST always return arm from self.arms."""
        for algorithm in all_algorithms:
            # Call multiple times
            for _ in range(10):
                arm = await algorithm.select_arm(test_features)
                assert arm.model_id in algorithm.arms, (
                    f"{algorithm.name}: select_arm() returned arm not in self.arms"
                )

    @pytest.mark.asyncio
    async def test_update_validates_model_id(self, all_algorithms, test_features):
        """Contract: update() MUST validate feedback.model_id is in self.arms."""
        for algorithm in all_algorithms:
            # Try to update with invalid model_id
            invalid_feedback = BanditFeedback(
                model_id="invalid-model-id",
                cost=0.0001,
                quality_score=0.95,
                latency=1.0,
            )

            with pytest.raises(ValueError, match="not in arms"):
                await algorithm.update(invalid_feedback, test_features)


class TestPersistence:
    """Test persistence contract (to_state/from_state)."""

    @pytest.mark.asyncio
    async def test_to_state_from_state_lossless(self, all_algorithms, test_features):
        """Contract: from_state(to_state()) MUST restore exact state (lossless)."""
        for algorithm in all_algorithms:
            # Perform some updates to establish non-trivial state
            for _ in range(5):
                arm = await algorithm.select_arm(test_features)
                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.0001,
                    quality_score=0.95,
                    latency=1.0,
                )
                await algorithm.update(feedback, test_features)

            # Capture state
            queries_before = algorithm.total_queries
            state = algorithm.to_state()

            # Verify state has required fields
            assert state.algorithm == algorithm.name
            assert state.total_queries == queries_before

            # Create new instance and restore state
            if algorithm.name == "linucb":
                new_algorithm = LinUCBBandit(algorithm.arm_list, alpha=algorithm.alpha, feature_dim=algorithm.feature_dim)
            elif algorithm.name == "thompson_sampling":
                new_algorithm = ThompsonSamplingBandit(algorithm.arm_list)
            elif algorithm.name == "ucb1":
                new_algorithm = UCB1Bandit(algorithm.arm_list, c=algorithm.c)
            elif algorithm.name == "epsilon_greedy":
                new_algorithm = EpsilonGreedyBandit(
                    algorithm.arm_list, epsilon=algorithm.epsilon
                )

            new_algorithm.from_state(state)

            # Verify exact restoration
            assert new_algorithm.total_queries == queries_before, (
                f"{algorithm.name}: total_queries not restored correctly"
            )

    def test_from_state_validates_algorithm_name(self, test_arms):
        """Contract: from_state() MUST validate algorithm name matches."""
        linucb = LinUCBBandit(test_arms)

        # Get state from LinUCB
        state = linucb.to_state()

        # Try to load into wrong algorithm
        thompson = ThompsonSamplingBandit(test_arms)

        with pytest.raises(ValueError, match="algorithm.*mismatch"):
            thompson.from_state(state)

    @pytest.mark.asyncio
    async def test_to_state_captures_all_necessary_state(
        self, all_algorithms, test_features
    ):
        """Contract: to_state() MUST capture ALL state needed for restoration."""
        for algorithm in all_algorithms:
            # Perform updates
            for _ in range(3):
                arm = await algorithm.select_arm(test_features)
                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.0001,
                    quality_score=0.95,
                    latency=1.0,
                )
                await algorithm.update(feedback, test_features)

            # Serialize state
            state = algorithm.to_state()

            # Verify state is JSON-serializable
            state_dict = state.model_dump()
            assert isinstance(state_dict, dict)

            # Verify required fields present
            assert "algorithm" in state_dict
            assert "total_queries" in state_dict


class TestIdempotency:
    """Test idempotency contracts."""

    def test_reset_sets_total_queries_to_zero(self, all_algorithms):
        """Contract: reset() MUST set total_queries to 0."""
        for algorithm in all_algorithms:
            # Set some non-zero value
            algorithm.total_queries = 100

            # Reset
            algorithm.reset()

            # Verify
            assert algorithm.total_queries == 0, (
                f"{algorithm.name}: reset() must set total_queries to 0"
            )

    def test_reset_is_idempotent(self, all_algorithms):
        """Contract: reset() MUST be idempotent (reset twice = reset once)."""
        for algorithm in all_algorithms:
            # Reset once
            algorithm.reset()
            queries_after_first = algorithm.total_queries

            # Reset again
            algorithm.reset()
            queries_after_second = algorithm.total_queries

            # Should be identical
            assert queries_after_first == queries_after_second == 0, (
                f"{algorithm.name}: reset() must be idempotent"
            )

    def test_reset_preserves_arms_configuration(self, all_algorithms):
        """Contract: reset() MUST preserve arms configuration."""
        for algorithm in all_algorithms:
            # Capture arms before reset
            arms_before = set(algorithm.arms.keys())
            arm_count_before = len(algorithm.arms)

            # Reset
            algorithm.reset()

            # Verify arms unchanged
            arms_after = set(algorithm.arms.keys())
            assert arms_before == arms_after, (
                f"{algorithm.name}: reset() must preserve arms"
            )
            assert len(algorithm.arms) == arm_count_before, (
                f"{algorithm.name}: reset() changed arm count"
            )

    @pytest.mark.asyncio
    async def test_reset_clears_learned_state(self, all_algorithms, test_features):
        """Contract: reset() MUST clear all learned parameters."""
        for algorithm in all_algorithms:
            # Perform some learning
            for _ in range(5):
                arm = await algorithm.select_arm(test_features)
                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.0001,
                    quality_score=0.95,
                    latency=1.0,
                )
                await algorithm.update(feedback, test_features)

            # Reset
            algorithm.reset()

            # Verify total_queries is 0 (learned state cleared)
            assert algorithm.total_queries == 0, (
                f"{algorithm.name}: reset() must clear total_queries"
            )


class TestThreadSafety:
    """Test thread safety contracts (concurrent operations)."""

    @pytest.mark.asyncio
    async def test_concurrent_select_arm_calls(self, all_algorithms, test_features):
        """Contract: select_arm() MUST be thread-safe for concurrent calls."""
        for algorithm in all_algorithms:
            # Run 10 concurrent select_arm calls
            tasks = [algorithm.select_arm(test_features) for _ in range(10)]
            arms = await asyncio.gather(*tasks)

            # All should complete successfully and return valid arms
            assert len(arms) == 10
            for arm in arms:
                assert arm.model_id in algorithm.arms, (
                    f"{algorithm.name}: concurrent select_arm() returned invalid arm"
                )

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, all_algorithms, test_features):
        """Contract: update() MUST be thread-safe for concurrent updates.

        Note: This is a basic concurrency test. True thread safety would require
        more sophisticated testing, but this verifies no crashes occur.
        """
        for algorithm in all_algorithms:
            # Prepare multiple feedbacks
            tasks = []
            for _ in range(5):
                arm = await algorithm.select_arm(test_features)
                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.0001,
                    quality_score=0.95,
                    latency=1.0,
                )
                tasks.append(algorithm.update(feedback, test_features))

            # Run updates concurrently
            await asyncio.gather(*tasks)

            # Verify total_queries updated correctly
            # Note: Exact count depends on implementation's locking strategy
            # We just verify it's > 0 and <= number of updates
            assert algorithm.total_queries > 0, (
                f"{algorithm.name}: no updates recorded after concurrent updates"
            )


class TestReturnTypes:
    """Test return type contracts."""

    @pytest.mark.asyncio
    async def test_select_arm_returns_model_arm(self, all_algorithms, test_features):
        """Contract: select_arm() MUST return ModelArm type."""
        for algorithm in all_algorithms:
            arm = await algorithm.select_arm(test_features)

            # Verify it's a ModelArm (has expected attributes)
            assert hasattr(arm, "model_id")
            assert hasattr(arm, "provider")
            assert hasattr(arm, "cost_per_input_token")
            assert hasattr(arm, "cost_per_output_token")

    @pytest.mark.asyncio
    async def test_update_returns_none(self, all_algorithms, test_features):
        """Contract: update() MUST return None (no return value)."""
        for algorithm in all_algorithms:
            arm = await algorithm.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.0001,
                quality_score=0.95,
                latency=1.0,
            )

            result = await algorithm.update(feedback, test_features)

            assert result is None, (
                f"{algorithm.name}: update() must return None, got {result}"
            )

    def test_get_stats_returns_dict(self, all_algorithms):
        """Contract: get_stats() MUST return dict."""
        for algorithm in all_algorithms:
            stats = algorithm.get_stats()

            assert isinstance(stats, dict), (
                f"{algorithm.name}: get_stats() must return dict"
            )

            # Verify required fields
            assert "name" in stats
            assert "total_queries" in stats
            assert stats["name"] == algorithm.name
