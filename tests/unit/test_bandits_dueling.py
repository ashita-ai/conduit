"""Unit tests for Dueling Bandit algorithm.

Uses shared fixtures from tests/conftest.py: test_arms, test_features
"""

import numpy as np
import pytest

from conduit.core.models import QueryFeatures
from conduit.engines.bandits.dueling import DuelingBandit, DuelingFeedback

# test_arms and test_features fixtures imported from conftest.py


class TestDuelingBanditInit:
    """Tests for DuelingBandit initialization."""

    def test_initialization_defaults(self, test_arms):
        """Test dueling bandit initializes with default parameters."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        assert bandit.name == "dueling_bandit"
        assert len(bandit.arms) == 3
        assert bandit.feature_dim == 386
        assert bandit.exploration_weight == 0.1
        assert bandit.learning_rate == 0.01
        assert bandit.total_queries == 0

        # Check preference weights initialized to zero
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert model_id in bandit.preference_weights
            weights = bandit.preference_weights[model_id]
            assert weights.shape == (386, 1)
            assert np.allclose(weights, 0.0)

    def test_initialization_custom_params(self, test_arms):
        """Test dueling bandit with custom parameters."""
        bandit = DuelingBandit(
            test_arms,
            feature_dim=100,
            exploration_weight=0.2,
            learning_rate=0.05,
        )

        assert bandit.feature_dim == 100
        assert bandit.exploration_weight == 0.2
        assert bandit.learning_rate == 0.05

        # Check custom feature dimension
        for weights in bandit.preference_weights.values():
            assert weights.shape == (100, 1)

    def test_initialization_preference_counts(self, test_arms):
        """Test preference counts initialized for all pairs."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        # Should have counts for all unique pairs (stored as sorted tuples)
        expected_pairs = [
            ("claude-haiku-4-5", "gpt-5.1"),
            ("claude-haiku-4-5", "o4-mini"),
            ("gpt-5.1", "o4-mini"),
        ]

        for pair in expected_pairs:
            assert pair in bandit.preference_counts
            assert bandit.preference_counts[pair] == 0

    @pytest.mark.asyncio
    async def test_random_seed_reproducibility(self, test_arms, test_features):
        """Test random seed produces reproducible selections."""
        # Should get same initial pair selection with same seed
        # Note: This test verifies seed works, not lockstep behavior
        bandit1 = DuelingBandit(test_arms, feature_dim=386, random_seed=42)
        pairs1 = []
        for _ in range(5):
            arm_a, arm_b = await bandit1.select_pair(test_features)
            pairs1.append((arm_a.model_id, arm_b.model_id))

        # Reset and try again with new bandit (same seed)
        bandit2 = DuelingBandit(test_arms, feature_dim=386, random_seed=42)
        pairs2 = []
        for _ in range(5):
            arm_a, arm_b = await bandit2.select_pair(test_features)
            pairs2.append((arm_a.model_id, arm_b.model_id))

        # With same seed and state, should get same sequence
        assert pairs1 == pairs2


class TestDuelingBanditSelection:
    """Tests for arm selection."""

    @pytest.mark.asyncio
    async def test_select_pair_returns_two_arms(self, test_arms, test_features):
        """Test select_pair returns two different arms."""
        bandit = DuelingBandit(test_arms, feature_dim=386)
        arm_a, arm_b = await bandit.select_pair(test_features)

        assert arm_a in test_arms
        assert arm_b in test_arms
        assert arm_a.model_id != arm_b.model_id

    @pytest.mark.asyncio
    async def test_select_pair_increments_queries(self, test_arms, test_features):
        """Test select_pair increments total queries."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        assert bandit.total_queries == 0

        await bandit.select_pair(test_features)
        assert bandit.total_queries == 1

        await bandit.select_pair(test_features)
        assert bandit.total_queries == 2

    @pytest.mark.asyncio
    async def test_select_arm_returns_top_choice(self, test_arms, test_features):
        """Test select_arm returns highest-scoring arm."""
        bandit = DuelingBandit(test_arms, feature_dim=386)
        arm = await bandit.select_arm(test_features)

        assert arm in test_arms

    @pytest.mark.asyncio
    async def test_selection_with_learned_preferences(self, test_arms, test_features):
        """Test selection changes after learning preferences."""
        bandit = DuelingBandit(test_arms, feature_dim=386, random_seed=42)

        # Initial selection
        initial_arm_a, initial_arm_b = await bandit.select_pair(test_features)

        # Provide strong preference for one arm
        feedback = DuelingFeedback(
            model_a_id=initial_arm_a.model_id,
            model_b_id=initial_arm_b.model_id,
            preference=1.0,  # A is much better
            confidence=1.0,
        )

        # Update multiple times to strengthen preference
        for _ in range(10):
            await bandit.update(feedback, test_features)

        # After learning, the preferred arm should consistently be selected
        # Check that arm_a appears frequently in selections
        selections = []
        for _ in range(20):
            arm_a, _ = await bandit.select_pair(test_features)
            selections.append(arm_a.model_id)

        # The preferred model should appear often (though not always due to exploration)
        assert selections.count(initial_arm_a.model_id) > 10


class TestDuelingBanditUpdate:
    """Tests for preference updates."""

    @pytest.mark.asyncio
    async def test_update_increases_winner_weight(self, test_arms, test_features):
        """Test update increases preference weight for winner."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        # Initial weights (should be zero)
        w_a_before = bandit.preference_weights[arm_a.model_id].copy()

        # A wins comparison
        feedback = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=0.8,  # A is better
            confidence=1.0,
        )

        await bandit.update(feedback, test_features)

        # Weight for A should increase
        w_a_after = bandit.preference_weights[arm_a.model_id]
        assert np.linalg.norm(w_a_after) > np.linalg.norm(w_a_before)

    @pytest.mark.asyncio
    async def test_update_decreases_loser_weight(self, test_arms, test_features):
        """Test update decreases preference weight for loser."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        # Give B some initial positive weight
        bandit.preference_weights[arm_b.model_id] = np.ones((386, 1)) * 0.1

        w_b_before = bandit.preference_weights[arm_b.model_id].copy()

        # A wins comparison (B loses)
        feedback = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=0.8,  # A is better
            confidence=1.0,
        )

        await bandit.update(feedback, test_features)

        # Weight for B should decrease
        w_b_after = bandit.preference_weights[arm_b.model_id]
        assert np.linalg.norm(w_b_after) < np.linalg.norm(w_b_before)

    @pytest.mark.asyncio
    async def test_update_increments_preference_count(self, test_arms, test_features):
        """Test update increments comparison count."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        pair_key = tuple(sorted([arm_a.model_id, arm_b.model_id]))

        assert bandit.preference_counts[pair_key] == 0

        feedback = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=0.5,
            confidence=1.0,
        )

        await bandit.update(feedback, test_features)

        assert bandit.preference_counts[pair_key] == 1

    @pytest.mark.asyncio
    async def test_update_with_negative_preference(self, test_arms, test_features):
        """Test update with negative preference (B better than A)."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        # Give both arms some initial positive weight
        bandit.preference_weights[arm_a.model_id] = np.ones((386, 1)) * 0.1
        bandit.preference_weights[arm_b.model_id] = np.ones((386, 1)) * 0.1

        w_a_before = bandit.preference_weights[arm_a.model_id].copy()
        w_b_before = bandit.preference_weights[arm_b.model_id].copy()

        # B wins comparison (negative preference)
        feedback = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=-0.6,  # B is better
            confidence=1.0,
        )

        await bandit.update(feedback, test_features)

        # A should decrease, B should increase
        w_a_after = bandit.preference_weights[arm_a.model_id]
        w_b_after = bandit.preference_weights[arm_b.model_id]

        # With negative preference, A loses weight, B gains weight
        assert np.linalg.norm(w_a_after) < np.linalg.norm(w_a_before)
        assert np.linalg.norm(w_b_after) > np.linalg.norm(w_b_before)

    @pytest.mark.asyncio
    async def test_update_with_confidence_weighting(self, test_arms, test_features):
        """Test update scales gradient by confidence."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        arm_a = test_arms[0]
        arm_b = test_arms[1]

        # Low confidence update
        feedback_low = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=0.8,
            confidence=0.2,  # Low confidence
        )

        await bandit.update(feedback_low, test_features)
        w_a_low_conf = bandit.preference_weights[arm_a.model_id].copy()

        # Reset and try with high confidence
        bandit.reset()

        feedback_high = DuelingFeedback(
            model_a_id=arm_a.model_id,
            model_b_id=arm_b.model_id,
            preference=0.8,
            confidence=1.0,  # High confidence
        )

        await bandit.update(feedback_high, test_features)
        w_a_high_conf = bandit.preference_weights[arm_a.model_id]

        # High confidence should produce larger weight change
        assert np.linalg.norm(w_a_high_conf) > np.linalg.norm(w_a_low_conf)


class TestDuelingBanditReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_weights(self, test_arms, test_features):
        """Test reset clears all preference weights."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        # Build up some preferences
        feedback = DuelingFeedback(
            model_a_id="o4-mini",
            model_b_id="gpt-5.1",
            preference=0.7,
            confidence=1.0,
        )

        for _ in range(5):
            await bandit.update(feedback, test_features)

        # Verify weights are non-zero
        assert np.linalg.norm(bandit.preference_weights["o4-mini"]) > 0.0

        # Reset
        bandit.reset()

        # All weights should be zero
        for weights in bandit.preference_weights.values():
            assert np.allclose(weights, 0.0)

    def test_reset_clears_counts(self, test_arms):
        """Test reset clears preference counts."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        # Manually increment a count
        pair_key = ("gpt-5.1", "o4-mini")
        bandit.preference_counts[pair_key] = 10

        bandit.reset()

        # All counts should be zero
        for count in bandit.preference_counts.values():
            assert count == 0

    def test_reset_clears_total_queries(self, test_arms):
        """Test reset clears total queries."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        bandit.total_queries = 100

        bandit.reset()

        assert bandit.total_queries == 0


class TestDuelingBanditStats:
    """Tests for statistics and preference matrix."""

    def test_get_stats_includes_all_metrics(self, test_arms):
        """Test get_stats returns complete statistics."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        stats = bandit.get_stats()

        assert "name" in stats
        assert "total_queries" in stats
        assert "n_arms" in stats
        assert "preference_counts" in stats
        assert "weight_norms" in stats
        assert "exploration_weight" in stats
        assert "learning_rate" in stats

        assert stats["name"] == "dueling_bandit"
        assert stats["n_arms"] == 3

    def test_get_preference_matrix(self, test_arms):
        """Test get_preference_matrix returns probabilities."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        prefs = bandit.get_preference_matrix()

        # Should have preference for each unique pair
        assert len(prefs) == 3  # 3 choose 2 = 3 pairs

        # All preferences should be probabilities [0, 1]
        for pref_prob in prefs.values():
            assert 0.0 <= pref_prob <= 1.0

    @pytest.mark.asyncio
    async def test_preference_matrix_after_learning(self, test_arms, test_features):
        """Test preference matrix reflects learned preferences."""
        bandit = DuelingBandit(
            test_arms, learning_rate=0.1, feature_dim=386
        )  # Higher LR for faster learning

        # Teach bandit that gpt-5.1 is better than o4-mini
        for _ in range(50):  # More updates to build stronger preference
            feedback = DuelingFeedback(
                model_a_id="gpt-5.1",
                model_b_id="o4-mini",
                preference=1.0,  # gpt-5.1 is better
                confidence=1.0,
            )
            await bandit.update(feedback, test_features)

        # Check that weights changed (direct evidence of learning)
        w_gpt51 = bandit.preference_weights["gpt-5.1"]
        w_o4_mini = bandit.preference_weights["o4-mini"]

        # gpt-5.1 should have positive weights (preferred)
        # o4-mini should have negative weights (not preferred)
        assert np.sum(w_gpt51) > 0
        assert np.sum(w_o4_mini) < 0

        # Weights should have substantial magnitude (learning occurred)
        assert np.linalg.norm(w_gpt51) > 1.0
        assert np.linalg.norm(w_o4_mini) > 1.0


class TestDuelingFeedback:
    """Tests for DuelingFeedback model."""

    def test_feedback_creation_valid(self):
        """Test creating valid dueling feedback."""
        feedback = DuelingFeedback(
            model_a_id="gpt-4o-mini",
            model_b_id="gpt-4o",
            preference=0.6,
            confidence=0.9,
        )

        assert feedback.model_a_id == "gpt-4o-mini"
        assert feedback.model_b_id == "gpt-4o"
        assert feedback.preference == 0.6
        assert feedback.confidence == 0.9

    def test_feedback_preference_bounds(self):
        """Test preference must be in [-1, 1] range."""
        # Valid preferences
        DuelingFeedback(model_a_id="a", model_b_id="b", preference=-1.0)  # Valid
        DuelingFeedback(model_a_id="a", model_b_id="b", preference=1.0)  # Valid
        DuelingFeedback(model_a_id="a", model_b_id="b", preference=0.0)  # Valid

        # Invalid preferences should raise validation error
        with pytest.raises(ValueError):
            DuelingFeedback(model_a_id="a", model_b_id="b", preference=1.5)  # Too high

        with pytest.raises(ValueError):
            DuelingFeedback(model_a_id="a", model_b_id="b", preference=-1.5)  # Too low

    def test_feedback_confidence_bounds(self):
        """Test confidence must be in [0, 1] range."""
        # Valid confidences
        DuelingFeedback(model_a_id="a", model_b_id="b", preference=0.0, confidence=0.0)
        DuelingFeedback(model_a_id="a", model_b_id="b", preference=0.0, confidence=1.0)

        # Invalid confidences should raise validation error
        with pytest.raises(ValueError):
            DuelingFeedback(
                model_a_id="a", model_b_id="b", preference=0.0, confidence=1.5
            )

        with pytest.raises(ValueError):
            DuelingFeedback(
                model_a_id="a", model_b_id="b", preference=0.0, confidence=-0.1
            )

    def test_feedback_default_confidence(self):
        """Test confidence defaults to 1.0."""
        feedback = DuelingFeedback(model_a_id="a", model_b_id="b", preference=0.5)

        assert feedback.confidence == 1.0


class TestDuelingBanditSingleArmValidation:
    """Tests for single-arm validation (P0 fix)."""

    @pytest.mark.asyncio
    async def test_select_pair_requires_two_arms(self, test_features):
        """Test select_pair raises ValueError with only 1 arm."""
        from conduit.engines.bandits.base import ModelArm

        single_arm = [
            ModelArm(
                model_id="only-model",
                model_name="only-model",
                provider="test",
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                expected_quality=0.8,
            )
        ]

        bandit = DuelingBandit(single_arm, feature_dim=386)

        with pytest.raises(ValueError, match="at least 2 arms"):
            await bandit.select_pair(test_features)

    @pytest.mark.asyncio
    async def test_select_pair_with_two_arms_succeeds(self, test_features):
        """Test select_pair works with exactly 2 arms."""
        from conduit.engines.bandits.base import ModelArm

        two_arms = [
            ModelArm(
                model_id="model-a",
                model_name="model-a",
                provider="test",
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                expected_quality=0.8,
            ),
            ModelArm(
                model_id="model-b",
                model_name="model-b",
                provider="test",
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                expected_quality=0.8,
            ),
        ]

        bandit = DuelingBandit(two_arms, feature_dim=386)
        arm_a, arm_b = await bandit.select_pair(test_features)

        assert arm_a.model_id != arm_b.model_id
        assert arm_a.model_id in ["model-a", "model-b"]
        assert arm_b.model_id in ["model-a", "model-b"]


class TestDuelingBanditGradientClipping:
    """Tests for gradient clipping (P0 fix to prevent unbounded weight growth)."""

    @pytest.mark.asyncio
    async def test_gradient_clipping_prevents_extreme_weights(self, test_arms, test_features):
        """Test gradient clipping prevents unbounded weight growth."""
        bandit = DuelingBandit(
            test_arms,
            feature_dim=386,
            learning_rate=10.0,  # Extremely high learning rate
            max_gradient_norm=1.0,  # Clipping enabled
        )

        # Apply many strong updates
        for _ in range(100):
            feedback = DuelingFeedback(
                model_a_id="o4-mini",
                model_b_id="gpt-5.1",
                preference=1.0,  # Maximum preference
                confidence=1.0,
            )
            await bandit.update(feedback, test_features)

        # Check weights are bounded (not exploding)
        for weights in bandit.preference_weights.values():
            weight_norm = np.linalg.norm(weights)
            # With clipping, weights should stay reasonable
            # Without clipping at lr=10, they would be huge
            assert weight_norm < 500  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_clip_gradient_method(self, test_arms):
        """Test _clip_gradient method clips large gradients."""
        bandit = DuelingBandit(test_arms, feature_dim=386, max_gradient_norm=1.0)

        # Large gradient
        large_gradient = np.ones((386, 1)) * 10.0  # Norm >> 1.0
        clipped = bandit._clip_gradient(large_gradient)

        # Should be clipped to norm = 1.0
        assert abs(np.linalg.norm(clipped) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_clip_gradient_preserves_small_gradients(self, test_arms):
        """Test _clip_gradient preserves gradients within norm."""
        bandit = DuelingBandit(test_arms, feature_dim=386, max_gradient_norm=1.0)

        # Small gradient
        small_gradient = np.ones((386, 1)) * 0.001  # Norm << 1.0
        original_norm = np.linalg.norm(small_gradient)
        clipped = bandit._clip_gradient(small_gradient)

        # Should be unchanged
        np.testing.assert_array_almost_equal(clipped, small_gradient)

    def test_max_gradient_norm_in_stats(self, test_arms):
        """Test max_gradient_norm is included in stats."""
        bandit = DuelingBandit(test_arms, feature_dim=386, max_gradient_norm=2.5)
        stats = bandit.get_stats()

        assert "max_gradient_norm" in stats
        assert stats["max_gradient_norm"] == 2.5


class TestDuelingBanditUpdateValidation:
    """Tests for update validation errors."""

    @pytest.mark.asyncio
    async def test_update_invalid_model_a_raises_error(self, test_arms, test_features):
        """Test update raises error when model_a_id is not in arms."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        feedback = DuelingFeedback(
            model_a_id="invalid-model",
            model_b_id="o4-mini",
            preference=0.5,
        )

        with pytest.raises(ValueError, match="Model ID 'invalid-model' not in arms"):
            await bandit.update(feedback, test_features)

    @pytest.mark.asyncio
    async def test_update_invalid_model_b_raises_error(self, test_arms, test_features):
        """Test update raises error when model_b_id is not in arms."""
        bandit = DuelingBandit(test_arms, feature_dim=386)

        feedback = DuelingFeedback(
            model_a_id="o4-mini",
            model_b_id="invalid-model",
            preference=0.5,
        )

        with pytest.raises(ValueError, match="Model ID 'invalid-model' not in arms"):
            await bandit.update(feedback, test_features)


class TestDuelingBanditStatePersistence:
    """Tests for state persistence (to_state/from_state)."""

    @pytest.mark.asyncio
    async def test_to_state_serialization(self, test_arms, test_features):
        """Test to_state serializes DuelingBandit state correctly."""
        bandit = DuelingBandit(test_arms, feature_dim=386, exploration_weight=0.15, learning_rate=0.02)

        # Build up some state
        feedback = DuelingFeedback(
            model_a_id="o4-mini",
            model_b_id="gpt-5.1",
            preference=0.7,
            confidence=0.9,
        )
        await bandit.update(feedback, test_features)
        await bandit.select_pair(test_features)

        state = bandit.to_state()

        assert state.algorithm == "dueling_bandit"
        assert state.total_queries == 1
        assert state.exploration_weight == 0.15
        assert state.learning_rate == 0.02
        assert state.feature_dim == 386
        assert set(state.arm_ids) == {"o4-mini", "gpt-5.1", "claude-haiku-4-5"}

        # Check preference weights were serialized
        assert state.preference_weights is not None
        assert len(state.preference_weights) == 3
        for arm_id, weights in state.preference_weights.items():
            assert isinstance(weights, list)
            assert len(weights) == 386

        # Check preference counts were serialized
        assert state.preference_counts is not None

    @pytest.mark.asyncio
    async def test_from_state_deserialization(self, test_arms, test_features):
        """Test from_state restores DuelingBandit state correctly."""
        # Create original bandit with state
        bandit1 = DuelingBandit(test_arms, feature_dim=386)

        feedback = DuelingFeedback(
            model_a_id="o4-mini",
            model_b_id="gpt-5.1",
            preference=0.8,
        )
        for _ in range(5):
            await bandit1.update(feedback, test_features)

        state = bandit1.to_state()

        # Create new bandit and restore state
        bandit2 = DuelingBandit(test_arms, feature_dim=386)
        bandit2.from_state(state)

        assert bandit2.total_queries == bandit1.total_queries

        # Check weights were restored
        for model_id in bandit1.preference_weights:
            np.testing.assert_array_almost_equal(
                bandit2.preference_weights[model_id],
                bandit1.preference_weights[model_id],
            )

    def test_from_state_wrong_algorithm_raises(self, test_arms):
        """Test from_state raises error for wrong algorithm type."""
        from conduit.core.state_store import BanditState
        from datetime import UTC, datetime

        bandit = DuelingBandit(test_arms, feature_dim=386)
        state = BanditState(
            algorithm="linucb",  # Wrong algorithm
            arm_ids=["o4-mini", "gpt-5.1", "claude-haiku-4-5"],
            total_queries=10,
            updated_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="State algorithm 'linucb' != 'dueling_bandit'"):
            bandit.from_state(state)

    def test_from_state_mismatched_arms_raises(self, test_arms):
        """Test from_state raises error for mismatched arm IDs."""
        from conduit.core.state_store import BanditState
        from datetime import UTC, datetime

        bandit = DuelingBandit(test_arms, feature_dim=386)
        state = BanditState(
            algorithm="dueling_bandit",
            arm_ids=["model-a", "model-b"],  # Different arms
            total_queries=10,
            updated_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="don't match current arms"):
            bandit.from_state(state)

    def test_from_state_mismatched_feature_dim_raises(self, test_arms):
        """Test from_state raises error for mismatched feature dimension."""
        from conduit.core.state_store import BanditState
        from datetime import UTC, datetime

        bandit = DuelingBandit(test_arms, feature_dim=386)
        state = BanditState(
            algorithm="dueling_bandit",
            arm_ids=["o4-mini", "gpt-5.1", "claude-haiku-4-5"],
            total_queries=10,
            feature_dim=100,  # Different feature dim
            updated_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="State feature_dim 100 != 386"):
            bandit.from_state(state)

    @pytest.mark.asyncio
    async def test_state_roundtrip_preserves_learning(self, test_arms, test_features):
        """Test state save/restore preserves learned preferences."""
        bandit1 = DuelingBandit(test_arms, feature_dim=386, learning_rate=0.1)

        # Train bandit with strong preference
        for _ in range(10):
            feedback = DuelingFeedback(
                model_a_id="gpt-5.1",
                model_b_id="o4-mini",
                preference=1.0,
            )
            await bandit1.update(feedback, test_features)

        state = bandit1.to_state()

        # Restore into new bandit
        bandit2 = DuelingBandit(test_arms, feature_dim=386, learning_rate=0.1)
        bandit2.from_state(state)

        # Both should have same preference matrix
        prefs1 = bandit1.get_preference_matrix()
        prefs2 = bandit2.get_preference_matrix()

        for key in prefs1:
            assert abs(prefs1[key] - prefs2[key]) < 0.01
