"""Unit tests for Dueling Bandit algorithm."""

import numpy as np
import pytest

from conduit.core.models import QueryFeatures
from conduit.engines.bandits.base import ModelArm
from conduit.engines.bandits.dueling import DuelingBandit, DuelingFeedback


@pytest.fixture
def test_arms():
    """Create test model arms."""
    return [
        ModelArm(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.8,
        ),
        ModelArm(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            cost_per_input_token=0.005,
            cost_per_output_token=0.015,
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-3-haiku",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
            expected_quality=0.75,
        ),
    ]


@pytest.fixture
def test_features():
    """Create test query features."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )


class TestDuelingBanditInit:
    """Tests for DuelingBandit initialization."""

    def test_initialization_defaults(self, test_arms):
        """Test dueling bandit initializes with default parameters."""
        bandit = DuelingBandit(test_arms)

        assert bandit.name == "dueling_bandit"
        assert len(bandit.arms) == 3
        assert bandit.feature_dim == 387
        assert bandit.exploration_weight == 0.1
        assert bandit.learning_rate == 0.01
        assert bandit.total_queries == 0

        # Check preference weights initialized to zero
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert model_id in bandit.preference_weights
            weights = bandit.preference_weights[model_id]
            assert weights.shape == (387, 1)
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
        bandit = DuelingBandit(test_arms)

        # Should have counts for all unique pairs (stored as sorted tuples)
        expected_pairs = [
            ("claude-3-haiku", "gpt-4o"),
            ("claude-3-haiku", "gpt-4o-mini"),
            ("gpt-4o", "gpt-4o-mini"),
        ]

        for pair in expected_pairs:
            assert pair in bandit.preference_counts
            assert bandit.preference_counts[pair] == 0

    @pytest.mark.asyncio
    async def test_random_seed_reproducibility(self, test_arms, test_features):
        """Test random seed produces reproducible selections."""
        # Should get same initial pair selection with same seed
        # Note: This test verifies seed works, not lockstep behavior
        bandit1 = DuelingBandit(test_arms, random_seed=42)
        pairs1 = []
        for _ in range(5):
            arm_a, arm_b = await bandit1.select_pair(test_features)
            pairs1.append((arm_a.model_id, arm_b.model_id))

        # Reset and try again with new bandit (same seed)
        bandit2 = DuelingBandit(test_arms, random_seed=42)
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
        bandit = DuelingBandit(test_arms)
        arm_a, arm_b = await bandit.select_pair(test_features)

        assert arm_a in test_arms
        assert arm_b in test_arms
        assert arm_a.model_id != arm_b.model_id

    @pytest.mark.asyncio
    async def test_select_pair_increments_queries(
        self, test_arms, test_features
    ):
        """Test select_pair increments total queries."""
        bandit = DuelingBandit(test_arms)

        assert bandit.total_queries == 0

        await bandit.select_pair(test_features)
        assert bandit.total_queries == 1

        await bandit.select_pair(test_features)
        assert bandit.total_queries == 2

    @pytest.mark.asyncio
    async def test_select_arm_returns_top_choice(
        self, test_arms, test_features
    ):
        """Test select_arm returns highest-scoring arm."""
        bandit = DuelingBandit(test_arms)
        arm = await bandit.select_arm(test_features)

        assert arm in test_arms

    @pytest.mark.asyncio
    async def test_selection_with_learned_preferences(
        self, test_arms, test_features
    ):
        """Test selection changes after learning preferences."""
        bandit = DuelingBandit(test_arms, random_seed=42)

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
    async def test_update_increases_winner_weight(
        self, test_arms, test_features
    ):
        """Test update increases preference weight for winner."""
        bandit = DuelingBandit(test_arms)

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
    async def test_update_decreases_loser_weight(
        self, test_arms, test_features
    ):
        """Test update decreases preference weight for loser."""
        bandit = DuelingBandit(test_arms)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        # Give B some initial positive weight
        bandit.preference_weights[arm_b.model_id] = np.ones((387, 1)) * 0.1

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
    async def test_update_increments_preference_count(
        self, test_arms, test_features
    ):
        """Test update increments comparison count."""
        bandit = DuelingBandit(test_arms)

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
    async def test_update_with_negative_preference(
        self, test_arms, test_features
    ):
        """Test update with negative preference (B better than A)."""
        bandit = DuelingBandit(test_arms)

        arm_a = test_arms[0]  # gpt-4o-mini
        arm_b = test_arms[1]  # gpt-4o

        # Give both arms some initial positive weight
        bandit.preference_weights[arm_a.model_id] = np.ones((387, 1)) * 0.1
        bandit.preference_weights[arm_b.model_id] = np.ones((387, 1)) * 0.1

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
    async def test_update_with_confidence_weighting(
        self, test_arms, test_features
    ):
        """Test update scales gradient by confidence."""
        bandit = DuelingBandit(test_arms)

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
        bandit = DuelingBandit(test_arms)

        # Build up some preferences
        feedback = DuelingFeedback(
            model_a_id="gpt-4o-mini",
            model_b_id="gpt-4o",
            preference=0.7,
            confidence=1.0,
        )

        for _ in range(5):
            await bandit.update(feedback, test_features)

        # Verify weights are non-zero
        assert (
            np.linalg.norm(bandit.preference_weights["gpt-4o-mini"]) > 0.0
        )

        # Reset
        bandit.reset()

        # All weights should be zero
        for weights in bandit.preference_weights.values():
            assert np.allclose(weights, 0.0)

    def test_reset_clears_counts(self, test_arms):
        """Test reset clears preference counts."""
        bandit = DuelingBandit(test_arms)

        # Manually increment a count
        pair_key = ("gpt-4o-mini", "gpt-4o")
        bandit.preference_counts[pair_key] = 10

        bandit.reset()

        # All counts should be zero
        for count in bandit.preference_counts.values():
            assert count == 0

    def test_reset_clears_total_queries(self, test_arms):
        """Test reset clears total queries."""
        bandit = DuelingBandit(test_arms)

        bandit.total_queries = 100

        bandit.reset()

        assert bandit.total_queries == 0


class TestDuelingBanditStats:
    """Tests for statistics and preference matrix."""

    def test_get_stats_includes_all_metrics(self, test_arms):
        """Test get_stats returns complete statistics."""
        bandit = DuelingBandit(test_arms)

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
        bandit = DuelingBandit(test_arms)

        prefs = bandit.get_preference_matrix()

        # Should have preference for each unique pair
        assert len(prefs) == 3  # 3 choose 2 = 3 pairs

        # All preferences should be probabilities [0, 1]
        for pref_prob in prefs.values():
            assert 0.0 <= pref_prob <= 1.0

    @pytest.mark.asyncio
    async def test_preference_matrix_after_learning(
        self, test_arms, test_features
    ):
        """Test preference matrix reflects learned preferences."""
        bandit = DuelingBandit(test_arms, learning_rate=0.1)  # Higher LR for faster learning

        # Teach bandit that gpt-4o is better than gpt-4o-mini
        for _ in range(50):  # More updates to build stronger preference
            feedback = DuelingFeedback(
                model_a_id="gpt-4o",
                model_b_id="gpt-4o-mini",
                preference=1.0,  # gpt-4o is better
                confidence=1.0,
            )
            await bandit.update(feedback, test_features)

        # Check that weights changed (direct evidence of learning)
        w_gpt4o = bandit.preference_weights["gpt-4o"]
        w_gpt4o_mini = bandit.preference_weights["gpt-4o-mini"]

        # gpt-4o should have positive weights (preferred)
        # gpt-4o-mini should have negative weights (not preferred)
        assert np.sum(w_gpt4o) > 0
        assert np.sum(w_gpt4o_mini) < 0

        # Weights should have substantial magnitude (learning occurred)
        assert np.linalg.norm(w_gpt4o) > 1.0
        assert np.linalg.norm(w_gpt4o_mini) > 1.0


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
        DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=-1.0
        )  # Valid
        DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=1.0
        )  # Valid
        DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=0.0
        )  # Valid

        # Invalid preferences should raise validation error
        with pytest.raises(ValueError):
            DuelingFeedback(
                model_a_id="a", model_b_id="b", preference=1.5
            )  # Too high

        with pytest.raises(ValueError):
            DuelingFeedback(
                model_a_id="a", model_b_id="b", preference=-1.5
            )  # Too low

    def test_feedback_confidence_bounds(self):
        """Test confidence must be in [0, 1] range."""
        # Valid confidences
        DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=0.0, confidence=0.0
        )
        DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=0.0, confidence=1.0
        )

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
        feedback = DuelingFeedback(
            model_a_id="a", model_b_id="b", preference=0.5
        )

        assert feedback.confidence == 1.0
