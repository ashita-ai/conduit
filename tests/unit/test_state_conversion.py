"""Unit tests for bandit state conversion utilities.

Tests the state conversion functions that enable hybrid routing transitions
between different bandit algorithms.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from conduit.core.state_store import (
    BanditState,
    deserialize_bandit_matrices,
    serialize_bandit_matrices,
)
from conduit.engines.bandits.state_conversion import convert_bandit_state


class TestSameAlgorithmConversion:
    """Tests for same-algorithm conversion (no-op)."""

    def test_same_algorithm_returns_original_state(self):
        """Test that converting to same algorithm returns original state."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a", "model-b"],
            arm_pulls={"model-a": 10, "model-b": 5},
            arm_successes={"model-a": 8, "model-b": 4},
            total_queries=15,
            mean_reward={"model-a": 0.8, "model-b": 0.75},
        )

        result = convert_bandit_state(state, "ucb1")

        assert result is state  # Same object


class TestUnsupportedConversion:
    """Tests for unsupported conversion paths."""

    def test_unsupported_conversion_raises_error(self):
        """Test that unsupported conversions raise ValueError."""
        state = BanditState(
            algorithm="random",  # Baseline algorithm
            arm_ids=["model-a"],
            arm_pulls={"model-a": 10},
            arm_successes={"model-a": 8},
            total_queries=10,
        )

        with pytest.raises(ValueError, match="Conversion not supported"):
            convert_bandit_state(state, "ucb1")

    def test_invalid_target_algorithm_raises_error(self):
        """Test that invalid target algorithm raises ValueError."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 10},
            arm_successes={"model-a": 8},
            total_queries=10,
            mean_reward={"model-a": 0.8},
        )

        with pytest.raises(ValueError, match="Conversion not supported"):
            convert_bandit_state(state, "invalid_algorithm")


class TestUCB1ToThompsonConversion:
    """Tests for UCB1 to Thompson Sampling conversion."""

    def test_ucb1_to_thompson_preserves_arm_ids(self):
        """Test that arm IDs are preserved during conversion."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a", "model-b", "model-c"],
            arm_pulls={"model-a": 100, "model-b": 50, "model-c": 25},
            arm_successes={"model-a": 80, "model-b": 40, "model-c": 20},
            total_queries=175,
            mean_reward={"model-a": 0.8, "model-b": 0.8, "model-c": 0.8},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert result.algorithm == "thompson_sampling"
        assert result.arm_ids == ["model-a", "model-b", "model-c"]

    def test_ucb1_to_thompson_computes_alpha_beta(self):
        """Test that alpha/beta parameters are computed correctly."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 80},
            total_queries=100,
            mean_reward={"model-a": 0.8},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        # alpha = 1 + mean_reward * pulls = 1 + 0.8 * 100 = 81
        # beta = 1 + (1 - mean_reward) * pulls = 1 + 0.2 * 100 = 21
        assert result.alpha_params is not None
        assert result.beta_params is not None
        assert abs(result.alpha_params["model-a"] - 81.0) < 0.01
        assert abs(result.beta_params["model-a"] - 21.0) < 0.01

    def test_ucb1_to_thompson_zero_pulls_uses_neutral_prior(self):
        """Test that zero-pull arms get neutral Beta(1,1) prior."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 0},
            arm_successes={"model-a": 0},
            total_queries=0,
            mean_reward={"model-a": 0.5},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert result.alpha_params["model-a"] == 1.0
        assert result.beta_params["model-a"] == 1.0

    def test_ucb1_to_thompson_preserves_total_queries(self):
        """Test that total_queries is preserved."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 40},
            total_queries=50,
            mean_reward={"model-a": 0.8},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert result.total_queries == 50


class TestThompsonToUCB1Conversion:
    """Tests for Thompson Sampling to UCB1 conversion."""

    def test_thompson_to_ucb1_extracts_mean_reward(self):
        """Test that mean_reward is computed from alpha/beta."""
        state = BanditState(
            algorithm="thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 80},
            total_queries=100,
            alpha_params={"model-a": 81.0},  # alpha / (alpha + beta) = 81/102 ≈ 0.794
            beta_params={"model-a": 21.0},
        )

        result = convert_bandit_state(state, "ucb1")

        assert result.algorithm == "ucb1"
        assert result.mean_reward is not None
        # mean_reward = 81 / (81 + 21) = 0.794...
        assert abs(result.mean_reward["model-a"] - 0.794) < 0.01

    def test_thompson_to_ucb1_reconstructs_pull_count(self):
        """Test that pull count is reconstructed from alpha+beta-2."""
        state = BanditState(
            algorithm="thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 80},
            total_queries=100,
            alpha_params={"model-a": 81.0},
            beta_params={"model-a": 21.0},
        )

        result = convert_bandit_state(state, "ucb1")

        # pulls = (alpha + beta) - 2 = 102 - 2 = 100
        assert result.arm_pulls["model-a"] == 100


class TestRoundTripConversions:
    """Tests for round-trip conversion accuracy."""

    def test_ucb1_thompson_roundtrip_preserves_mean(self):
        """Test UCB1 -> Thompson -> UCB1 preserves mean reward.

        Note: The roundtrip may have small numerical differences due to:
        - Beta distribution parameter reconstruction from mean
        - Integer conversion of pull counts
        """
        original = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a", "model-b"],
            arm_pulls={"model-a": 100, "model-b": 50},
            arm_successes={"model-a": 80, "model-b": 40},
            total_queries=150,
            mean_reward={"model-a": 0.8, "model-b": 0.8},
        )

        # Convert UCB1 -> Thompson -> UCB1
        thompson = convert_bandit_state(original, "thompson_sampling")
        roundtrip = convert_bandit_state(thompson, "ucb1")

        assert roundtrip.algorithm == "ucb1"
        assert roundtrip.mean_reward is not None
        # Mean should be preserved within numerical tolerance
        # The formula is: mean = alpha / (alpha + beta)
        # With alpha = 1 + 0.8*100 = 81, beta = 1 + 0.2*100 = 21
        # Reconstructed mean = 81 / 102 ≈ 0.794
        assert abs(roundtrip.mean_reward["model-a"] - 0.794) < 0.01
        assert abs(roundtrip.mean_reward["model-b"] - 0.794) < 0.01


class TestLinUCBToContextualThompsonConversion:
    """Tests for LinUCB to Contextual Thompson Sampling conversion."""

    def test_linucb_to_cts_creates_mu_and_sigma(self):
        """Test that conversion creates proper mu vectors and sigma matrices."""
        feature_dim = 5
        arm_ids = ["model-a"]

        # Create LinUCB state with A and b matrices
        A_matrices_dict = {"model-a": np.identity(feature_dim)}
        b_vectors_dict = {"model-a": np.ones((feature_dim, 1)) * 0.5}
        A_matrices, b_vectors = serialize_bandit_matrices(
            A_matrices_dict, b_vectors_dict
        )

        state = BanditState(
            algorithm="linucb",
            arm_ids=arm_ids,
            arm_pulls={"model-a": 10},
            arm_successes={"model-a": 8},
            total_queries=10,
            A_matrices=A_matrices,
            b_vectors=b_vectors,
            feature_dim=feature_dim,
        )

        result = convert_bandit_state(
            state, "contextual_thompson_sampling", feature_dim
        )

        assert result.algorithm == "contextual_thompson_sampling"
        assert result.mu_vectors is not None
        assert result.sigma_matrices is not None
        assert "model-a" in result.mu_vectors
        assert "model-a" in result.sigma_matrices

        # mu = A^-1 @ b = I @ [0.5, 0.5, ...] = [0.5, 0.5, ...]
        mu = np.array(result.mu_vectors["model-a"])
        assert len(mu) == feature_dim
        assert np.allclose(mu, 0.5, atol=0.01)

    def test_linucb_to_cts_sigma_is_a_inverse(self):
        """Test that Sigma = A^-1."""
        feature_dim = 3
        A_matrices_dict = {"model-a": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])}
        b_vectors_dict = {"model-a": np.array([[1], [1], [1]])}
        A_matrices, b_vectors = serialize_bandit_matrices(
            A_matrices_dict, b_vectors_dict
        )

        state = BanditState(
            algorithm="linucb",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 10},
            arm_successes={"model-a": 8},
            total_queries=10,
            A_matrices=A_matrices,
            b_vectors=b_vectors,
            feature_dim=feature_dim,
        )

        result = convert_bandit_state(
            state, "contextual_thompson_sampling", feature_dim
        )

        # Sigma should be A^-1 = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        sigma = np.array(result.sigma_matrices["model-a"])
        expected_sigma = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        assert np.allclose(sigma, expected_sigma, atol=0.01)


class TestContextualThompsonToLinUCBConversion:
    """Tests for Contextual Thompson to LinUCB conversion."""

    def test_cts_to_linucb_creates_a_and_b(self):
        """Test that conversion creates A matrices and b vectors."""
        feature_dim = 4
        mu_vectors = {"model-a": [0.5] * feature_dim}
        sigma_matrices = {"model-a": np.identity(feature_dim).tolist()}

        state = BanditState(
            algorithm="contextual_thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 10},
            arm_successes={"model-a": 8},
            total_queries=10,
            mu_vectors=mu_vectors,
            sigma_matrices=sigma_matrices,
            feature_dim=feature_dim,
        )

        result = convert_bandit_state(state, "linucb", feature_dim)

        assert result.algorithm == "linucb"
        assert result.A_matrices is not None
        assert result.b_vectors is not None

    def test_cts_linucb_roundtrip_preserves_theta(self):
        """Test CTS -> LinUCB -> CTS preserves parameter estimates."""
        feature_dim = 3
        original_mu = [0.3, 0.5, 0.7]
        original_sigma = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        original = BanditState(
            algorithm="contextual_thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 20},
            arm_successes={"model-a": 16},
            total_queries=20,
            mu_vectors={"model-a": original_mu},
            sigma_matrices={"model-a": original_sigma},
            feature_dim=feature_dim,
        )

        # CTS -> LinUCB -> CTS
        linucb = convert_bandit_state(original, "linucb", feature_dim)
        roundtrip = convert_bandit_state(
            linucb, "contextual_thompson_sampling", feature_dim
        )

        # mu should be preserved
        roundtrip_mu = np.array(roundtrip.mu_vectors["model-a"])
        assert np.allclose(roundtrip_mu, original_mu, atol=0.01)


class TestCrossCategoryConversions:
    """Tests for conversions between non-contextual and contextual algorithms."""

    def test_ucb1_to_linucb_initializes_with_warm_start(self):
        """Test UCB1 -> LinUCB initializes b with mean_reward."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 40},
            total_queries=50,
            mean_reward={"model-a": 0.8},
        )

        feature_dim = 10
        result = convert_bandit_state(state, "linucb", feature_dim)

        assert result.algorithm == "linucb"
        assert result.feature_dim == feature_dim

        # A should be identity
        A_matrices, b_vectors = deserialize_bandit_matrices(
            result.A_matrices, result.b_vectors
        )
        A = A_matrices["model-a"]
        assert A.shape == (feature_dim, feature_dim)
        assert np.allclose(A, np.identity(feature_dim))

        # b[0] should have warm start from mean_reward
        b = b_vectors["model-a"]
        assert b.shape == (feature_dim, 1)
        # b[0] = mean_reward * scaling_factor
        # With 50 pulls out of 50 total, proportion = 1.0, scaling = min(10, 1*20) = 10
        # b[0] = 0.8 * 10 = 8.0
        assert b[0, 0] > 0  # Warm start applied

    def test_thompson_to_contextual_thompson_initializes_mu(self):
        """Test Thompson -> CTS initializes mu with mean from alpha/beta."""
        state = BanditState(
            algorithm="thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 80},
            total_queries=100,
            alpha_params={"model-a": 81.0},
            beta_params={"model-a": 21.0},
        )

        feature_dim = 8
        result = convert_bandit_state(
            state, "contextual_thompson_sampling", feature_dim
        )

        assert result.algorithm == "contextual_thompson_sampling"
        assert result.feature_dim == feature_dim

        # mu should have warm start in first dimension
        mu = np.array(result.mu_vectors["model-a"])
        assert len(mu) == feature_dim
        assert mu[0] > 0  # Warm start applied

        # Sigma should be identity
        sigma = np.array(result.sigma_matrices["model-a"])
        assert np.allclose(sigma, np.identity(feature_dim))

    def test_linucb_to_ucb1_extracts_mean_from_theta(self):
        """Test LinUCB -> UCB1 extracts mean_reward from theta."""
        feature_dim = 5
        # Set theta such that theta[0] = 0.7
        A = np.identity(feature_dim)
        b = np.zeros((feature_dim, 1))
        b[0] = 0.7  # theta = A^-1 @ b = I @ b = b, so theta[0] = 0.7

        A_matrices, b_vectors = serialize_bandit_matrices(
            {"model-a": A}, {"model-a": b}
        )

        state = BanditState(
            algorithm="linucb",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 35},
            total_queries=50,
            A_matrices=A_matrices,
            b_vectors=b_vectors,
            feature_dim=feature_dim,
        )

        result = convert_bandit_state(state, "ucb1", feature_dim)

        assert result.algorithm == "ucb1"
        assert result.mean_reward is not None
        # mean_reward clipped to [0, 1]
        assert result.mean_reward["model-a"] == 0.7

    def test_contextual_thompson_to_thompson_extracts_mean(self):
        """Test CTS -> Thompson extracts mean and creates alpha/beta."""
        feature_dim = 4
        mu = [0.75, 0.0, 0.0, 0.0]  # First dimension is 0.75
        sigma = np.identity(feature_dim).tolist()

        state = BanditState(
            algorithm="contextual_thompson_sampling",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 75},
            total_queries=100,
            mu_vectors={"model-a": mu},
            sigma_matrices={"model-a": sigma},
            feature_dim=feature_dim,
        )

        result = convert_bandit_state(state, "thompson_sampling", feature_dim)

        assert result.algorithm == "thompson_sampling"
        assert result.alpha_params is not None
        assert result.beta_params is not None

        # mean = alpha / (alpha + beta) should be approximately 0.75
        alpha = result.alpha_params["model-a"]
        beta = result.beta_params["model-a"]
        computed_mean = alpha / (alpha + beta)
        assert abs(computed_mean - 0.75) < 0.01


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_arm_conversion(self):
        """Test conversion with single arm."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["only-model"],
            arm_pulls={"only-model": 1000},
            arm_successes={"only-model": 900},
            total_queries=1000,
            mean_reward={"only-model": 0.9},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert len(result.arm_ids) == 1
        assert "only-model" in result.alpha_params

    def test_many_arms_conversion(self):
        """Test conversion with many arms."""
        arm_ids = [f"model-{i}" for i in range(20)]
        arm_pulls = dict.fromkeys(arm_ids, 10)
        arm_successes = dict.fromkeys(arm_ids, 8)
        mean_reward = dict.fromkeys(arm_ids, 0.8)

        state = BanditState(
            algorithm="ucb1",
            arm_ids=arm_ids,
            arm_pulls=arm_pulls,
            arm_successes=arm_successes,
            total_queries=200,
            mean_reward=mean_reward,
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert len(result.arm_ids) == 20
        assert all(arm in result.alpha_params for arm in arm_ids)

    def test_extreme_mean_rewards(self):
        """Test conversion with extreme mean reward values."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["perfect", "terrible"],
            arm_pulls={"perfect": 100, "terrible": 100},
            arm_successes={"perfect": 100, "terrible": 0},
            total_queries=200,
            mean_reward={"perfect": 1.0, "terrible": 0.0},
        )

        result = convert_bandit_state(state, "thompson_sampling")

        # Perfect arm: alpha = 1 + 1.0*100 = 101, beta = 1 + 0*100 = 1
        assert result.alpha_params["perfect"] == 101.0
        assert result.beta_params["perfect"] == 1.0

        # Terrible arm: alpha = 1 + 0*100 = 1, beta = 1 + 1.0*100 = 101
        assert result.alpha_params["terrible"] == 1.0
        assert result.beta_params["terrible"] == 101.0

    def test_preserves_window_size(self):
        """Test that window_size is preserved during conversion."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 40},
            total_queries=50,
            mean_reward={"model-a": 0.8},
            window_size=500,
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert result.window_size == 500

    def test_preserves_updated_at(self):
        """Test that updated_at timestamp is preserved."""
        timestamp = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 40},
            total_queries=50,
            mean_reward={"model-a": 0.8},
            updated_at=timestamp,
        )

        result = convert_bandit_state(state, "thompson_sampling")

        assert result.updated_at == timestamp


class TestNumericalStability:
    """Tests for numerical stability with edge case inputs."""

    def test_near_singular_matrix_conversion(self):
        """Test LinUCB conversion with near-singular A matrix."""
        feature_dim = 3
        # Create A that is positive definite but has varying eigenvalues
        A = np.array([[10.0, 0, 0], [0, 1.0, 0], [0, 0, 0.1]])
        b = np.array([[5.0], [0.5], [0.05]])

        A_matrices, b_vectors = serialize_bandit_matrices(
            {"model-a": A}, {"model-a": b}
        )

        state = BanditState(
            algorithm="linucb",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 100},
            arm_successes={"model-a": 80},
            total_queries=100,
            A_matrices=A_matrices,
            b_vectors=b_vectors,
            feature_dim=feature_dim,
        )

        # Should not raise numerical errors
        result = convert_bandit_state(
            state, "contextual_thompson_sampling", feature_dim
        )

        assert result.algorithm == "contextual_thompson_sampling"
        # Verify mu is computed correctly: mu = A^-1 @ b
        mu = np.array(result.mu_vectors["model-a"])
        expected_mu = np.linalg.inv(A) @ b
        assert np.allclose(mu.reshape(-1, 1), expected_mu, atol=0.01)

    def test_large_feature_dimension(self):
        """Test conversion with large feature dimension."""
        feature_dim = 100
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 50},
            arm_successes={"model-a": 40},
            total_queries=50,
            mean_reward={"model-a": 0.8},
        )

        result = convert_bandit_state(state, "linucb", feature_dim)

        assert result.feature_dim == feature_dim

        A_matrices, b_vectors = deserialize_bandit_matrices(
            result.A_matrices, result.b_vectors
        )
        A = A_matrices["model-a"]
        b = b_vectors["model-a"]

        assert A.shape == (feature_dim, feature_dim)
        assert b.shape == (feature_dim, 1)
