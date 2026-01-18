"""Tests for conduit.yaml configuration validation.

Tests the Pydantic schema models that validate the YAML configuration file,
ensuring errors are caught early with clear messages (Issue #219).
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from conduit.core.config_schema import (
    AlgorithmsConfig,
    ArbiterConfig,
    CacheConfig,
    ConfigValidationError,
    ConduitConfig,
    EmbeddingsConfig,
    FeedbackConfig,
    HybridRoutingConfig,
    PresetWeights,
    RoutingConfig,
    validate_config_dict,
    validate_config_file,
)


class TestPresetWeights:
    """Tests for PresetWeights validation."""

    def test_valid_weights(self) -> None:
        """Valid weights that sum to 1.0 pass."""
        weights = PresetWeights(quality=0.7, cost=0.2, latency=0.1)
        assert weights.quality == 0.7
        assert weights.cost == 0.2
        assert weights.latency == 0.1

    def test_weights_must_sum_to_one(self) -> None:
        """Weights that don't sum to 1.0 fail validation."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PresetWeights(quality=0.5, cost=0.5, latency=0.5)

    def test_weights_out_of_range(self) -> None:
        """Weights outside 0-1 range fail validation."""
        with pytest.raises(ValueError):
            PresetWeights(quality=1.5, cost=0.0, latency=-0.5)


class TestRoutingConfig:
    """Tests for RoutingConfig validation."""

    def test_valid_optimization(self) -> None:
        """Valid optimization presets pass."""
        for preset in ["balanced", "quality", "cost", "speed"]:
            config = RoutingConfig(default_optimization=preset)  # type: ignore[arg-type]
            assert config.default_optimization == preset

    def test_invalid_optimization(self) -> None:
        """Invalid optimization preset fails."""
        with pytest.raises(ValueError):
            RoutingConfig(default_optimization="invalid")  # type: ignore[arg-type]


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig validation."""

    def test_valid_providers(self) -> None:
        """Valid embedding providers pass."""
        for provider in ["auto", "openai", "cohere", "fastembed"]:
            config = EmbeddingsConfig(provider=provider)
            assert config.provider == provider

    def test_invalid_provider(self) -> None:
        """Invalid embedding provider fails."""
        with pytest.raises(ValueError, match="Invalid embedding provider"):
            EmbeddingsConfig(provider="invalid_provider")

    def test_pca_components_range(self) -> None:
        """PCA components must be within valid range."""
        # Valid
        config = EmbeddingsConfig()
        config.pca.components = 64
        assert config.pca.components == 64

        # Too low
        with pytest.raises(ValueError):
            EmbeddingsConfig(pca={"components": 1})  # type: ignore[arg-type]

        # Too high
        with pytest.raises(ValueError):
            EmbeddingsConfig(pca={"components": 10000})  # type: ignore[arg-type]


class TestAlgorithmsConfig:
    """Tests for AlgorithmsConfig validation."""

    def test_default_values(self) -> None:
        """Default algorithm hyperparameters are valid."""
        config = AlgorithmsConfig()
        assert config.linucb.alpha == 1.0
        assert config.epsilon_greedy.epsilon == 0.1
        assert config.ucb1.c == 1.5

    def test_alpha_out_of_range(self) -> None:
        """LinUCB alpha outside range fails."""
        with pytest.raises(ValueError):
            AlgorithmsConfig(linucb={"alpha": 100.0})  # type: ignore[arg-type]


class TestConduitConfig:
    """Tests for full ConduitConfig validation."""

    def test_empty_config_valid(self) -> None:
        """Empty config uses all defaults."""
        config = ConduitConfig()
        assert config.routing.default_optimization == "balanced"
        assert config.embeddings.provider == "auto"

    def test_valid_priors(self) -> None:
        """Valid priors structure passes."""
        config = ConduitConfig(
            priors={
                "code": {"gpt-4o": 0.9, "claude-3": 0.85},
                "creative": {"claude-3": 0.95},
            }
        )
        assert config.priors["code"]["gpt-4o"] == 0.9

    def test_invalid_prior_context(self) -> None:
        """Invalid context name in priors fails."""
        with pytest.raises(ValueError, match="Invalid context in priors"):
            ConduitConfig(priors={"invalid_context": {"gpt-4o": 0.9}})

    def test_invalid_prior_quality_range(self) -> None:
        """Prior quality outside 0-1 range fails."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ConduitConfig(priors={"code": {"gpt-4o": 1.5}})

    def test_invalid_prior_quality_type(self) -> None:
        """Prior quality with wrong type fails."""
        with pytest.raises(Exception, match="valid number|float_parsing"):
            ConduitConfig(priors={"code": {"gpt-4o": "high"}})  # type: ignore[dict-item]


class TestValidateConfigDict:
    """Tests for validate_config_dict function."""

    def test_valid_dict(self) -> None:
        """Valid config dict passes validation."""
        config = validate_config_dict(
            {
                "routing": {"default_optimization": "cost"},
                "embeddings": {"provider": "openai"},
            }
        )
        assert config.routing.default_optimization == "cost"
        assert config.embeddings.provider == "openai"

    def test_invalid_dict(self) -> None:
        """Invalid config dict raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            validate_config_dict({"routing": {"default_optimization": "invalid"}})

    def test_empty_dict(self) -> None:
        """Empty dict uses defaults."""
        config = validate_config_dict({})
        assert config.routing.default_optimization == "balanced"


class TestValidateConfigFile:
    """Tests for validate_config_file function."""

    def test_valid_file(self) -> None:
        """Valid YAML file passes validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"routing": {"default_optimization": "quality"}}, f)
            f.flush()

            config = validate_config_file(f.name)
            assert config.routing.default_optimization == "quality"

            Path(f.name).unlink()

    def test_missing_file(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_config_file("/nonexistent/path.yaml")

    def test_invalid_yaml(self) -> None:
        """Invalid YAML syntax raises ConfigValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            f.flush()

            with pytest.raises(ConfigValidationError, match="Invalid YAML"):
                validate_config_file(f.name)

            Path(f.name).unlink()

    def test_empty_file(self) -> None:
        """Empty file uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            config = validate_config_file(f.name)
            assert config.routing.default_optimization == "balanced"

            Path(f.name).unlink()

    def test_non_dict_config(self) -> None:
        """Non-dict YAML raises ConfigValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- just\n- a\n- list")
            f.flush()

            with pytest.raises(ConfigValidationError, match="must be a YAML mapping"):
                validate_config_file(f.name)

            Path(f.name).unlink()


class TestRealConfigFile:
    """Tests against the actual conduit.yaml in the project."""

    def test_actual_conduit_yaml_is_valid(self) -> None:
        """The project's conduit.yaml passes validation."""
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "conduit.yaml"

        if config_path.exists():
            # Should not raise
            config = validate_config_file(config_path)
            assert config.routing.default_optimization in [
                "balanced",
                "quality",
                "cost",
                "speed",
            ]
        else:
            pytest.skip("conduit.yaml not found in project root")


class TestFeedbackConfig:
    """Tests for FeedbackConfig validation."""

    def test_weights_must_sum_to_one(self) -> None:
        """Feedback weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FeedbackConfig(weights={"explicit": 0.5, "implicit": 0.2})  # type: ignore[arg-type]

    def test_valid_weights(self) -> None:
        """Valid feedback weights pass."""
        config = FeedbackConfig(weights={"explicit": 0.6, "implicit": 0.4})  # type: ignore[arg-type]
        assert config.weights.explicit == 0.6
        assert config.weights.implicit == 0.4


class TestCacheConfig:
    """Tests for CacheConfig validation."""

    def test_ttl_range(self) -> None:
        """Cache TTL must be within range."""
        # Valid
        config = CacheConfig(ttl=3600)
        assert config.ttl == 3600

        # Too low
        with pytest.raises(ValueError):
            CacheConfig(ttl=10)

        # Too high
        with pytest.raises(ValueError):
            CacheConfig(ttl=10000000)


class TestHybridRoutingConfig:
    """Tests for HybridRoutingConfig validation."""

    def test_switch_threshold_range(self) -> None:
        """Switch threshold must be within range."""
        # Valid
        config = HybridRoutingConfig(switch_threshold=5000)
        assert config.switch_threshold == 5000

        # Too low
        with pytest.raises(ValueError):
            HybridRoutingConfig(switch_threshold=10)


class TestArbiterConfig:
    """Tests for ArbiterConfig validation."""

    def test_sample_rate_range(self) -> None:
        """Sample rate must be 0-1."""
        # Valid
        config = ArbiterConfig(sample_rate=0.5)
        assert config.sample_rate == 0.5

        # Invalid
        with pytest.raises(ValueError):
            ArbiterConfig(sample_rate=1.5)


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_error_attributes(self) -> None:
        """ConfigValidationError has expected attributes."""
        error = ConfigValidationError(
            message="Test error",
            field="test_field",
            value={"bad": "value"},
        )
        assert error.message == "Test error"
        assert error.field == "test_field"
        assert error.value == {"bad": "value"}
        assert str(error) == "Test error"
