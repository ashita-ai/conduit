"""Tests for configuration loaders in conduit/core/config.py.

Tests verify 3-tier fallback chain:
1. YAML config files (conduit.yaml, pricing.yaml)
2. Environment variables
3. Hardcoded defaults (embedded in loaders)
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from conduit.core.config import (
    load_algorithm_config,
    load_arbiter_config,
    load_cache_config,
    load_feature_dimensions,
    load_feedback_config,
    load_hybrid_routing_config,
    load_litellm_config,
    load_pricing_config,
    load_quality_estimation_config,
    load_routing_config,
    parse_env_value,
)


class TestParseEnvValue:
    """Test environment variable parsing."""

    def test_parse_bool_true(self):
        assert parse_env_value("true") is True
        assert parse_env_value("True") is True
        assert parse_env_value("TRUE") is True

    def test_parse_bool_false(self):
        assert parse_env_value("false") is False
        assert parse_env_value("False") is False
        assert parse_env_value("FALSE") is False

    def test_parse_int(self):
        assert parse_env_value("42") == 42
        assert parse_env_value("-10") == -10

    def test_parse_float(self):
        assert parse_env_value("3.14") == 3.14
        assert parse_env_value("0.85") == 0.85

    def test_parse_string(self):
        assert parse_env_value("hello") == "hello"
        assert parse_env_value("o4-mini") == "o4-mini"


class TestLoadRoutingConfig:
    """Test routing configuration loader."""

    def test_hardcoded_defaults(self, tmp_path, monkeypatch):
        """Test fallback to hardcoded defaults when no YAML or env."""
        monkeypatch.chdir(tmp_path)  # No conduit.yaml in temp dir
        config = load_routing_config()

        assert config["default_optimization"] == "balanced"
        assert config["presets"]["balanced"]["quality"] == 0.7
        assert config["presets"]["balanced"]["cost"] == 0.2
        assert config["presets"]["balanced"]["latency"] == 0.1

    def test_yaml_loading(self, tmp_path, monkeypatch):
        """Test loading from conduit.yaml."""
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "routing": {
                "default_optimization": "quality",
                "presets": {
                    "custom": {"quality": 0.9, "cost": 0.05, "latency": 0.05}
                },
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_routing_config()
        assert config["default_optimization"] == "quality"
        assert "custom" in config["presets"]

    def test_env_override(self, tmp_path, monkeypatch):
        """Test environment variable override."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ROUTING_DEFAULT_OPTIMIZATION", "cost")

        config = load_routing_config()
        assert config["default_optimization"] == "cost"


class TestLoadAlgorithmConfig:
    """Test algorithm configuration loader."""

    def test_linucb_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_algorithm_config("linucb")

        assert config["alpha"] == 1.0
        assert config["success_threshold"] == 0.85

    def test_epsilon_greedy_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_algorithm_config("epsilon_greedy")

        assert config["epsilon"] == 0.1
        assert config["decay"] == 1.0
        assert config["min_epsilon"] == 0.01

    def test_ucb1_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_algorithm_config("ucb1")

        assert config["c"] == 1.5

    def test_thompson_sampling_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_algorithm_config("thompson_sampling")

        assert config["lambda"] == 1.0

    def test_yaml_override(self, tmp_path, monkeypatch):
        """Test YAML overrides defaults."""
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "algorithms": {
                "linucb": {"alpha": 2.0, "success_threshold": 0.9}
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_algorithm_config("linucb")
        assert config["alpha"] == 2.0
        assert config["success_threshold"] == 0.9

    def test_env_override(self, tmp_path, monkeypatch):
        """Test environment variable override."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ALGORITHM_LINUCB_ALPHA", "3.5")

        config = load_algorithm_config("linucb")
        assert config["alpha"] == 3.5


class TestLoadFeatureDimensions:
    """Test feature dimensions configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_feature_dimensions()

        assert config["embedding_dim"] == 384
        assert config["full_dim"] == 386
        assert config["pca_dim"] == 66
        assert config["token_count_normalization"] == 1000.0

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "features": {
                "embedding_dim": 512,
                "full_dim": 515,
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_feature_dimensions()
        assert config["embedding_dim"] == 512
        assert config["full_dim"] == 515
        assert config["pca_dim"] == 66  # Not overridden, uses default


class TestLoadQualityEstimationConfig:
    """Test quality estimation configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_quality_estimation_config()

        assert config["base_quality"] == 0.9
        assert config["empty_quality"] == 0.1
        assert config["penalties"]["short_response"] == 0.15
        assert config["thresholds"]["keyword_overlap_very_low"] == 0.05
        assert config["bounds"]["min_quality"] == 0.1

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "quality_estimation": {
                "base_quality": 0.95,
                "penalties": {"short_response": 0.2},
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_quality_estimation_config()
        assert config["base_quality"] == 0.95
        assert config["penalties"]["short_response"] == 0.2
        # Verify nested merge preserved other penalties
        assert config["penalties"]["repetition"] == 0.30


class TestLoadFeedbackConfig:
    """Test feedback detection configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_feedback_config()

        assert config["retry_detection"]["similarity_threshold"] == 0.85
        assert config["latency_detection"]["high_tolerance_max"] == 10.0
        assert config["latency_detection"]["medium_tolerance_max"] == 30.0
        assert config["weights"]["explicit"] == 0.7
        assert config["rewards"]["error"] == 0.0

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "feedback": {
                "retry_detection": {"similarity_threshold": 0.9},
                "latency_detection": {"high_tolerance_max": 5.0},
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_feedback_config()
        assert config["retry_detection"]["similarity_threshold"] == 0.9
        assert config["latency_detection"]["high_tolerance_max"] == 5.0
        # Verify nested merge preserved other values
        assert config["latency_detection"]["medium_tolerance_max"] == 30.0


class TestLoadHybridRoutingConfig:
    """Test hybrid routing configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_hybrid_routing_config()

        assert config["switch_threshold"] == 2000
        assert config["ucb1_c"] == 1.5
        assert config["linucb_alpha"] == 1.0

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "hybrid_routing": {
                "switch_threshold": 5000,
                "ucb1_c": 2.0,
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_hybrid_routing_config()
        assert config["switch_threshold"] == 5000
        assert config["ucb1_c"] == 2.0


class TestLoadArbiterConfig:
    """Test arbiter configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_arbiter_config()

        assert config["sample_rate"] == 0.1
        assert config["daily_budget"] == 10.0
        assert config["model"] == "o4-mini"
        assert config["evaluators"] == ["semantic", "factuality"]

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "arbiter": {
                "sample_rate": 0.2,
                "model": "gpt-5",
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_arbiter_config()
        assert config["sample_rate"] == 0.2
        assert config["model"] == "gpt-5"


class TestLoadCacheConfig:
    """Test cache configuration loader."""

    def test_yaml_override(self, tmp_path, monkeypatch):
        """Test YAML overrides Settings class defaults."""
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "cache": {
                "enabled": False,
                "ttl": 3600,
                "circuit_breaker": {"threshold": 10},
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_cache_config()
        assert config["enabled"] is False
        assert config["ttl"] == 3600
        assert config["circuit_breaker"]["threshold"] == 10


class TestLoadLiteLLMConfig:
    """Test LiteLLM configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_litellm_config()

        mappings = config["model_mappings"]
        assert mappings["gpt-4o-mini"] == "o4-mini"
        assert mappings["claude-3-5-sonnet-20241022"] == "claude-sonnet-4.5"
        assert mappings["gemini-1.5-pro"] == "gemini-2.5-pro"

    def test_yaml_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "litellm": {
                "model_mappings": {
                    "custom-model": "conduit-model",
                }
            }
        }

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_litellm_config()
        assert config["model_mappings"]["custom-model"] == "conduit-model"
        # Verify merge preserved defaults
        assert config["model_mappings"]["gpt-4o-mini"] == "o4-mini"


class TestLoadPricingConfig:
    """Test pricing configuration loader."""

    def test_defaults(self, tmp_path, monkeypatch):
        """Test hardcoded pricing fallbacks."""
        monkeypatch.chdir(tmp_path)
        pricing = load_pricing_config()

        assert pricing["o4-mini"]["input"] == 1.10
        assert pricing["o4-mini"]["output"] == 4.40
        assert pricing["gpt-5.1"]["input"] == 2.00
        assert pricing["claude-opus-4-5-20241124"]["input"] == 5.00

    def test_yaml_loading(self, tmp_path, monkeypatch):
        """Test loading from pricing.yaml."""
        monkeypatch.chdir(tmp_path)

        yaml_content = {
            "pricing": {
                "custom-model": {"input": 0.5, "output": 1.0},
                "o4-mini": {"input": 1.5, "output": 5.0},  # Override default
            }
        }

        yaml_path = tmp_path / "pricing.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        pricing = load_pricing_config()
        assert pricing["custom-model"]["input"] == 0.5
        assert pricing["o4-mini"]["input"] == 1.5  # Overridden


class TestErrorHandling:
    """Test graceful error handling."""

    def test_invalid_yaml(self, tmp_path, monkeypatch):
        """Test graceful fallback when YAML is invalid."""
        monkeypatch.chdir(tmp_path)

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should fall back to defaults without crashing
        config = load_algorithm_config("linucb")
        assert config["alpha"] == 1.0

    def test_missing_yaml(self, tmp_path, monkeypatch):
        """Test behavior when YAML file doesn't exist."""
        monkeypatch.chdir(tmp_path)

        config = load_feature_dimensions()
        assert config["embedding_dim"] == 384

    def test_empty_yaml_section(self, tmp_path, monkeypatch):
        """Test behavior when YAML section is empty."""
        monkeypatch.chdir(tmp_path)

        yaml_content = {"algorithms": {}}

        yaml_path = tmp_path / "conduit.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_algorithm_config("linucb")
        assert config["alpha"] == 1.0  # Falls back to defaults
