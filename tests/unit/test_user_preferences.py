"""Tests for user preferences and reward weight customization."""

import pytest
from pathlib import Path

from conduit.core.config import load_preference_weights
from conduit.core.models import Query, UserPreferences


def test_user_preferences_defaults():
    """Test UserPreferences model with default value."""
    prefs = UserPreferences()
    assert prefs.optimize_for == "balanced"


def test_user_preferences_all_presets():
    """Test all 4 preset options are valid."""
    presets = ["balanced", "quality", "cost", "speed"]

    for preset in presets:
        prefs = UserPreferences(optimize_for=preset)  # type: ignore[arg-type]
        assert prefs.optimize_for == preset


def test_query_with_preferences():
    """Test Query model includes preferences field."""
    query = Query(text="Test query", preferences=UserPreferences(optimize_for="cost"))  # type: ignore[arg-type]

    assert query.preferences.optimize_for == "cost"
    assert query.text == "Test query"


def test_query_default_preferences():
    """Test Query uses default balanced preferences if not specified."""
    query = Query(text="Test query")

    assert query.preferences.optimize_for == "balanced"


def test_load_preference_weights_balanced():
    """Test loading balanced preset weights."""
    weights = load_preference_weights("balanced")

    assert weights["quality"] == 0.7
    assert weights["cost"] == 0.2
    assert weights["latency"] == 0.1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_load_preference_weights_quality():
    """Test loading quality preset weights."""
    weights = load_preference_weights("quality")

    assert weights["quality"] == 0.8
    assert weights["cost"] == 0.1
    assert weights["latency"] == 0.1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_load_preference_weights_cost():
    """Test loading cost preset weights."""
    weights = load_preference_weights("cost")

    assert weights["quality"] == 0.4
    assert weights["cost"] == 0.5
    assert weights["latency"] == 0.1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_load_preference_weights_speed():
    """Test loading speed preset weights."""
    weights = load_preference_weights("speed")

    assert weights["quality"] == 0.4
    assert weights["cost"] == 0.1
    assert weights["latency"] == 0.5
    assert sum(weights.values()) == pytest.approx(1.0)


def test_load_preference_weights_fallback_no_config(tmp_path, monkeypatch):
    """Test fallback to defaults when conduit.yaml doesn't exist."""
    # Change to temp directory without conduit.yaml
    monkeypatch.chdir(tmp_path)

    weights = load_preference_weights("balanced")

    assert weights["quality"] == 0.7
    assert weights["cost"] == 0.2
    assert weights["latency"] == 0.1


def test_load_preference_weights_fallback_invalid_yaml(tmp_path, monkeypatch):
    """Test fallback to defaults when YAML is invalid."""
    # Create invalid YAML file
    config_path = tmp_path / "conduit.yaml"
    config_path.write_text("invalid: yaml: content: [")

    monkeypatch.chdir(tmp_path)

    weights = load_preference_weights("balanced")

    # Should still return defaults
    assert weights["quality"] == 0.7
    assert weights["cost"] == 0.2
    assert weights["latency"] == 0.1


def test_load_preference_weights_custom_yaml(tmp_path, monkeypatch):
    """Test loading custom weights from conduit.yaml."""
    # Create custom config
    config_path = tmp_path / "conduit.yaml"
    config_path.write_text("""
routing:
  default_optimization: balanced

  presets:
    balanced:
      quality: 0.6
      cost: 0.3
      latency: 0.1
    quality:
      quality: 0.9
      cost: 0.05
      latency: 0.05
""")

    monkeypatch.chdir(tmp_path)

    # Test balanced custom weights
    weights_balanced = load_preference_weights("balanced")
    assert weights_balanced["quality"] == 0.6
    assert weights_balanced["cost"] == 0.3
    assert weights_balanced["latency"] == 0.1

    # Test quality custom weights
    weights_quality = load_preference_weights("quality")
    assert weights_quality["quality"] == 0.9
    assert weights_quality["cost"] == 0.05
    assert weights_quality["latency"] == 0.05


@pytest.mark.asyncio
async def test_router_uses_query_preferences():
    """Test Router applies query preferences to reward weights."""
    from conduit.engines.router import Router

    # Create router
    router = Router(models=["gpt-4o-mini", "gpt-4o"], cache_enabled=False)

    # Create query with cost optimization
    query = Query(
        text="Simple query",
        preferences=UserPreferences(optimize_for="cost")  # type: ignore[arg-type]
    )

    # Route query
    decision = await router.route(query)

    # Verify bandit has cost weights applied
    assert router.hybrid_router.ucb1.reward_weights["cost"] == 0.5
    assert router.hybrid_router.ucb1.reward_weights["quality"] == 0.4
    assert router.hybrid_router.linucb.reward_weights["cost"] == 0.5
    assert router.hybrid_router.linucb.reward_weights["quality"] == 0.4

    # Cleanup
    await router.close()


@pytest.mark.asyncio
async def test_router_default_preferences_when_none():
    """Test Router uses default weights when query has default preferences."""
    from conduit.engines.router import Router

    # Create router
    router = Router(models=["gpt-4o-mini", "gpt-4o"], cache_enabled=False)

    # Create query with default preferences
    query = Query(text="Simple query")  # Uses default balanced

    # Route query
    decision = await router.route(query)

    # Verify bandit has balanced weights applied
    assert router.hybrid_router.ucb1.reward_weights["quality"] == 0.7
    assert router.hybrid_router.ucb1.reward_weights["cost"] == 0.2
    assert router.hybrid_router.linucb.reward_weights["quality"] == 0.7
    assert router.hybrid_router.linucb.reward_weights["cost"] == 0.2

    # Cleanup
    await router.close()
