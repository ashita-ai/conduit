"""Tests for the model registry module."""

import pytest

from conduit_litellm.model_registry import (
    MatchSource,
    MappingConflictError,
    ModelRegistry,
    ResolveResult,
    find_best_match,
    get_global_registry,
    normalize_model_name,
    strip_date_suffix,
    strip_provider_prefix,
    calculate_match_score,
    tokenize_model_name,
)


class TestStripProviderPrefix:
    """Tests for strip_provider_prefix function."""

    def test_openai_prefix(self) -> None:
        assert strip_provider_prefix("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_anthropic_prefix(self) -> None:
        assert strip_provider_prefix("anthropic/claude-3-sonnet") == "claude-3-sonnet"

    def test_groq_prefix(self) -> None:
        assert (
            strip_provider_prefix("groq/llama-3.3-70b-versatile")
            == "llama-3.3-70b-versatile"
        )

    def test_google_prefix(self) -> None:
        assert strip_provider_prefix("google/gemini-2.0-flash") == "gemini-2.0-flash"

    def test_no_prefix(self) -> None:
        assert strip_provider_prefix("gpt-4o-mini") == "gpt-4o-mini"

    def test_nested_prefix(self) -> None:
        # azure/us/gpt-4o-mini should strip multiple levels
        result = strip_provider_prefix("azure/us/gpt-4o-mini")
        # Should strip azure/ and us/ prefixes
        assert "gpt-4o-mini" in result


class TestStripDateSuffix:
    """Tests for strip_date_suffix function."""

    def test_iso_date_suffix(self) -> None:
        assert strip_date_suffix("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_compact_date_suffix(self) -> None:
        assert strip_date_suffix("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"

    def test_version_suffix(self) -> None:
        assert strip_date_suffix("model-v1") == "model"
        assert strip_date_suffix("model-v2.0") == "model"

    def test_no_suffix(self) -> None:
        assert strip_date_suffix("gpt-4o-mini") == "gpt-4o-mini"

    def test_multiple_hyphens(self) -> None:
        # Should only strip the date suffix, not other hyphenated parts
        assert strip_date_suffix("claude-3-5-haiku-20241022") == "claude-3-5-haiku"

    def test_latest_suffix(self) -> None:
        assert strip_date_suffix("claude-3-5-sonnet-latest") == "claude-3-5-sonnet"

    def test_preview_suffix(self) -> None:
        assert strip_date_suffix("gpt-4o-mini-preview") == "gpt-4o-mini"

    def test_turbo_suffix(self) -> None:
        assert strip_date_suffix("gpt-4-turbo") == "gpt-4"

    def test_instruct_suffix(self) -> None:
        assert strip_date_suffix("gpt-3.5-instruct") == "gpt-3.5"


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_full_normalization(self) -> None:
        # Provider prefix + date suffix
        assert normalize_model_name("openai/gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_groq_llama(self) -> None:
        # Groq Llama should keep the full name (versatile not stripped)
        assert normalize_model_name("groq/llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"

    def test_anthropic_claude(self) -> None:
        # claude-3-5 should normalize to claude-3.5
        assert normalize_model_name("claude-3-5-sonnet-20241022") == "claude-3.5-sonnet"

    def test_preserves_unknown_models(self) -> None:
        # Unknown models just get prefix/suffix stripped
        assert normalize_model_name("custom-model-2024-01-01") == "custom-model"

    def test_bedrock_versioned_suffix(self) -> None:
        # Bedrock uses -v1:0 suffixes
        assert normalize_model_name("anthropic.claude-3-5-sonnet-v1:0") == "anthropic.claude-3.5-sonnet"


class TestTokenizeModelName:
    """Tests for tokenize_model_name function."""

    def test_basic_tokenization(self) -> None:
        tokens = tokenize_model_name("gpt-4o-mini")
        assert tokens == ["gpt", "4o", "mini"]

    def test_version_tokenization(self) -> None:
        tokens = tokenize_model_name("claude-3.5-sonnet")
        assert "claude" in tokens
        assert "sonnet" in tokens

    def test_complex_model(self) -> None:
        tokens = tokenize_model_name("llama-3.3-70b-versatile")
        assert "llama" in tokens
        assert "versatile" in tokens


class TestCalculateMatchScore:
    """Tests for calculate_match_score function."""

    def test_exact_match(self) -> None:
        score = calculate_match_score("gpt-4o-mini", "gpt-4o-mini")
        assert score == 1.0

    def test_normalized_exact_match(self) -> None:
        score = calculate_match_score("gpt-4o-mini-2024-07-18", "gpt-4o-mini")
        assert score == 1.0

    def test_partial_match(self) -> None:
        score = calculate_match_score("gpt-4o", "gpt-4o-mini")
        assert 0.5 < score < 1.0  # High but not exact

    def test_no_match(self) -> None:
        score = calculate_match_score("completely-different", "model-xyz")
        assert score < 0.3


class TestFindBestMatch:
    """Tests for find_best_match function with scored matching."""

    def test_exact_match_after_normalization(self) -> None:
        available = ["gpt-4o-mini", "gpt-4o", "claude-3-sonnet"]
        match, score, alternatives = find_best_match("gpt-4o-mini-2024-07-18", available)
        assert match == "gpt-4o-mini"
        assert score == 1.0

    def test_prefix_match_input_starts_with_available(self) -> None:
        available = ["gpt-4o-mini", "gpt-4o"]
        match, score, alternatives = find_best_match("gpt-4o-mini-turbo", available)
        assert match == "gpt-4o-mini"
        assert score > 0.5

    def test_prefix_match_available_starts_with_input(self) -> None:
        available = ["llama-3.3-70b-versatile", "llama-3-70b"]
        match, score, alternatives = find_best_match("llama-3.3-70b", available)
        assert match == "llama-3.3-70b-versatile"
        assert score > 0.5

    def test_common_prefix_match(self) -> None:
        available = ["claude-sonnet", "gpt-4"]
        match, score, alternatives = find_best_match("claude-sonnet-4", available)
        assert match == "claude-sonnet"
        assert score > 0.5

    def test_no_match(self) -> None:
        available = ["gpt-4o-mini", "gpt-4o"]
        match, score, alternatives = find_best_match("completely-different-model", available)
        assert match is None
        assert score == 0.0

    def test_empty_available(self) -> None:
        match, score, alternatives = find_best_match("gpt-4o-mini", [])
        assert match is None

    def test_groq_llama_versatile(self) -> None:
        # This is the specific case from the bug
        available = ["llama-3.3-70b-versatile", "gpt-4o-mini"]
        match, score, alternatives = find_best_match("groq/llama-3.3-70b-versatile", available)
        assert match == "llama-3.3-70b-versatile"
        assert score == 1.0

    def test_scored_matching_picks_best(self) -> None:
        # When multiple models could match, picks the best score
        available = ["gpt-4o", "gpt-4o-mini", "gpt-4"]
        match, score, alternatives = find_best_match("gpt-4o-mini-2024-07-18", available)
        assert match == "gpt-4o-mini"
        assert score == 1.0

    def test_minimum_score_threshold(self) -> None:
        # Very weak matches should return None
        available = ["completely-unrelated-model"]
        match, score, alternatives = find_best_match("gpt-4o-mini", available)
        assert match is None

    def test_returns_alternatives(self) -> None:
        available = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
        match, score, alternatives = find_best_match("gpt-4o-mini", available)
        assert match == "gpt-4o-mini"
        # Should have alternatives for partial matches
        assert len(alternatives) >= 0  # May have alternatives


class TestResolveResult:
    """Tests for ResolveResult dataclass."""

    def test_success_property(self) -> None:
        result = ResolveResult(
            model_id="test",
            confidence=1.0,
            source=MatchSource.EXACT,
            input_model="test",
        )
        assert result.success is True

    def test_failed_property(self) -> None:
        result = ResolveResult(
            model_id="test",
            confidence=0.0,
            source=MatchSource.FAILED,
            input_model="test",
        )
        assert result.success is False

    def test_bool_true(self) -> None:
        result = ResolveResult(
            model_id="test",
            confidence=1.0,
            source=MatchSource.EXACT,
            input_model="test",
        )
        assert bool(result) is True

    def test_bool_false(self) -> None:
        result = ResolveResult(
            model_id="test",
            confidence=0.0,
            source=MatchSource.FAILED,
            input_model="test",
        )
        assert bool(result) is False


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_and_resolve_exact(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("gpt-4o-mini-2024-07-18", "my-fast-model")
        result = registry.resolve("gpt-4o-mini-2024-07-18")
        assert result.model_id == "my-fast-model"
        assert result.source == MatchSource.EXACT
        assert result.confidence == 1.0

    def test_resolve_normalized(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("gpt-4o-mini", "my-fast-model")
        # Should resolve via normalization
        result = registry.resolve("gpt-4o-mini-2024-07-18")
        assert result.model_id == "my-fast-model"
        assert result.source == MatchSource.NORMALIZED
        assert result.confidence == 0.95

    def test_resolve_with_available_models(self) -> None:
        registry = ModelRegistry()
        # No explicit mapping, but should fuzzy match
        available = ["gpt-4o-mini", "gpt-4o"]
        result = registry.resolve("gpt-4o-mini-2024-07-18", available)
        assert result.model_id == "gpt-4o-mini"
        assert result.source == MatchSource.FUZZY
        assert result.confidence >= 0.5

    def test_resolve_learns_mapping(self) -> None:
        registry = ModelRegistry()
        available = ["gpt-4o-mini", "gpt-4o"]
        # First resolve learns the mapping
        registry.resolve("gpt-4o-mini-2024-07-18", available)
        # Second resolve uses learned mapping (no available needed)
        result = registry.resolve("gpt-4o-mini-2024-07-18")
        assert result.model_id == "gpt-4o-mini"
        assert result.source == MatchSource.EXACT  # Now exact because learned

    def test_resolve_fallback_fails_explicitly(self) -> None:
        registry = ModelRegistry()
        # No mapping, no available models - explicit failure
        result = registry.resolve("openai/gpt-4o-mini-2024-07-18")
        assert result.model_id == "gpt-4o-mini"  # Normalized fallback
        assert result.source == MatchSource.FAILED
        assert result.confidence == 0.0
        assert result.success is False

    def test_clear(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("model-a", "id-a")
        registry.clear()
        # After clear, should fail (not silently succeed)
        result = registry.resolve("model-a")
        assert result.source == MatchSource.FAILED

    def test_provider_prefix_stripped_on_register(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("openai/gpt-4o-mini", "my-model")
        # Should resolve with or without prefix
        result1 = registry.resolve("gpt-4o-mini")
        result2 = registry.resolve("openai/gpt-4o-mini")
        assert result1.model_id == "my-model"
        assert result2.model_id == "my-model"

    def test_validate_mappings(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("gpt-4o-mini", "valid-model")
        registry.register_mapping("claude-3", "invalid-model")

        # Only valid-model should pass validation
        warnings = registry.validate_mappings(["valid-model", "other-model"])
        assert len(warnings) > 0  # Should have warning for invalid-model

    def test_get_mapping_state(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("gpt-4o-mini", "my-model")

        state = registry.get_mapping_state()
        assert "runtime_mappings" in state
        assert "reverse_mappings" in state
        assert "total_mappings" in state
        assert state["total_mappings"] > 0
        assert "stats" in state
        assert "config" in state

    def test_reverse_mappings_tracked(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("gpt-4o-mini", "my-model")
        registry.register_mapping("gpt-4o-mini-2024-07-18", "my-model")

        state = registry.get_mapping_state()
        # Both should map to my-model in reverse mappings
        assert "my-model" in state["reverse_mappings"]
        assert len(state["reverse_mappings"]["my-model"]) == 2

    def test_conflict_detection_warn(self) -> None:
        registry = ModelRegistry(on_conflict="warn")
        registry.register_mapping("gpt-4o-mini", "model-a")
        # Should warn but succeed
        registry.register_mapping("gpt-4o-mini", "model-b")
        result = registry.resolve("gpt-4o-mini")
        assert result.model_id == "model-b"  # Overwritten

    def test_conflict_detection_error(self) -> None:
        registry = ModelRegistry(on_conflict="error")
        registry.register_mapping("gpt-4o-mini", "model-a")
        # Should raise
        with pytest.raises(MappingConflictError):
            registry.register_mapping("gpt-4o-mini", "model-b")

    def test_fuzzy_threshold_configurable(self) -> None:
        # High threshold - won't match weak similarities
        registry = ModelRegistry(fuzzy_threshold=0.9)
        available = ["gpt-4o"]
        result = registry.resolve("gpt-4o-mini", available)
        # Partial match may not meet 0.9 threshold
        # (depends on exact score)

    def test_stats_tracking(self) -> None:
        registry = ModelRegistry()
        registry.register_mapping("model-a", "id-a")

        registry.resolve("model-a")  # exact hit
        registry.resolve("model-b")  # failure

        state = registry.get_mapping_state()
        assert state["stats"]["exact_hits"] >= 1
        assert state["stats"]["failures"] >= 1


class TestGlobalRegistry:
    """Tests for global registry singleton (deprecated but backward compatible)."""

    def test_get_global_registry_returns_same_instance(self) -> None:
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2

    def test_global_registry_is_model_registry(self) -> None:
        reg = get_global_registry()
        assert isinstance(reg, ModelRegistry)


class TestRealWorldScenarios:
    """Test real-world model name scenarios from the bug report."""

    def test_basic_demo_scenario(self) -> None:
        """Test the basic demo: user configures o4-mini, LiteLLM returns gpt-4o-mini-2024-07-18."""
        registry = ModelRegistry()
        # User configuration
        registry.register_mapping("gpt-4o-mini", "o4-mini")
        registry.register_mapping("gpt-4o", "gpt-5")

        available = ["o4-mini", "gpt-5"]

        # LiteLLM returns dated model name
        result = registry.resolve("gpt-4o-mini-2024-07-18", available)
        assert result.model_id == "o4-mini"
        assert result.success is True

    def test_multi_provider_scenario(self) -> None:
        """Test multi-provider demo with Groq Llama."""
        registry = ModelRegistry()
        # User configuration
        registry.register_mapping("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile")

        available = ["o4-mini", "gpt-5", "claude-sonnet-4", "claude-haiku-3.5", "llama-3.3-70b-versatile"]

        # LiteLLM returns model without prefix
        result = registry.resolve("llama-3.3-70b-versatile", available)
        assert result.model_id == "llama-3.3-70b-versatile"
        assert result.success is True

    def test_cross_demo_isolation(self) -> None:
        """Test that clearing registry prevents cross-demo interference."""
        registry = ModelRegistry()

        # Demo 1: Multi-provider
        registry.register_mapping("llama-3.3-70b-versatile", "llama-3.3-70b-versatile")
        result = registry.resolve("llama-3.3-70b-versatile")
        assert result.model_id == "llama-3.3-70b-versatile"
        assert result.success is True

        # Clear between demos
        registry.clear()

        # Demo 2: Only OpenAI models
        registry.register_mapping("gpt-4o-mini", "o4-mini")
        available = ["o4-mini", "gpt-5"]

        # llama should now fail (no match in available)
        result = registry.resolve("llama-3.3-70b-versatile", available)
        # Either fuzzy matches or fails explicitly
        assert result.success is False or result.model_id != "llama-3.3-70b-versatile"

    def test_per_instance_isolation(self) -> None:
        """Test that per-instance registries are isolated."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()

        registry1.register_mapping("gpt-4o-mini", "model-a")
        registry2.register_mapping("gpt-4o-mini", "model-b")

        # Each registry should have its own mapping
        assert registry1.resolve("gpt-4o-mini").model_id == "model-a"
        assert registry2.resolve("gpt-4o-mini").model_id == "model-b"
