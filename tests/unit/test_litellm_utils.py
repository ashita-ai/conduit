"""Tests for conduit_litellm/utils.py utility functions."""

from typing import Any
from unittest.mock import patch

import pytest

from conduit_litellm.utils import (
    _generate_model_id,
    extract_model_ids,
    extract_query_text,
    format_routing_metadata,
    map_litellm_to_conduit,
    normalize_model_name,
    validate_litellm_model_list,
)


class TestValidateLitellmModelList:
    """Tests for validate_litellm_model_list function."""

    def test_valid_model_list_with_model_name(self) -> None:
        """Valid model_list with model_name passes validation."""
        model_list = [
            {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3"},
            },
        ]
        validate_litellm_model_list(model_list)

    def test_valid_model_list_with_litellm_params_only(self) -> None:
        """Valid model_list with only litellm_params.model passes validation."""
        model_list = [
            {"litellm_params": {"model": "openai/gpt-4"}},
        ]
        validate_litellm_model_list(model_list)

    def test_invalid_not_a_list(self) -> None:
        """Non-list input raises ValueError."""
        with pytest.raises(ValueError, match="model_list must be a list"):
            validate_litellm_model_list("not a list")  # type: ignore[arg-type]

    def test_invalid_empty_list(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="model_list cannot be empty"):
            validate_litellm_model_list([])

    def test_invalid_deployment_not_dict(self) -> None:
        """Non-dict deployment raises ValueError."""
        with pytest.raises(ValueError, match="Deployment 0 must be a dictionary"):
            validate_litellm_model_list(["not a dict"])  # type: ignore[list-item]

    def test_invalid_deployment_missing_model_info(self) -> None:
        """Deployment without model_name or litellm_params.model raises ValueError."""
        model_list = [{"some_other_key": "value"}]
        with pytest.raises(
            ValueError,
            match="Deployment 0 must have 'model_name' or 'litellm_params.model'",
        ):
            validate_litellm_model_list(model_list)

    def test_invalid_deployment_litellm_params_without_model(self) -> None:
        """Deployment with litellm_params but no model key raises ValueError."""
        model_list = [{"litellm_params": {"api_key": "xxx"}}]
        with pytest.raises(
            ValueError,
            match="Deployment 0 must have 'model_name' or 'litellm_params.model'",
        ):
            validate_litellm_model_list(model_list)

    def test_valid_model_list_with_model_info(self) -> None:
        """Valid model_list with model_info passes validation."""
        model_list = [
            {"model_name": "gpt-4", "model_info": {"id": "gpt-4-openai"}},
        ]
        validate_litellm_model_list(model_list)


class TestGenerateModelId:
    """Tests for _generate_model_id function."""

    def test_extracts_from_model_info_dict_with_id(self) -> None:
        """Extracts ID from model_info.id when present."""
        deployment = {"model_info": {"id": "custom-model-id"}}
        assert _generate_model_id(deployment, 0) == "custom-model-id"

    def test_extracts_from_model_info_string(self) -> None:
        """Extracts ID from model_info when it is a string."""
        deployment = {"model_info": "string-model-id"}
        assert _generate_model_id(deployment, 0) == "string-model-id"

    def test_extracts_from_model_name(self) -> None:
        """Extracts ID from model_name when model_info not present."""
        deployment = {"model_name": "gpt-4o-mini"}
        assert _generate_model_id(deployment, 0) == "gpt-4o-mini"

    def test_extracts_from_litellm_params_model(self) -> None:
        """Extracts ID from litellm_params.model when model_name not present."""
        deployment = {"litellm_params": {"model": "openai/gpt-4"}}
        assert _generate_model_id(deployment, 0) == "gpt-4"

    def test_normalizes_model_name_with_provider_prefix(self) -> None:
        """Normalizes model_name by removing provider prefix."""
        deployment = {"model_name": "openai/gpt-4o-mini"}
        assert _generate_model_id(deployment, 0) == "gpt-4o-mini"

    def test_fallback_to_index(self) -> None:
        """Falls back to index-based ID when no other source available."""
        deployment: dict[str, Any] = {}
        assert _generate_model_id(deployment, 5) == "model-5"

    def test_model_info_dict_without_id(self) -> None:
        """Falls back when model_info is dict but has no id key."""
        deployment = {"model_info": {"description": "A model"}, "model_name": "gpt-4"}
        assert _generate_model_id(deployment, 0) == "gpt-4"


class TestExtractModelIds:
    """Tests for extract_model_ids function."""

    def test_extracts_from_model_names(self) -> None:
        """Extracts model IDs from model_name fields."""
        model_list = [
            {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3"},
            },
        ]
        ids = extract_model_ids(model_list)
        assert ids == ["gpt-4", "claude-3"]

    def test_extracts_from_model_info_ids(self) -> None:
        """Extracts model IDs from model_info.id fields."""
        model_list = [
            {"model_info": {"id": "custom-gpt-4"}, "model_name": "gpt-4"},
            {"model_info": {"id": "custom-claude"}, "model_name": "claude"},
        ]
        ids = extract_model_ids(model_list)
        assert ids == ["custom-gpt-4", "custom-claude"]

    def test_mixed_sources(self) -> None:
        """Handles mixed sources for model IDs - model_info.id takes precedence."""
        model_list = [
            # model_info.id takes precedence over model_name
            {"model_info": {"id": "explicit-id"}, "model_name": "ignored"},
            # model_name used when no model_info.id
            {"model_name": "model-name-id"},
            # litellm_params.model used when neither above present
            {"litellm_params": {"model": "provider/litellm-model"}},
        ]
        ids = extract_model_ids(model_list)
        assert ids == ["explicit-id", "model-name-id", "litellm-model"]

    def test_validates_before_extraction(self) -> None:
        """Validates model_list before extracting IDs."""
        with pytest.raises(ValueError, match="model_list cannot be empty"):
            extract_model_ids([])


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_removes_provider_prefix(self) -> None:
        """Removes provider prefix from model name."""
        assert normalize_model_name("openai/gpt-4") == "gpt-4"
        assert normalize_model_name("anthropic/claude-3-opus") == "claude-3-opus"
        assert normalize_model_name("google/gemini-pro") == "gemini-pro"

    def test_handles_no_prefix(self) -> None:
        """Returns model name unchanged when no prefix present."""
        assert normalize_model_name("gpt-4") == "gpt-4"
        assert normalize_model_name("claude-3-opus") == "claude-3-opus"

    def test_handles_multiple_slashes(self) -> None:
        """Only removes first prefix when multiple slashes present."""
        assert normalize_model_name("azure/openai/gpt-4") == "openai/gpt-4"


class TestMapLitellmToConduit:
    """Tests for map_litellm_to_conduit function."""

    def test_maps_known_model_with_config(self) -> None:
        """Maps known model using config mappings."""
        mock_mappings = {
            "gpt-4o-mini": "o4-mini",
            "claude-3-5-sonnet-20241022": "claude-sonnet-4.5",
        }
        with patch(
            "conduit_litellm.utils.load_litellm_config",
            return_value={"model_mappings": mock_mappings},
        ):
            assert map_litellm_to_conduit("gpt-4o-mini") == "o4-mini"
            assert (
                map_litellm_to_conduit("claude-3-5-sonnet-20241022")
                == "claude-sonnet-4.5"
            )

    def test_returns_unchanged_for_unknown_model(self) -> None:
        """Returns model name unchanged when no mapping exists."""
        with patch(
            "conduit_litellm.utils.load_litellm_config",
            return_value={"model_mappings": {}},
        ):
            assert map_litellm_to_conduit("unknown-model") == "unknown-model"

    def test_normalizes_before_mapping(self) -> None:
        """Normalizes model name (removes provider prefix) before mapping."""
        mock_mappings = {"gpt-4o-mini": "o4-mini"}
        with patch(
            "conduit_litellm.utils.load_litellm_config",
            return_value={"model_mappings": mock_mappings},
        ):
            assert map_litellm_to_conduit("openai/gpt-4o-mini") == "o4-mini"

    def test_handles_empty_mappings(self) -> None:
        """Handles config with no model_mappings key."""
        with patch(
            "conduit_litellm.utils.load_litellm_config",
            return_value={},
        ):
            assert map_litellm_to_conduit("some-model") == "some-model"


class TestFormatRoutingMetadata:
    """Tests for format_routing_metadata function."""

    def test_formats_complete_metadata(self) -> None:
        """Formats metadata with all fields."""

        class MockFeatures:
            complexity_score = 0.75
            domain = "code"

        metadata = format_routing_metadata(
            selected_model="gpt-4",
            confidence=0.85,
            reasoning="High quality for complex query",
            features=MockFeatures(),
        )

        assert metadata == {
            "conduit_selected_model": "gpt-4",
            "conduit_confidence": 0.85,
            "conduit_reasoning": "High quality for complex query",
            "conduit_complexity": 0.75,
            "conduit_domain": "code",
        }

    def test_handles_missing_feature_attributes(self) -> None:
        """Handles features object without complexity_score or domain."""

        class MinimalFeatures:
            pass

        metadata = format_routing_metadata(
            selected_model="claude-3",
            confidence=0.9,
            reasoning="Selected for general task",
            features=MinimalFeatures(),
        )

        assert metadata["conduit_selected_model"] == "claude-3"
        assert metadata["conduit_confidence"] == 0.9
        assert metadata["conduit_complexity"] is None
        assert metadata["conduit_domain"] is None

    def test_handles_none_features(self) -> None:
        """Handles None as features argument."""
        metadata = format_routing_metadata(
            selected_model="model",
            confidence=0.5,
            reasoning="Default",
            features=None,
        )

        assert metadata["conduit_complexity"] is None
        assert metadata["conduit_domain"] is None


class TestExtractQueryText:
    """Tests for extract_query_text function."""

    def test_extracts_from_user_message(self) -> None:
        """Extracts text from last user message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert extract_query_text(messages=messages) == "What is 2+2?"

    def test_extracts_last_user_message(self) -> None:
        """Extracts from last user message when multiple exist."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"},
        ]
        assert extract_query_text(messages=messages) == "Follow-up question"

    def test_extracts_from_multimodal_message(self) -> None:
        """Extracts text from multimodal message with list content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ]
        assert extract_query_text(messages=messages) == "Describe this image"

    def test_extracts_from_string_list_content(self) -> None:
        """Extracts from list content with string items."""
        messages = [
            {"role": "user", "content": ["Direct string content", "more content"]}
        ]
        assert extract_query_text(messages=messages) == "Direct string content"

    def test_extracts_from_input_data_string(self) -> None:
        """Extracts from input_data when it is a string."""
        assert extract_query_text(input_data="Direct prompt") == "Direct prompt"

    def test_extracts_from_input_data_list(self) -> None:
        """Extracts from input_data when it is a list."""
        assert extract_query_text(input_data=["First", "Second"]) == "First Second"

    def test_prefers_messages_over_input_data(self) -> None:
        """Prefers messages when both are provided."""
        messages = [{"role": "user", "content": "From messages"}]
        result = extract_query_text(messages=messages, input_data="From input")
        assert result == "From messages"

    def test_returns_empty_for_no_input(self) -> None:
        """Returns empty string when no input provided."""
        assert extract_query_text() == ""
        assert extract_query_text(messages=None, input_data=None) == ""

    def test_returns_empty_for_no_user_message(self) -> None:
        """Returns empty string when no user message in messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Assistant response"},
        ]
        assert extract_query_text(messages=messages) == ""

    def test_handles_non_dict_messages(self) -> None:
        """Handles messages list with non-dict items."""
        messages = [
            "not a dict",  # type: ignore[list-item]
            {"role": "user", "content": "Valid message"},
        ]
        assert extract_query_text(messages=messages) == "Valid message"

    def test_handles_user_message_with_empty_content(self) -> None:
        """Handles user message with empty string content."""
        messages = [{"role": "user", "content": ""}]
        assert extract_query_text(messages=messages) == ""

    def test_handles_user_message_with_list_but_no_text(self) -> None:
        """Returns empty when user message has list content but no text type."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com"}}
                ],
            }
        ]
        assert extract_query_text(messages=messages) == ""
