"""Utility functions for configuration loading.

These helpers are used by other config modules for parsing values
and loading configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type.

    Args:
        value: String value from environment variable

    Returns:
        Parsed value (int, float, bool, or str)

    Example:
        >>> parse_env_value("1.5")
        1.5
        >>> parse_env_value("true")
        True
        >>> parse_env_value("hello")
        'hello'
    """
    # Try boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def load_default_models() -> list[str]:
    """Load default model list from priors defined in conduit.yaml.

    Extracts all unique model IDs across all context priors (code, creative, etc.).
    This ensures consistency between priors and available models.

    Returns:
        List of model IDs to use for routing. Falls back to hardcoded defaults if YAML not found.
    """
    # Search for conduit.yaml in multiple locations (prioritize project root)
    search_paths = [
        Path(__file__).parent.parent.parent.parent
        / "conduit.yaml",  # Conduit project root (priority)
        Path("conduit.yaml"),  # Current directory
        Path.cwd() / "conduit.yaml",  # Explicit current directory
    ]

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        continue

                    priors_section = config.get("priors", {})
                    if not isinstance(priors_section, dict):
                        continue

                    # Extract all unique model IDs from all contexts
                    model_ids: set[str] = set()
                    for context_priors in priors_section.values():
                        if isinstance(context_priors, dict):
                            model_ids.update(context_priors.keys())

                    if model_ids:
                        # Return sorted list for consistent ordering
                        return sorted(model_ids)
            except Exception:
                continue

    # Fallback: hardcoded defaults if no YAML config found
    return [
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "gpt-5",
        "gpt-5.1",
        "o4-mini",
    ]


def load_embeddings_config() -> dict[str, Any]:
    """Load embeddings configuration from conduit.yaml.

    Returns configuration for embedding provider, model selection, and PCA settings.
    Falls back to hardcoded defaults if YAML not found.

    Returns:
        Dictionary with embedding configuration:
        - provider: Embedding provider type (auto, openai, cohere, fastembed, etc.)
        - model: Provider-specific model identifier (null = use provider default)
        - pca_enabled: Enable PCA dimensionality reduction
        - pca_components: Number of PCA components
        - pca_auto_retrain: Auto-retrain PCA on workload
        - pca_retrain_threshold: Minimum queries before auto-retraining
    """
    # Defaults (PCA disabled by default)
    defaults = {
        "provider": "auto",
        "model": None,
        "pca_enabled": False,
        "pca_components": 128,
        "pca_auto_retrain": True,
        "pca_retrain_threshold": 150,
    }

    # Search for conduit.yaml
    search_paths = [
        Path(__file__).parent.parent.parent.parent
        / "conduit.yaml",  # Conduit project root
        Path("conduit.yaml"),  # Current directory
        Path.cwd() / "conduit.yaml",  # Explicit current directory
    ]

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        embeddings_config = config.get("embeddings", {})
                        if isinstance(embeddings_config, dict) and embeddings_config:
                            # Build result from YAML
                            result = defaults.copy()

                            # Top-level settings
                            if "provider" in embeddings_config:
                                result["provider"] = embeddings_config["provider"]
                            if "model" in embeddings_config:
                                result["model"] = embeddings_config["model"]

                            # PCA settings (nested)
                            pca_config = embeddings_config.get("pca", {})
                            if isinstance(pca_config, dict) and pca_config:
                                if "enabled" in pca_config:
                                    result["pca_enabled"] = pca_config["enabled"]
                                if "components" in pca_config:
                                    result["pca_components"] = pca_config["components"]
                                if "auto_retrain" in pca_config:
                                    result["pca_auto_retrain"] = pca_config[
                                        "auto_retrain"
                                    ]
                                if "retrain_threshold" in pca_config:
                                    result["pca_retrain_threshold"] = pca_config[
                                        "retrain_threshold"
                                    ]

                            return result
            except Exception:
                continue

    # Try environment variables as fallback
    env_overrides: dict[str, str | bool | int] = {}
    if os.getenv("EMBEDDING_PROVIDER"):
        env_overrides["provider"] = os.getenv("EMBEDDING_PROVIDER", "auto")
    if os.getenv("EMBEDDING_MODEL"):
        env_val = os.getenv("EMBEDDING_MODEL")
        if env_val is not None:
            env_overrides["model"] = env_val
    if os.getenv("USE_PCA"):
        env_overrides["pca_enabled"] = os.getenv("USE_PCA", "false").lower() == "true"
    if os.getenv("PCA_COMPONENTS"):
        env_overrides["pca_components"] = int(os.getenv("PCA_COMPONENTS", "128"))

    return {**defaults, **env_overrides}
