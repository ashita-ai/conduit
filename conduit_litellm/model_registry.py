"""Robust model alias registry for LiteLLM to Conduit mapping.

This module provides production-grade model name resolution with:
- Explicit success/failure results (no silent fallbacks)
- Token-based fuzzy matching with configurable thresholds
- Conflict detection on mapping registration
- Thread-safe operations
- Structured logging for debugging
- Performance-optimized caching

Key design decisions:
1. ResolveResult tells you exactly what happened (no guessing)
2. Failures are explicit, not silent fallbacks
3. Fuzzy matching uses Jaccard similarity on tokens (semantic, not just prefix)
4. All public methods are thread-safe
5. Metrics callback for observability integration
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal

logger = logging.getLogger(__name__)


class MatchSource(str, Enum):
    """How a model mapping was resolved."""

    EXACT = "exact"  # Direct key match in aliases
    NORMALIZED = "normalized"  # Matched after normalization
    FUZZY = "fuzzy"  # Token-based fuzzy match
    FAILED = "failed"  # No match found


@dataclass(frozen=True)
class ResolveResult:
    """Result of model name resolution with confidence and provenance.

    This is the key improvement over the old design: callers know exactly
    whether resolution succeeded and how confident the match is.

    Attributes:
        model_id: The resolved Conduit model ID (or normalized input if failed)
        confidence: Match confidence (1.0 = exact, 0.95 = normalized, 0.3-0.9 = fuzzy, 0.0 = failed)
        source: How the match was found (EXACT, NORMALIZED, FUZZY, or FAILED)
        input_model: The original LiteLLM model name that was resolved
        alternatives: Other possible matches with their scores (for debugging)

    Example:
        >>> result = registry.resolve("gpt-4o-mini-2024-07-18", available_models)
        >>> if result:
        ...     print(f"Matched {result.model_id} via {result.source.value}")
        ... else:
        ...     print(f"Failed to match, best guess: {result.model_id}")
    """

    model_id: str
    confidence: float
    source: MatchSource
    input_model: str
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def success(self) -> bool:
        """True if resolution succeeded (not a failure)."""
        return self.source != MatchSource.FAILED

    def __bool__(self) -> bool:
        """Allow `if result:` pattern."""
        return self.success


# Regex patterns for stripping date/version suffixes
# Comprehensive list based on real-world LiteLLM responses
DATE_SUFFIX_PATTERNS = [
    r"-\d{4}-\d{2}-\d{2}$",  # -2024-07-18
    r"-\d{8}$",  # -20241022
    r"-\d{4}$",  # -0125 (gpt-4-0125-preview)
    r"-v\d+(\.\d+)?$",  # -v1, -v2.0
    r"-v\d+:\d+$",  # -v1:0 (bedrock)
    r":\d{4}-\d{2}-\d{2}$",  # :2024-07-18
    r"@\d+(\.\d+)*$",  # @1.0, @2.1.3 (some providers)
    r"-latest$",
    r"-preview$",
    r"-turbo$",
    r"-instruct$",
]

# Compiled regex for performance (applied once, not per-pattern)
_DATE_SUFFIX_RE = re.compile("|".join(f"({p})" for p in DATE_SUFFIX_PATTERNS))


@functools.lru_cache(maxsize=1)
def _get_provider_prefixes() -> frozenset[str]:
    """Get known provider prefixes, cached for performance.

    This is called frequently during normalization, so we cache the result.
    The frozenset is hashable and immutable.
    """
    prefixes: set[str] = set()

    try:
        from litellm import model_cost

        for model_name in model_cost.keys():
            if "/" in model_name:
                prefix = model_name.split("/")[0] + "/"
                # Skip fine-tuned model prefixes
                if not prefix.startswith("ft:"):
                    prefixes.add(prefix)
    except ImportError:
        pass

    # Always include common prefixes (in case LiteLLM not installed or incomplete)
    prefixes.update(
        {
            "openai/",
            "anthropic/",
            "google/",
            "groq/",
            "azure/",
            "bedrock/",
            "vertex_ai/",
            "cohere/",
            "mistral/",
            "together_ai/",
            "anyscale/",
            "deepinfra/",
            "perplexity/",
            "fireworks_ai/",
            "cloudflare/",
            "huggingface/",
            "ollama/",
            "replicate/",
            "gemini/",
            "voyage/",
            "jina/",
        }
    )

    return frozenset(prefixes)


def strip_provider_prefix(model_name: str) -> str:
    """Remove provider prefix from model name.

    Handles nested prefixes like "azure/us-east-1/gpt-4o-mini".

    Args:
        model_name: Model name potentially with provider prefix

    Returns:
        Model name without provider prefix

    Example:
        >>> strip_provider_prefix("openai/gpt-4o-mini")
        'gpt-4o-mini'
        >>> strip_provider_prefix("azure/us-east-1/gpt-4o")
        'gpt-4o'
    """
    result = model_name
    prefixes = _get_provider_prefixes()
    region_pattern = re.compile(r"^[a-z]{2,4}(-[a-z]+-\d+)?/")

    # Limit iterations to prevent infinite loops on malformed input
    for _ in range(5):
        changed = False

        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix) :]
                changed = True
                break

        # Also strip region prefixes (us-east-1/, eu/, apac/)
        match = region_pattern.match(result)
        if match and "/" in result:
            remainder = result[match.end() :]
            if remainder and not remainder.startswith("/"):
                result = remainder
                changed = True

        if not changed:
            break

    return result


def strip_date_suffix(model_name: str) -> str:
    """Remove date/version suffix from model name.

    Args:
        model_name: Model name potentially with date suffix

    Returns:
        Model name without date suffix

    Example:
        >>> strip_date_suffix("gpt-4o-mini-2024-07-18")
        'gpt-4o-mini'
        >>> strip_date_suffix("claude-3-5-sonnet-20241022")
        'claude-3-5-sonnet'
    """
    return _DATE_SUFFIX_RE.sub("", model_name)


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to canonical form.

    Applies:
    1. Strip provider prefix
    2. Strip date suffix
    3. Normalize Claude version format (3-5 -> 3.5)
    4. Lowercase

    Args:
        model_name: Raw model name from LiteLLM

    Returns:
        Normalized canonical model name

    Example:
        >>> normalize_model_name("openai/gpt-4o-mini-2024-07-18")
        'gpt-4o-mini'
        >>> normalize_model_name("claude-3-5-sonnet-20241022")
        'claude-3.5-sonnet'
    """
    result = strip_provider_prefix(model_name)
    result = strip_date_suffix(result)
    # Claude: 3-5 -> 3.5 for consistency
    result = re.sub(r"claude-(\d+)-(\d+)", r"claude-\1.\2", result)
    return result.lower()


def tokenize_model_name(model_name: str) -> list[str]:
    """Split model name into semantic tokens.

    Tokens are the meaningful parts of a model name that we can compare.
    This is more robust than prefix matching because it handles cases like
    "o4-mini" matching "gpt-4o-mini" (shared tokens: "4", "o", "mini").

    Args:
        model_name: Model name to tokenize

    Returns:
        List of lowercase tokens

    Example:
        >>> tokenize_model_name("gpt-4o-mini")
        ['gpt', '4o', 'mini']
        >>> tokenize_model_name("claude-3.5-sonnet")
        ['claude', '3', '5', 'sonnet']
    """
    normalized = normalize_model_name(model_name)
    # Split on common separators
    tokens = re.split(r"[-_./]", normalized)
    return [t for t in tokens if t]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Used as a secondary signal for fuzzy matching.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def calculate_match_score(query: str, candidate: str) -> float:
    """Calculate similarity score using token-based matching.

    This is a major improvement over the old prefix-only matching.
    Uses Jaccard similarity on tokens with bonuses for:
    - Exact normalized match: 1.0
    - Prefix match (query is prefix of candidate): +0.15
    - Token sequence match (model family): +0.1
    - Low edit distance: +0.05

    Args:
        query: Query model name
        candidate: Candidate model name

    Returns:
        Score in [0.0, 1.0], higher is better match

    Example:
        >>> calculate_match_score("gpt-4o-mini-2024-07-18", "gpt-4o-mini")
        1.0  # Exact after normalization
        >>> calculate_match_score("gpt-4o", "gpt-4o-mini")
        0.75  # High overlap but not exact
    """
    q_norm = normalize_model_name(query)
    c_norm = normalize_model_name(candidate)

    # Exact normalized match
    if q_norm == c_norm:
        return 1.0

    q_tokens = tokenize_model_name(query)
    c_tokens = tokenize_model_name(candidate)

    if not q_tokens or not c_tokens:
        return 0.0

    q_set = set(q_tokens)
    c_set = set(c_tokens)

    # Jaccard similarity: intersection / union
    intersection = len(q_set & c_set)
    union = len(q_set | c_set)
    jaccard = intersection / union if union > 0 else 0.0

    # Bonus for prefix match (query is prefix of candidate)
    # This handles cases like "llama-3.3-70b" matching "llama-3.3-70b-versatile"
    prefix_bonus = 0.0
    if c_norm.startswith(q_norm):
        prefix_bonus = 0.15

    # Bonus for matching token sequence (first tokens = model family)
    sequence_bonus = 0.0
    if len(q_tokens) >= 2 and len(c_tokens) >= 2:
        if q_tokens[0] == c_tokens[0]:
            sequence_bonus += 0.05
            if q_tokens[1] == c_tokens[1]:
                sequence_bonus += 0.05

    # Small bonus for low edit distance on normalized form
    max_len = max(len(q_norm), len(c_norm))
    if max_len > 0:
        edit_dist = _levenshtein_distance(q_norm, c_norm)
        edit_bonus = max(0, 0.05 * (1 - edit_dist / max_len))
    else:
        edit_bonus = 0.0

    return min(1.0, jaccard + sequence_bonus + prefix_bonus + edit_bonus)


def find_best_match(
    model_name: str, available_models: list[str], threshold: float = 0.5
) -> tuple[str | None, float, list[tuple[str, float]]]:
    """Find best matching model using token-based scoring.

    Scores all candidates and returns the best one above threshold.
    Unlike prefix matching, this handles semantic similarity.

    Args:
        model_name: Query model name
        available_models: Candidate models to match against
        threshold: Minimum score to accept (0.0-1.0)

    Returns:
        Tuple of (best_match, score, alternatives)
        - best_match: Best matching model or None if below threshold
        - score: Score of best match
        - alternatives: Other matches above threshold (for debugging)

    Example:
        >>> find_best_match("gpt-4o-mini-2024-07-18", ["gpt-4o-mini", "gpt-4o"])
        ('gpt-4o-mini', 1.0, [('gpt-4o', 0.67)])
    """
    if not available_models:
        return None, 0.0, []

    scores: list[tuple[str, float]] = []
    for candidate in available_models:
        score = calculate_match_score(model_name, candidate)
        if score >= threshold:
            scores.append((candidate, score))

    if not scores:
        return None, 0.0, []

    scores.sort(key=lambda x: x[1], reverse=True)
    best_model, best_score = scores[0]
    alternatives = scores[1:5]  # Top 4 alternatives for debugging

    return best_model, best_score, alternatives


class MappingConflictError(ValueError):
    """Raised when registering a mapping that conflicts with existing one."""

    pass


class ModelRegistry:
    """Thread-safe registry for model alias resolution.

    Key improvements over the old design:
    1. ResolveResult tells you exactly what happened (success, confidence, source)
    2. Failures are explicit, not silent fallbacks
    3. Token-based fuzzy matching (semantic, not just prefix)
    4. Conflict detection catches configuration errors early
    5. Thread-safe for concurrent use
    6. Metrics callback for observability

    Example:
        >>> registry = ModelRegistry(fuzzy_threshold=0.6)
        >>> registry.register("gpt-4o-mini", "fast-model")
        >>> result = registry.resolve("gpt-4o-mini-2024-07-18", ["fast-model", "smart-model"])
        >>> if result:
        ...     print(f"Resolved to {result.model_id} via {result.source.value}")
        ... else:
        ...     print(f"Failed to resolve, fallback: {result.model_id}")
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.5,
        on_conflict: Literal["error", "warn", "silent"] = "warn",
        metrics_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            fuzzy_threshold: Minimum score (0.0-1.0) for fuzzy matches.
                           0.5 = at least half the tokens must match.
                           Higher = fewer false positives, more misses.
            on_conflict: How to handle mapping conflicts:
                        "error" = raise MappingConflictError
                        "warn" = log warning and overwrite
                        "silent" = silently overwrite
            metrics_callback: Optional callback for resolution metrics.
                            Called with dict containing resolution details.
        """
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_threshold must be in [0.0, 1.0], got {fuzzy_threshold}"
            )

        self._fuzzy_threshold = fuzzy_threshold
        self._on_conflict = on_conflict
        self._metrics_callback = metrics_callback

        self._lock = threading.RLock()
        self._mappings: dict[str, str] = {}  # normalized_key -> conduit_id
        self._aliases: dict[str, str] = {}  # exact litellm name -> conduit_id
        self._reverse: dict[str, set[str]] = {}  # conduit_id -> litellm names

        # Stats for observability
        self._stats = {
            "exact_hits": 0,
            "normalized_hits": 0,
            "fuzzy_hits": 0,
            "failures": 0,
            "conflicts": 0,
        }

    def register_mapping(
        self,
        litellm_model: str,
        conduit_id: str,
        validate: bool = False,
    ) -> None:
        """Register a mapping from LiteLLM model name to Conduit ID.

        Stores both the exact form and normalized form for flexible matching.
        Detects conflicts where the same normalized form maps to different IDs.

        Args:
            litellm_model: Model name as returned by LiteLLM
            conduit_id: Conduit model ID to map to
            validate: If True, warn if litellm_model not in LiteLLM's model_cost

        Raises:
            MappingConflictError: If on_conflict="error" and mapping conflicts
        """
        normalized = normalize_model_name(litellm_model)

        with self._lock:
            # Check for conflicts
            if normalized in self._mappings:
                existing = self._mappings[normalized]
                if existing != conduit_id:
                    self._stats["conflicts"] += 1
                    msg = (
                        f"Mapping conflict: '{litellm_model}' (normalized: '{normalized}') "
                        f"already maps to '{existing}', attempted to map to '{conduit_id}'"
                    )
                    if self._on_conflict == "error":
                        raise MappingConflictError(msg)
                    elif self._on_conflict == "warn":
                        logger.warning(msg)
                    # For "silent", we just continue and overwrite

            # Validate against LiteLLM model_cost if requested
            if validate:
                try:
                    from litellm import model_cost

                    if litellm_model not in model_cost and normalized not in model_cost:
                        logger.warning(
                            f"Model '{litellm_model}' not found in LiteLLM model_cost. "
                            f"Mapping to '{conduit_id}' may not work as expected."
                        )
                except ImportError:
                    pass

            # Store mappings
            self._mappings[normalized] = conduit_id
            self._aliases[litellm_model] = conduit_id

            # Also store common variations
            without_prefix = strip_provider_prefix(litellm_model)
            if without_prefix != litellm_model:
                without_prefix_normalized = normalize_model_name(without_prefix)
                self._mappings[without_prefix_normalized] = conduit_id
                self._aliases[without_prefix] = conduit_id

            # Track reverse mapping for debugging
            if conduit_id not in self._reverse:
                self._reverse[conduit_id] = set()
            self._reverse[conduit_id].add(litellm_model)

            logger.debug(
                f"Registered mapping: '{litellm_model}' -> '{conduit_id}' "
                f"(normalized: '{normalized}')"
            )

    def resolve(
        self,
        litellm_model: str,
        available_models: list[str] | None = None,
    ) -> ResolveResult:
        """Resolve LiteLLM model name to Conduit ID.

        This is the main entry point. It tries multiple strategies in order
        and returns a ResolveResult that tells you exactly what happened.

        Resolution order:
        1. Exact alias match (confidence: 1.0)
        2. Normalized mapping match (confidence: 0.95)
        3. Fuzzy match against available_models (confidence: threshold-1.0)
        4. Failure with normalized form as fallback (confidence: 0.0)

        Args:
            litellm_model: Model name from LiteLLM response
            available_models: Available Conduit model IDs for fuzzy matching

        Returns:
            ResolveResult with model_id, confidence, source, and alternatives.
            Check `result.success` or `bool(result)` to determine if resolution succeeded.

        Example:
            >>> result = registry.resolve("gpt-4o-mini-2024-07-18", ["o4-mini"])
            >>> if result:
            ...     feedback.model_id = result.model_id
            ... else:
            ...     logger.warning(f"Skipping feedback: {result}")
        """
        normalized = normalize_model_name(litellm_model)

        with self._lock:
            # 1. Exact alias match (highest confidence)
            if litellm_model in self._aliases:
                self._stats["exact_hits"] += 1
                result = ResolveResult(
                    model_id=self._aliases[litellm_model],
                    confidence=1.0,
                    source=MatchSource.EXACT,
                    input_model=litellm_model,
                )
                self._emit_metrics(result)
                return result

            # 2. Normalized mapping match
            if normalized in self._mappings:
                self._stats["normalized_hits"] += 1
                result = ResolveResult(
                    model_id=self._mappings[normalized],
                    confidence=0.95,
                    source=MatchSource.NORMALIZED,
                    input_model=litellm_model,
                )
                self._emit_metrics(result)
                return result

            # 3. Fuzzy match against available models
            if available_models:
                best_match, score, alternatives = find_best_match(
                    litellm_model, available_models, self._fuzzy_threshold
                )
                if best_match:
                    self._stats["fuzzy_hits"] += 1
                    # Learn this mapping for future use
                    self.register_mapping(litellm_model, best_match)

                    result = ResolveResult(
                        model_id=best_match,
                        confidence=score,
                        source=MatchSource.FUZZY,
                        input_model=litellm_model,
                        alternatives=tuple(alternatives),
                    )
                    logger.info(
                        f"Fuzzy matched '{litellm_model}' -> '{best_match}' "
                        f"(score: {score:.2f}, alternatives: {len(alternatives)})"
                    )
                    self._emit_metrics(result)
                    return result

            # 4. Failure - EXPLICIT, not silent
            self._stats["failures"] += 1
            result = ResolveResult(
                model_id=normalized,  # Return normalized form as fallback value
                confidence=0.0,
                source=MatchSource.FAILED,
                input_model=litellm_model,
            )
            logger.warning(
                f"Failed to resolve model '{litellm_model}' (normalized: '{normalized}'). "
                f"Available models: {available_models[:5] if available_models else 'none'}..."
            )
            self._emit_metrics(result)
            return result

    def _emit_metrics(self, result: ResolveResult) -> None:
        """Emit metrics for observability."""
        if self._metrics_callback:
            try:
                self._metrics_callback(
                    {
                        "input": result.input_model,
                        "output": result.model_id,
                        "confidence": result.confidence,
                        "source": result.source.value,
                        "success": result.success,
                    }
                )
            except Exception as e:
                logger.debug(f"Metrics callback failed: {e}")

    def validate_mappings(self, available_models: list[str]) -> list[str]:
        """Validate all registered mappings resolve to available models.

        Call after initialization to catch configuration errors early.

        Args:
            available_models: List of valid target model IDs

        Returns:
            List of warning messages for invalid mappings
        """
        warnings = []
        available_set = set(available_models)

        with self._lock:
            for litellm_name, conduit_id in self._aliases.items():
                if conduit_id not in available_set:
                    msg = (
                        f"Invalid mapping: '{litellm_name}' -> '{conduit_id}' "
                        f"('{conduit_id}' not in available models)"
                    )
                    warnings.append(msg)
                    logger.warning(msg)

        return warnings

    def get_mapping_state(self) -> dict:
        """Get current registry state for debugging.

        Returns:
            Dict with mappings, aliases, reverse mappings, stats, and config
        """
        with self._lock:
            return {
                "runtime_mappings": dict(self._mappings),
                "reverse_mappings": {k: list(v) for k, v in self._reverse.items()},
                "total_mappings": len(self._mappings),
                "stats": dict(self._stats),
                "config": {
                    "fuzzy_threshold": self._fuzzy_threshold,
                    "on_conflict": self._on_conflict,
                },
            }

    def clear(self) -> None:
        """Clear all mappings and reset stats."""
        with self._lock:
            self._mappings.clear()
            self._aliases.clear()
            self._reverse.clear()
            for key in self._stats:
                self._stats[key] = 0


# Backward compatibility with deprecation warning
_global_registry: ModelRegistry | None = None


def get_global_registry() -> ModelRegistry:
    """Get global registry instance.

    .. deprecated::
        Use per-instance ModelRegistry instead to avoid state pollution.
    """
    import warnings

    warnings.warn(
        "get_global_registry() is deprecated. Create per-instance ModelRegistry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
