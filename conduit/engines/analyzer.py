"""Query analysis and feature extraction for routing decisions."""

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

from conduit.cache import CacheService
from conduit.core.models import QueryFeatures
from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.embeddings.factory import create_embedding_provider

if TYPE_CHECKING:
    from sklearn.decomposition import PCA  # type: ignore[import-untyped,unused-ignore]

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Extract semantic and structural features from queries.

    Uses configurable embedding providers (HuggingFace API, OpenAI, Cohere, or
    sentence-transformers) for embeddings and heuristics for complexity scoring
    and domain classification.

    Features caching with Redis for performance optimization of the
    expensive embedding computation (200ms -> 5ms on cache hit).

    Default: HuggingFace Inference API (free, no API key needed)
    Recommended: OpenAI or Cohere (better quality, requires API keys)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_type: str = "huggingface",
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        cache_service: CacheService | None = None,
        use_pca: bool = False,
        pca_dimensions: int = 64,
        pca_model_path: str = "models/pca.pkl",
    ):
        """Initialize analyzer with embedding provider and optional cache.

        Args:
            embedding_provider: Pre-configured embedding provider (optional)
            embedding_provider_type: Provider type ("huggingface", "openai", "cohere", "sentence-transformers")
            embedding_model: Model identifier (provider-specific, optional)
            embedding_api_key: API key for providers that require it (optional)
            cache_service: Optional cache service for feature caching
            use_pca: Enable PCA dimensionality reduction (default: False)
            pca_dimensions: Target dimensions after PCA (default: 64)
            pca_model_path: Path to fitted PCA model (default: models/pca.pkl)

        Example:
            >>> # Free default (HuggingFace API, no API key needed)
            >>> analyzer = QueryAnalyzer()
            >>>
            >>> # OpenAI (recommended for production)
            >>> analyzer = QueryAnalyzer(
            ...     embedding_provider_type="openai",
            ...     embedding_model="text-embedding-3-small",
            ...     embedding_api_key="sk-..."
            ... )
            >>>
            >>> # Cohere (recommended for production)
            >>> analyzer = QueryAnalyzer(
            ...     embedding_provider_type="cohere",
            ...     embedding_api_key="..."
            ... )
            >>>
            >>> # Reduced 64-dim embeddings with PCA
            >>> analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)
        """
        # Use provided provider or create one
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            # Create provider based on type
            self.embedding_provider = create_embedding_provider(
                provider=embedding_provider_type,
                model=embedding_model,
                api_key=embedding_api_key,
            )

        self.domain_classifier = DomainClassifier()
        self.cache = cache_service

        # PCA configuration
        self.use_pca = use_pca
        self.pca_dimensions = pca_dimensions
        self.pca_model_path = pca_model_path
        self.pca: PCA | None = None

        if use_pca:
            # Lazy load sklearn.decomposition.PCA only when needed
            from sklearn.decomposition import PCA  # type: ignore[import-untyped]

            # Try to load pre-fitted PCA model
            self.pca = self._load_pca()
            if self.pca is None:
                # Create new PCA (needs fitting before use)
                self.pca = PCA(n_components=pca_dimensions)

    async def analyze(self, query: str) -> QueryFeatures:
        """Extract features from query for routing decision.

        Args:
            query: User query text

        Returns:
            QueryFeatures with embedding, complexity, domain

        Performance:
            - Cache hit: ~5ms (Redis GET + deserialize)
            - Cache miss: ~200ms (embedding computation)
            - Expected hit rate: 80%+ after 1 week

        Example:
            >>> analyzer = QueryAnalyzer()
            >>> features = await analyzer.analyze("What is photosynthesis?")
            >>> features.complexity_score
            0.3
            >>> features.domain
            "science"
        """
        # Try cache first if available
        if self.cache:
            cached_features = await self.cache.get(query)
            if cached_features is not None:
                return cached_features

        # Cache miss or no cache - compute features
        # Generate embedding using provider
        embedding_list = await self.embedding_provider.embed(query)

        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            # Check if PCA is fitted
            if not hasattr(self.pca, "components_"):
                raise RuntimeError(
                    "PCA is enabled but not fitted. Call fit_pca() with training data first."
                )
            # Transform to reduced dimensions (PCA expects numpy array)
            import numpy as np

            embedding_array = np.array(embedding_list)
            embedding_array = self.pca.transform([embedding_array])[0]
            embedding_list = embedding_array.tolist()

        # Estimate token count (rough approximation)
        token_count = self._estimate_tokens(query)

        # Compute complexity score (0.0-1.0)
        complexity_score = self._compute_complexity(query, token_count)

        # Classify domain
        domain, domain_confidence = self.domain_classifier.classify(query)

        features = QueryFeatures(
            embedding=embedding_list,
            token_count=token_count,
            complexity_score=complexity_score,
            domain=domain,
            domain_confidence=domain_confidence,
        )

        # Store in cache for future requests
        if self.cache:
            await self.cache.set(query, features)

        return features

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word count heuristic.

        Args:
            text: Input text

        Returns:
            Estimated token count (words * 1.3)
        """
        words = len(text.split())
        return int(words * 1.3)  # Rough approximation

    def _compute_complexity(self, text: str, token_count: int) -> float:
        """Compute complexity score based on structural features.

        Args:
            text: Input text
            token_count: Estimated tokens

        Returns:
            Complexity score (0.0-1.0)

        Complexity Factors:
            - Length: Longer queries more complex
            - Technical terms: Code, math, jargon
            - Question depth: Multiple questions
            - Specificity: Detailed requirements
        """
        complexity = 0.0

        # Length factor (0.0-0.3)
        if token_count < 20:
            complexity += 0.1
        elif token_count < 50:
            complexity += 0.2
        else:
            complexity += 0.3

        # Technical indicators (0.0-0.3)
        technical_patterns = [
            r"\b(function|class|algorithm|implementation)\b",
            r"\b(optimization|complexity|performance)\b",
            r"\b(SQL|API|HTTP|REST|JSON)\b",
            r"```|`[\w]+`",  # Code blocks
            r"\b(theorem|proof|equation|formula)\b",
        ]

        for pattern in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                complexity += 0.06
                if complexity >= 0.6:
                    break

        # Multiple questions (0.0-0.2)
        question_count = text.count("?")
        if question_count > 1:
            complexity += min(0.2, question_count * 0.05)

        # Detailed requirements (0.0-0.2)
        requirement_indicators = [
            r"\b(must|should|need to|require)\b",
            r"\b(ensure|guarantee|verify)\b",
            r"\b(step\s+\d+|first|second|third)\b",
        ]

        for pattern in requirement_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                complexity += 0.05
                if complexity >= 1.0:
                    break

        return min(1.0, max(0.0, complexity))

    async def fit_pca(self, queries: list[str]) -> None:
        """Fit PCA on representative query set (one-time setup).

        Args:
            queries: Representative queries for PCA fitting (1000+ recommended)

        Raises:
            ValueError: If PCA not enabled or insufficient queries
            RuntimeError: If embedding generation fails

        Example:
            >>> analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)
            >>> training_queries = [...]  # 1,000+ diverse queries
            >>> analyzer.fit_pca(training_queries)
            >>> # PCA is now fitted and saved to disk
        """
        if not self.use_pca or self.pca is None:
            raise ValueError("PCA not enabled. Set use_pca=True in constructor.")

        if len(queries) < 100:
            raise ValueError(
                f"Need at least 100 queries for PCA fitting, got {len(queries)}"
            )

        # Generate embeddings for all queries using provider
        embeddings_list = await self.embedding_provider.embed_batch(queries)

        # Convert to numpy array for PCA
        import numpy as np

        embeddings = np.array(embeddings_list)

        # Fit PCA
        self.pca.fit(embeddings)

        # Save fitted model
        self._save_pca()

    def _load_pca(self) -> "PCA | None":
        """Load pre-fitted PCA model from disk.

        Returns:
            Fitted PCA model or None if file doesn't exist
        """
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]

        pca_path = Path(self.pca_model_path)
        if not pca_path.exists():
            return None

        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
                if not isinstance(pca, PCA):
                    raise ValueError("Loaded object is not a PCA model")
                return pca
        except Exception as e:
            # Log warning but don't fail - will create new PCA
            logger.warning(f"Failed to load PCA model from {pca_path}: {e}")
            return None

    def _save_pca(self) -> None:
        """Save fitted PCA model to disk."""
        if self.pca is None:
            return

        pca_path = Path(self.pca_model_path)
        pca_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pca_path, "wb") as f:
            pickle.dump(self.pca, f)

    @property
    def feature_dim(self) -> int:
        """Get total feature dimensionality (embedding + metadata).

        Returns:
            Total feature dimensions (embedding_dim + 3 metadata)

        Example:
            >>> # Without PCA
            >>> analyzer = QueryAnalyzer()
            >>> analyzer.feature_dim
            387  # 384 + 3 (HuggingFace default)

            >>> # With PCA
            >>> analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)
            >>> analyzer.feature_dim
            67  # 64 + 3

            >>> # OpenAI embeddings (1536 dims)
            >>> analyzer = QueryAnalyzer(embedding_provider_type="openai")
            >>> analyzer.feature_dim
            1539  # 1536 + 3
        """
        if self.use_pca:
            embedding_dim = self.pca_dimensions
        else:
            embedding_dim = self.embedding_provider.dimension
        metadata_dim = 3  # token_count, complexity_score, domain_confidence
        return embedding_dim + metadata_dim


class DomainClassifier:
    """Classify query domain using keyword matching.

    Simple keyword-based classifier for Phase 1.
    Phase 2+ will use ML-based classification.
    """

    DOMAIN_KEYWORDS = {
        "code": [
            "function",
            "class",
            "algorithm",
            "implementation",
            "debug",
            "refactor",
            "code",
            "programming",
            "python",
            "javascript",
            "java",
        ],
        "analysis": [
            "analyze",
            "evaluation",
            "compare",
            "pros",
            "cons",
            "advantages",
            "disadvantages",
            "assessment",
            "review",
            "examine",
        ],
        "simple_qa": [
            "what is",
            "who is",
            "when",
            "where",
            "capital",
            "define",
            "explain simply",
            "quick question",
        ],
        "math": [
            "equation",
            "formula",
            "theorem",
            "proof",
            "calculate",
            "derivative",
            "integral",
        ],
        "science": [
            "photosynthesis",
            "molecule",
            "experiment",
            "hypothesis",
            "theory",
            "evolution",
        ],
        "business": [
            "revenue",
            "strategy",
            "market",
            "customer",
            "profit",
            "growth",
            "ROI",
        ],
        "creative": [
            "story",
            "poem",
            "creative",
            "imagine",
            "describe",
            "brainstorm",
        ],
        "general": [],  # Default fallback
    }

    def classify(self, query: str) -> tuple[str, float]:
        """Classify query domain using keyword matching.

        Args:
            query: User query text

        Returns:
            Tuple of (domain, confidence)

        Example:
            >>> classifier = DomainClassifier()
            >>> domain, confidence = classifier.classify("Write a function to sort numbers")
            >>> domain
            "code"
            >>> confidence > 0.7
            True
        """
        query_lower = query.lower()
        domain_scores: dict[str, int] = dict.fromkeys(self.DOMAIN_KEYWORDS, 0)

        # Count keyword matches per domain
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    domain_scores[domain] += 1

        # Find domain with highest score
        max_score = max(domain_scores.values())

        if max_score == 0:
            return ("general", 0.5)  # Default domain with medium confidence

        best_domain = max(domain_scores, key=domain_scores.get)  # type: ignore
        confidence = min(1.0, 0.6 + (max_score * 0.1))  # 0.6-1.0 range

        return (best_domain, confidence)
