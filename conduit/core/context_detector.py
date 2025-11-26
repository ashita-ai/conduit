"""Context detection for query classification.

Maps query domains to routing contexts for context-specific priors.
"""

from conduit.core.models import QueryFeatures


class ContextDetector:
    """Detect query context for context-specific priors.

    Maps DomainClassifier domains to routing contexts:
    - code -> code
    - creative -> creative
    - math, science -> analysis
    - business -> analysis
    - general -> simple_qa (default)

    Example:
        >>> detector = ContextDetector()
        >>> context = detector.detect_from_features(features)
        >>> print(context)
        "code"
    """

    # Map DomainClassifier domains to routing contexts
    DOMAIN_TO_CONTEXT = {
        "code": "code",
        "creative": "creative",
        "math": "analysis",
        "science": "analysis",
        "business": "analysis",
        "general": "simple_qa",
    }

    def detect_from_features(self, features: QueryFeatures) -> str:
        """Detect context from query features.

        Args:
            features: QueryFeatures with domain classification

        Returns:
            Context string: "code", "creative", "analysis", or "simple_qa"

        Example:
            >>> features = QueryFeatures(
            ...     domain="code",
            ...     domain_confidence=0.9,
            ...     ...
            ... )
            >>> detector = ContextDetector()
            >>> context = detector.detect_from_features(features)
            >>> assert context == "code"
        """
        domain = features.domain
        return self.DOMAIN_TO_CONTEXT.get(domain, "simple_qa")

    def detect_from_text(self, query_text: str) -> str:
        """Detect context from query text using simple keyword matching.

        This is a fallback when features are not available.
        Uses keyword-based detection similar to DomainClassifier.

        Args:
            query_text: Raw query text

        Returns:
            Context string: "code", "creative", "analysis", or "simple_qa"

        Example:
            >>> detector = ContextDetector()
            >>> context = detector.detect_from_text("Write a Python function")
            >>> assert context == "code"
        """
        query_lower = query_text.lower()

        # Code keywords
        code_keywords = [
            "python",
            "code",
            "function",
            "debug",
            "class",
            "algorithm",
            "programming",
        ]
        if any(kw in query_lower for kw in code_keywords):
            return "code"

        # Creative keywords
        creative_keywords = [
            "write",
            "story",
            "poem",
            "creative",
            "imagine",
            "describe",
        ]
        if any(kw in query_lower for kw in creative_keywords):
            return "creative"

        # Analysis keywords
        analysis_keywords = ["analyze", "explain", "compare", "evaluate", "why", "how"]
        if any(kw in query_lower for kw in analysis_keywords):
            return "analysis"

        # Default to simple_qa
        return "simple_qa"
