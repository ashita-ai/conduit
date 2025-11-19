"""Query analysis and feature extraction for routing decisions."""

import asyncio
import re

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped,unused-ignore]

from conduit.core.models import QueryFeatures


class QueryAnalyzer:
    """Extract semantic and structural features from queries.

    Uses sentence-transformers for embeddings and heuristics for
    complexity scoring and domain classification.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize analyzer with embedding model.

        Args:
            embedding_model: HuggingFace model for embeddings
        """
        self.embedder = SentenceTransformer(embedding_model)
        self.domain_classifier = DomainClassifier()

    async def analyze(self, query: str) -> QueryFeatures:
        """Extract features from query for routing decision.

        Args:
            query: User query text

        Returns:
            QueryFeatures with embedding, complexity, domain

        Example:
            >>> analyzer = QueryAnalyzer()
            >>> features = await analyzer.analyze("What is photosynthesis?")
            >>> features.complexity_score
            0.3
            >>> features.domain
            "science"
        """
        # Generate embedding (offload CPU work to thread pool to avoid blocking event loop)
        embedding_tensor = await asyncio.to_thread(self.embedder.encode, query)
        # sentence_transformers returns numpy array-like objects
        embedding_list: list[float] = embedding_tensor.tolist()

        # Estimate token count (rough approximation)
        token_count = self._estimate_tokens(query)

        # Compute complexity score (0.0-1.0)
        complexity_score = self._compute_complexity(query, token_count)

        # Classify domain
        domain, domain_confidence = self.domain_classifier.classify(query)

        return QueryFeatures(
            embedding=embedding_list,
            token_count=token_count,
            complexity_score=complexity_score,
            domain=domain,
            domain_confidence=domain_confidence,
        )

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
