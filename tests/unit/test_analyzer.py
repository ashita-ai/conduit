"""Unit tests for QueryAnalyzer and DomainClassifier."""

import pytest
from unittest.mock import Mock, patch

from conduit.engines.analyzer import DomainClassifier, QueryAnalyzer
from conduit.core.models import QueryFeatures


class TestDomainClassifier:
    """Tests for DomainClassifier."""

    def test_classify_code_domain(self):
        """Test classification of code-related queries."""
        classifier = DomainClassifier()
        domain, confidence = classifier.classify(
            "Write a function to implement binary search algorithm"
        )

        assert domain == "code"
        assert confidence > 0.6

    def test_classify_math_domain(self):
        """Test classification of math-related queries."""
        classifier = DomainClassifier()
        domain, confidence = classifier.classify(
            "Prove the derivative of x^2 is 2x using first principles"
        )

        assert domain == "math"
        assert confidence > 0.6

    def test_classify_science_domain(self):
        """Test classification of science-related queries."""
        classifier = DomainClassifier()
        domain, confidence = classifier.classify("Explain photosynthesis process")

        assert domain == "science"
        assert confidence > 0.6

    def test_classify_business_domain(self):
        """Test classification of business-related queries."""
        classifier = DomainClassifier()
        domain, confidence = classifier.classify(
            "Analyze market strategy and revenue growth"
        )

        assert domain == "business"
        assert confidence > 0.6

    def test_classify_general_domain(self):
        """Test classification defaults to general for unclear queries."""
        classifier = DomainClassifier()
        domain, confidence = classifier.classify("Tell me about yesterday")

        assert domain == "general"
        assert confidence == 0.5


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    @pytest.mark.asyncio
    @patch('conduit.engines.analyzer.SentenceTransformer')
    async def test_analyze_simple_query(self, mock_transformer_class):
        """Test analysis of simple query."""
        # Mock the sentence transformer
        mock_transformer = Mock()
        mock_transformer.encode.return_value = Mock(tolist=lambda: [0.1] * 384)
        mock_transformer_class.return_value = mock_transformer

        from conduit.engines.analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        features = await analyzer.analyze("What is 2+2?")

        assert isinstance(features, QueryFeatures)
        assert len(features.embedding) == 384
        assert features.token_count > 0
        assert 0.0 <= features.complexity_score <= 1.0
        assert features.complexity_score < 0.3  # Simple query
        assert features.domain_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_complex_query(self):
        """Test analysis of complex technical query."""
        analyzer = QueryAnalyzer()
        features = await analyzer.analyze(
            """
            Implement a REST API using FastAPI with the following requirements:
            1. Authentication using JWT tokens
            2. Rate limiting middleware
            3. Database connection pooling
            4. Comprehensive error handling
            Ensure all endpoints are properly documented with OpenAPI specs.
            """
        )

        assert isinstance(features, QueryFeatures)
        assert features.complexity_score > 0.25  # Moderately complex query
        assert features.domain in ["code", "general"]  # Code or general domain
        assert features.token_count > 20

    @pytest.mark.asyncio
    async def test_embedding_dimension(self):
        """Test embedding has correct dimensionality."""
        analyzer = QueryAnalyzer()
        features = await analyzer.analyze("Test query")

        assert len(features.embedding) == 384
        assert all(isinstance(x, float) for x in features.embedding)

    @pytest.mark.asyncio
    async def test_token_estimation(self):
        """Test token count estimation."""
        analyzer = QueryAnalyzer()

        # Short query
        short_features = await analyzer.analyze("Hello world")
        assert short_features.token_count < 10

        # Long query
        long_text = " ".join(["word"] * 100)
        long_features = await analyzer.analyze(long_text)
        assert long_features.token_count > 100

    @pytest.mark.asyncio
    async def test_complexity_scoring_length_factor(self):
        """Test complexity increases with length."""
        analyzer = QueryAnalyzer()

        short = await analyzer.analyze("Hi")
        medium = await analyzer.analyze(" ".join(["word"] * 30))
        long = await analyzer.analyze(" ".join(["word"] * 60))

        assert short.complexity_score < medium.complexity_score
        assert medium.complexity_score <= long.complexity_score

    @pytest.mark.asyncio
    async def test_complexity_scoring_technical_terms(self):
        """Test complexity increases with technical terms."""
        analyzer = QueryAnalyzer()

        simple = await analyzer.analyze("Tell me a story")
        technical = await analyzer.analyze(
            "Implement algorithm for SQL query optimization with REST API"
        )

        assert technical.complexity_score > simple.complexity_score

    @pytest.mark.asyncio
    async def test_complexity_scoring_multiple_questions(self):
        """Test complexity increases with multiple questions."""
        analyzer = QueryAnalyzer()

        single = await analyzer.analyze("What is AI?")
        multiple = await analyzer.analyze(
            "What is AI? How does it work? What are its applications? What are the risks?"
        )

        assert multiple.complexity_score > single.complexity_score

    @pytest.mark.asyncio
    async def test_complexity_scoring_requirements(self):
        """Test complexity increases with detailed requirements."""
        analyzer = QueryAnalyzer()

        vague = await analyzer.analyze("Build a website")
        detailed = await analyzer.analyze(
            "Build a website that must support authentication, should handle 1000 users, "
            "and needs to ensure GDPR compliance in three steps"
        )

        assert detailed.complexity_score > vague.complexity_score

    @pytest.mark.asyncio
    async def test_complexity_score_bounds(self):
        """Test complexity score stays within 0.0-1.0 bounds."""
        analyzer = QueryAnalyzer()

        # Very simple query
        simple = await analyzer.analyze("Hi")
        assert 0.0 <= simple.complexity_score <= 1.0

        # Very complex query
        complex_text = """
        Implement a distributed microservices architecture with the following requirements:
        Must support horizontal scaling, should handle failover, needs to ensure ACID transactions.
        First, design the API schema. Second, implement service mesh. Third, configure load balancing.
        Include OAuth2 authentication, REST endpoints, SQL database optimization, and HTTP/2 protocol.
        Prove correctness using formal methods and verify performance with theorem proving.
        """
        complex_query = await analyzer.analyze(complex_text)
        assert 0.0 <= complex_query.complexity_score <= 1.0

    @pytest.mark.asyncio
    async def test_complexity_with_many_technical_patterns(self):
        """Test complexity calculation with many technical patterns."""
        analyzer = QueryAnalyzer()

        # Query with many technical patterns
        technical_text = """
        Write a function to implement a class-based algorithm for optimization of complexity
        analysis. Use SQL queries via API with HTTP REST endpoints and JSON schema validation.
        Include performance measurements and theorem proving capabilities with equation solving.
        """
        result = await analyzer.analyze(technical_text)

        # Should have elevated complexity due to technical terms
        assert result.complexity_score >= 0.4

    @pytest.mark.asyncio
    async def test_complexity_with_many_requirements(self):
        """Test complexity calculation with many requirement indicators."""
        analyzer = QueryAnalyzer()

        # Query with many requirements
        requirement_text = """
        Write a function to implement a class-based algorithm for optimization of complexity
        with SQL API HTTP REST JSON requirements. First, you must implement authentication.
        Second, you should add logging. Third, you need to ensure security. You must verify
        compliance. You should guarantee uptime. You need to require encryption. Step 1 is
        design. Step 2 is implementation. Step 3 is testing. Step 4 is deployment.
        """
        result = await analyzer.analyze(requirement_text)

        # Should have high complexity due to requirements
        assert result.complexity_score >= 0.6
