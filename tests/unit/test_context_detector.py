"""Unit tests for ContextDetector."""

import pytest

from conduit.core.context_detector import ContextDetector
from conduit.core.models import QueryFeatures


class TestContextDetector:
    """Tests for context detection."""

    def test_detect_from_features_code(self):
        """Test code context detection from features."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.6,
            domain="code",
            domain_confidence=0.9,
        )

        context = detector.detect_from_features(features)
        assert context == "code"

    def test_detect_from_features_creative(self):
        """Test creative context detection from features."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=30,
            complexity_score=0.4,
            domain="creative",
            domain_confidence=0.85,
        )

        context = detector.detect_from_features(features)
        assert context == "creative"

    def test_detect_from_features_analysis_math(self):
        """Test analysis context detection from math domain."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=40,
            complexity_score=0.7,
            domain="math",
            domain_confidence=0.8,
        )

        context = detector.detect_from_features(features)
        assert context == "analysis"

    def test_detect_from_features_analysis_science(self):
        """Test analysis context detection from science domain."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=45,
            complexity_score=0.65,
            domain="science",
            domain_confidence=0.75,
        )

        context = detector.detect_from_features(features)
        assert context == "analysis"

    def test_detect_from_features_analysis_business(self):
        """Test analysis context detection from business domain."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=35,
            complexity_score=0.55,
            domain="business",
            domain_confidence=0.7,
        )

        context = detector.detect_from_features(features)
        assert context == "analysis"

    def test_detect_from_features_general_fallback(self):
        """Test general domain falls back to simple_qa."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=20,
            complexity_score=0.3,
            domain="general",
            domain_confidence=0.5,
        )

        context = detector.detect_from_features(features)
        assert context == "simple_qa"

    def test_detect_from_text_code(self):
        """Test code context detection from text."""
        detector = ContextDetector()

        assert detector.detect_from_text("Write a Python function") == "code"
        assert detector.detect_from_text("Debug this code") == "code"
        assert detector.detect_from_text("Implement an algorithm") == "code"

    def test_detect_from_text_creative(self):
        """Test creative context detection from text."""
        detector = ContextDetector()

        assert detector.detect_from_text("Write a creative story") == "creative"
        assert detector.detect_from_text("Write a poem about") == "creative"
        assert detector.detect_from_text("Imagine a world") == "creative"

    def test_detect_from_text_analysis(self):
        """Test analysis context detection from text."""
        detector = ContextDetector()

        assert detector.detect_from_text("Analyze the pros and cons") == "analysis"
        assert detector.detect_from_text("Explain why this happens") == "analysis"
        assert detector.detect_from_text("Compare these options") == "analysis"

    def test_detect_from_text_simple_qa_fallback(self):
        """Test simple_qa fallback for unknown queries."""
        detector = ContextDetector()

        assert detector.detect_from_text("What is the capital of France?") == "simple_qa"
        assert detector.detect_from_text("Hello") == "simple_qa"


