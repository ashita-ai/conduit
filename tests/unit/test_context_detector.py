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
            query_text="Write a Python function to sort a list"
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
            query_text="Write a story about a brave knight"
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
            query_text="Analyze the proof of the Pythagorean theorem"
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
            query_text="Explain how photosynthesis works"
        )

        context = detector.detect_from_features(features)
        assert context == "analysis"

    def test_detect_from_features_analysis_business(self):
        """Test analysis context detection from business domain."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=35,
            complexity_score=0.6,
            query_text="Analyze market trends for Q4"
        )

        context = detector.detect_from_features(features)
        assert context == "analysis"

    def test_detect_from_features_general_fallback(self):
        """Test fallback to simple_qa for general queries."""
        detector = ContextDetector()
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=10,
            complexity_score=0.3,
            query_text="What is the weather today?"
        )

        context = detector.detect_from_features(features)
        assert context == "simple_qa"

    def test_detect_from_text_code(self):
        """Test code context detection from text."""
        detector = ContextDetector()
        context = detector.detect_from_text("Write a Python function to implement binary search")
        assert context == "code"

    def test_detect_from_text_creative(self):
        """Test creative context detection from text."""
        detector = ContextDetector()
        context = detector.detect_from_text("Write a poem about the ocean")
        assert context == "creative"

    def test_detect_from_text_analysis(self):
        """Test analysis context detection from text."""
        detector = ContextDetector()
        context = detector.detect_from_text("Analyze the causes of climate change")
        assert context == "analysis"

    def test_detect_from_text_simple_qa(self):
        """Test simple_qa fallback for general queries."""
        detector = ContextDetector()
        context = detector.detect_from_text("What time is it?")
        assert context == "simple_qa"
