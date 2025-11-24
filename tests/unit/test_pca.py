"""Tests for PCA dimensionality reduction in QueryAnalyzer."""

import pytest
import tempfile
from pathlib import Path

from conduit.engines.analyzer import QueryAnalyzer


class TestPCAIntegration:
    """Test PCA feature reduction functionality."""

    @pytest.mark.asyncio
    async def test_analyzer_without_pca(self):
        """Test analyzer produces 384-dim embeddings without PCA."""
        analyzer = QueryAnalyzer(use_pca=False)

        features = await analyzer.analyze("What is machine learning?")

        assert len(features.embedding) == 384
        assert analyzer.feature_dim == 387  # 384 + 3 metadata

    @pytest.mark.asyncio
    async def test_analyzer_with_pca_not_fitted_raises(self):
        """Test that using PCA without fitting raises error."""
        # Use non-existent path to prevent loading pre-fitted model
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QueryAnalyzer(
                use_pca=True,
                pca_dimensions=64,
                pca_model_path=str(Path(tmpdir) / "nonexistent.pkl")
            )

            with pytest.raises(RuntimeError, match="PCA is enabled but not fitted"):
                await analyzer.analyze("Test query")

    @pytest.mark.asyncio
    async def test_pca_fitting_requires_sufficient_queries(self):
        """Test that PCA fitting requires at least 100 queries."""
        analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)

        # Too few queries
        with pytest.raises(ValueError, match="at least 100 queries"):
            await analyzer.fit_pca(["query1", "query2", "query3"])

    @pytest.mark.asyncio
    async def test_pca_fitting_success(self):
        """Test successful PCA fitting on sufficient queries."""
        analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)

        # Generate 100+ diverse queries
        training_queries = [
            f"Query {i} about various topics" for i in range(150)
        ]

        # Should not raise
        await analyzer.fit_pca(training_queries)

        # PCA should now be fitted
        assert hasattr(analyzer.pca, "components_")
        assert analyzer.pca.n_components_ == 64

    @pytest.mark.asyncio
    async def test_analyzer_with_fitted_pca(self):
        """Test analyzer produces reduced dimensions after PCA fitting."""
        analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)

        # Fit PCA
        training_queries = [f"Training query {i}" for i in range(150)]
        await analyzer.fit_pca(training_queries)

        # Analyze query
        features = await analyzer.analyze("What is deep learning?")

        assert len(features.embedding) == 64  # Reduced from 384
        assert analyzer.feature_dim == 67  # 64 + 3 metadata

    @pytest.mark.asyncio
    async def test_pca_persistence(self):
        """Test PCA model save/load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = str(Path(tmpdir) / "test_pca.pkl")

            # Create and fit analyzer
            analyzer1 = QueryAnalyzer(
                use_pca=True,
                pca_dimensions=64,
                pca_model_path=pca_path
            )
            training_queries = [f"Query {i}" for i in range(150)]
            await analyzer1.fit_pca(training_queries)  # Saves to pca_path

            # Create new analyzer that loads the saved PCA
            analyzer2 = QueryAnalyzer(
                use_pca=True,
                pca_dimensions=64,
                pca_model_path=pca_path
            )

            # Should have loaded fitted PCA
            assert analyzer2.pca is not None
            assert hasattr(analyzer2.pca, "components_")

            # Both analyzers should produce same embeddings
            features1 = await analyzer1.analyze("Test query")
            features2 = await analyzer2.analyze("Test query")

            assert len(features1.embedding) == len(features2.embedding) == 64
            assert features1.embedding == features2.embedding

    def test_feature_dim_property(self):
        """Test feature_dim property returns correct dimensions."""
        # Without PCA
        analyzer1 = QueryAnalyzer(use_pca=False)
        assert analyzer1.feature_dim == 387  # 384 + 3

        # With PCA (64 dims)
        analyzer2 = QueryAnalyzer(use_pca=True, pca_dimensions=64)
        assert analyzer2.feature_dim == 67  # 64 + 3

        # With PCA (128 dims)
        analyzer3 = QueryAnalyzer(use_pca=True, pca_dimensions=128)
        assert analyzer3.feature_dim == 131  # 128 + 3

    @pytest.mark.asyncio
    async def test_pca_preserves_metadata_features(self):
        """Test that PCA only reduces embedding, not metadata features."""
        analyzer = QueryAnalyzer(use_pca=True, pca_dimensions=64)

        training_queries = [f"Query {i}" for i in range(150)]
        await analyzer.fit_pca(training_queries)

        features = await analyzer.analyze("What is Python?")

        # Should have all metadata fields
        assert features.token_count > 0
        assert 0.0 <= features.complexity_score <= 1.0
        assert features.domain in ["code", "math", "science", "business", "creative", "general"]
        assert 0.0 <= features.domain_confidence <= 1.0

        # Embedding should be reduced
        assert len(features.embedding) == 64

    def test_pca_not_enabled_when_use_pca_false(self):
        """Test that PCA is None when use_pca=False."""
        analyzer = QueryAnalyzer(use_pca=False)
        assert analyzer.pca is None
        assert not analyzer.use_pca

    @pytest.mark.asyncio
    async def test_pca_fit_requires_pca_enabled(self):
        """Test that fit_pca raises error when PCA not enabled."""
        analyzer = QueryAnalyzer(use_pca=False)

        with pytest.raises(ValueError, match="PCA not enabled"):
            await analyzer.fit_pca(["query1", "query2"])

    @pytest.mark.asyncio
    async def test_different_pca_dimensions(self):
        """Test PCA with different target dimensions."""
        dimensions_to_test = [32, 64, 128]

        for target_dim in dimensions_to_test:
            # Use temp path to prevent loading pre-fitted model
            # Need at least 300 queries for 128 dims (>= target_dim * 2)
            with tempfile.TemporaryDirectory() as tmpdir:
                pca_path = str(Path(tmpdir) / f"pca_{target_dim}.pkl")
                analyzer = QueryAnalyzer(
                    use_pca=True,
                    pca_dimensions=target_dim,
                    pca_model_path=pca_path
                )
                training_queries = [f"Query {i}" for i in range(300)]
                await analyzer.fit_pca(training_queries)

                features = await analyzer.analyze("Test query")

                assert len(features.embedding) == target_dim
                assert analyzer.feature_dim == target_dim + 3
