"""PCA-Enabled Routing - Using reduced 67-dim features for faster convergence.

This example demonstrates routing with PCA enabled.

Prerequisites:
  1. Run pca_setup.py first to create models/pca.pkl
  2. Set USE_PCA=true in .env
"""

import asyncio

from conduit.core.models import Query
from conduit.engines.router import Router


async def main():
    """Demonstrate PCA-enabled routing."""

    print("="*60)
    print("PCA-Enabled Routing Example")
    print("="*60)
    print()

    # Initialize router (will use PCA from settings)
    print("Initializing router with PCA...")
    router = Router()

    print(f"  Analyzer feature dim: {router.analyzer.feature_dim}")
    print(f"  PCA enabled: {router.analyzer.use_pca}")

    if router.analyzer.use_pca:
        print(f"  PCA dimensions: {router.analyzer.pca_dimensions}")
        print("  ✓ Router will use 67-dim features (75% sample reduction)")
    else:
        print("  ⚠ PCA not enabled - using full 387-dim features")
        print("  Run pca_setup.py and set USE_PCA=true in .env")
    print()

    # Route some queries
    queries = [
        "What is 2+2?",
        "Explain quantum physics",
        "Write a Python function for Fibonacci",
        "Summarize the American Revolution",
        "Calculate the derivative of x^2",
    ]

    print("Routing queries...")
    print()

    for i, query_text in enumerate(queries, 1):
        query = Query(text=query_text)
        decision = await router.route(query)

        print(f"{i}. Query: \"{query_text}\"")
        print(f"   Model: {decision.selected_model}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Feature dims: {len(decision.features.embedding)} "
              f"({'PCA' if router.analyzer.use_pca else 'Full'})")
        print()

    # Show convergence benefit
    print("="*60)
    print("Convergence Benefit")
    print("="*60)
    print()
    print("With PCA (67 dims):")
    print("  • Needs ~1,000 samples per model for convergence")
    print("  • 17 models × 1,000 = 17,000 total queries")
    print("  • Faster matrix operations (smaller A matrices)")
    print()
    print("Without PCA (387 dims):")
    print("  • Needs ~4,000 samples per model for convergence")
    print("  • 17 models × 4,000 = 68,000 total queries")
    print("  • Slower matrix inversions")
    print()
    print("Result: 75% faster learning with PCA!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
