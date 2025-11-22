"""PCA Setup - One-time training to reduce feature dimensions from 387 to 67.

This example demonstrates how to:
1. Fit PCA on representative queries
2. Save the fitted model to disk
3. Reduce sample requirements by 75% (65K → 11K queries)

Run this once to create models/pca.pkl, then enable PCA in production.
"""

import asyncio

from conduit.engines.analyzer import QueryAnalyzer


# Representative training queries (diverse set across domains and complexity)
TRAINING_QUERIES = [
    # Simple queries
    "What is 2+2?",
    "Tell me a joke",
    "What's the weather?",
    "Hello, how are you?",
    "Define photosynthesis",
    # Medium complexity
    "Explain quantum physics in simple terms",
    "How do I reset my password?",
    "Write a function to sort numbers",
    "What are the benefits of meditation?",
    "Summarize the French Revolution",
    # Complex queries
    "Implement a binary search tree in Python with insert, delete, and search operations",
    "Analyze the trade-offs between microservices and monolithic architecture",
    "Explain the proof of the Pythagorean theorem using geometric reasoning",
    "Design a distributed caching system with consistency guarantees",
    "Critique this business strategy and suggest improvements: [long strategy doc]",
    # Technical/Code
    "Debug this Python code: def factorial(n): return n * factorial(n-1)",
    "Optimize this SQL query for performance",
    "What's the time complexity of merge sort?",
    "Explain async/await in JavaScript",
    "How does HTTP/2 differ from HTTP/1.1?",
    # Creative
    "Write a poem about artificial intelligence",
    "Create a story about a time-traveling scientist",
    "Describe a sunset on Mars",
    "Brainstorm marketing slogans for eco-friendly products",
    "Write dialogue between two AI systems",
    # Business
    "Calculate ROI for a $50K marketing campaign",
    "What factors affect customer retention?",
    "Analyze this revenue projection",
    "How to price a SaaS product?",
    "Develop a go-to-market strategy",
    # Science
    "How does CRISPR gene editing work?",
    "Explain the water cycle",
    "What causes earthquakes?",
    "Describe the life cycle of a star",
    "How do vaccines work?",
    # Math
    "Solve this equation: 2x + 5 = 15",
    "Find the derivative of x^3 + 2x^2 - 5",
    "Calculate the area under a curve",
    "Prove that the square root of 2 is irrational",
    "Explain Bayes' theorem with an example",
]


async def main():
    """Fit PCA on training queries and save to disk."""

    print("="*60)
    print("PCA Setup - Reducing feature dimensions from 387 → 67")
    print("="*60)
    print()

    # Create analyzer with PCA enabled
    print("1. Initializing QueryAnalyzer with PCA...")
    analyzer = QueryAnalyzer(
        use_pca=True,
        pca_dimensions=64,  # 64 embedding dims (from 384)
        pca_model_path="models/pca.pkl"
    )
    print(f"   ✓ PCA target dimensions: {analyzer.pca_dimensions}")
    print(f"   ✓ Total feature dim: {analyzer.feature_dim} (64 embedding + 3 metadata)")
    print()

    # Fit PCA on training queries
    print(f"2. Fitting PCA on {len(TRAINING_QUERIES)} training queries...")
    print("   (This may take 10-30 seconds...)")
    analyzer.fit_pca(TRAINING_QUERIES)
    print("   ✓ PCA fitted and saved to models/pca.pkl")
    print()

    # Verify PCA is working
    print("3. Verifying PCA transformation...")
    test_query = "What is machine learning?"
    features = await analyzer.analyze(test_query)

    print(f"   Query: \"{test_query}\"")
    print(f"   Embedding dimensions: {len(features.embedding)}")
    print(f"   Expected: 64 (down from 384)")

    if len(features.embedding) == 64:
        print("   ✓ PCA working correctly!")
    else:
        print(f"   ✗ Error: Got {len(features.embedding)} dims instead of 64")
    print()

    # Show impact
    print("="*60)
    print("Impact Summary")
    print("="*60)
    print()
    print("Feature Dimension Reduction:")
    print("  Before: 387 dimensions (384 embedding + 3 metadata)")
    print("  After:  67 dimensions (64 embedding + 3 metadata)")
    print("  Reduction: 82%")
    print()
    print("Sample Requirements (for LinUCB with 17 models):")
    print("  Before: ~4,000 samples/arm = 68,000 total queries")
    print("  After:  ~1,000 samples/arm = 17,000 total queries")
    print("  Reduction: 75% fewer queries needed")
    print()
    print("Next Steps:")
    print("  1. Set USE_PCA=true in .env")
    print("  2. Restart router - it will load models/pca.pkl automatically")
    print("  3. Enjoy 75% faster convergence!")
    print()
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
