"""Model Discovery - See what models Conduit can route to.

Demonstrates how to discover which models are available in your
Conduit configuration based on your API keys.
"""

import os


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "Groq": bool(os.getenv("GROQ_API_KEY")),
    }


def get_models_by_provider() -> dict[str, list[dict]]:
    """Get available models organized by provider."""
    return {
        "openai": [
            {"id": "gpt-4o", "quality": 0.95, "cost_per_1k": 0.0025},
            {"id": "gpt-4o-mini", "quality": 0.85, "cost_per_1k": 0.00015},
            {"id": "gpt-4-turbo", "quality": 0.92, "cost_per_1k": 0.01},
            {"id": "gpt-3.5-turbo", "quality": 0.75, "cost_per_1k": 0.0005},
        ],
        "anthropic": [
            {"id": "claude-3-5-sonnet-20241022", "quality": 0.95, "cost_per_1k": 0.003},
            {"id": "claude-3-5-haiku-20241022", "quality": 0.85, "cost_per_1k": 0.00025},
            {"id": "claude-3-opus-20240229", "quality": 0.97, "cost_per_1k": 0.015},
        ],
        "google": [
            {"id": "gemini-1.5-pro", "quality": 0.92, "cost_per_1k": 0.00125},
            {"id": "gemini-1.5-flash", "quality": 0.85, "cost_per_1k": 0.000075},
        ],
        "groq": [
            {"id": "llama-3.1-70b-versatile", "quality": 0.88, "cost_per_1k": 0.00059},
            {"id": "llama-3.1-8b-instant", "quality": 0.75, "cost_per_1k": 0.00005},
            {"id": "mixtral-8x7b-32768", "quality": 0.82, "cost_per_1k": 0.00024},
        ],
    }


def main():
    print("Conduit Model Discovery\n")

    # 1. Check API keys
    print("=" * 60)
    print("API Key Status")
    print("=" * 60)

    api_keys = check_api_keys()
    available_providers = []

    for provider, has_key in api_keys.items():
        status = "✅" if has_key else "❌"
        print(f"  {status} {provider}")
        if has_key:
            available_providers.append(provider.lower())

    if not available_providers:
        print("\n⚠️  No API keys found!")
        print("   Set environment variables to enable providers:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        print("   - GROQ_API_KEY")

    # 2. All supported models
    print("\n" + "=" * 60)
    print("All Supported Models")
    print("=" * 60)

    all_models = get_models_by_provider()
    total_count = sum(len(models) for models in all_models.values())
    print(f"Total: {total_count} models across {len(all_models)} providers\n")

    for provider, models in all_models.items():
        is_available = provider in available_providers
        status = "✅" if is_available else "🔒"
        print(f"{status} {provider.upper()} ({len(models)} models)")

        for model in models:
            print(f"   - {model['id']}")
            print(f"     Quality: {model['quality']:.0%}, Cost: ${model['cost_per_1k']:.5f}/1K tokens")

    # 3. Your available models
    print("\n" + "=" * 60)
    print("Your Available Models")
    print("=" * 60)

    if available_providers:
        print(f"Based on your API keys, you can use:\n")
        your_models = []

        for provider in available_providers:
            for model in all_models.get(provider, []):
                your_models.append({**model, "provider": provider})

        # Sort by quality
        your_models.sort(key=lambda x: x["quality"], reverse=True)

        for model in your_models:
            print(f"  {model['id']} ({model['provider']})")
            print(f"    Quality: {model['quality']:.0%}, Cost: ${model['cost_per_1k']:.5f}/1K tokens")
    else:
        print("  No models available - add API keys to .env")

    # 4. Budget recommendations
    print("\n" + "=" * 60)
    print("Budget Model Recommendations")
    print("=" * 60)
    print("High quality + Low cost models:\n")

    budget_models = []
    for provider, models in all_models.items():
        for model in models:
            # Quality > 80% and cost < $0.001/1K tokens
            if model["quality"] >= 0.80 and model["cost_per_1k"] < 0.001:
                budget_models.append({**model, "provider": provider})

    budget_models.sort(key=lambda x: (x["quality"], -x["cost_per_1k"]), reverse=True)

    for model in budget_models[:5]:
        available = "✅" if model["provider"] in available_providers else "🔒"
        print(f"  {available} {model['id']}")
        print(f"     Quality: {model['quality']:.0%}, Cost: ${model['cost_per_1k']:.5f}/1K")

    print("\n" + "=" * 60)
    print("Usage Tips")
    print("=" * 60)
    print("1. Conduit automatically routes to optimal models based on query")
    print("2. Add more API keys to expand your routing options")
    print("3. Use constraints to control cost/quality tradeoffs")
    print("4. The bandit learns which models work best for your workload")


if __name__ == "__main__":
    main()
