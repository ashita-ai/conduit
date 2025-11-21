"""Model Discovery - See what models you can use.

Demonstrates how to discover which models are supported by Conduit
and which ones you can actually use based on your API keys.
"""

from conduit.models import available_models, get_available_providers, supported_models


def main():
    print("Conduit Model Discovery\n")

    # 1. All models Conduit supports
    print("=" * 60)
    all_models = supported_models()
    print(f"All Supported Models ({len(all_models)} total from llm-prices.com)")
    print("=" * 60)

    providers = {}
    for model in all_models:
        if model.provider not in providers:
            providers[model.provider] = []
        providers[model.provider].append(model)

    for provider, models in sorted(providers.items()):
        print(f"\n{provider.upper()}: {len(models)} models")
        for model in models:
            cost_avg = (model.cost_per_input_token + model.cost_per_output_token) / 2
            print(f"  - {model.model_name}")
            print(f"    Quality: {model.expected_quality:.0%}, Cost: ${cost_avg*1000:.4f}/1K tokens")

    # 2. Models YOU can use (based on .env)
    print("\n" + "=" * 60)
    print("Your Available Models (based on .env API keys)")
    print("=" * 60)

    available = get_available_providers()
    if not available:
        print("⚠️  No API keys found in .env")
        print("   Add keys to .env to enable model routing")
        print("   See .env.example for all supported providers")
    else:
        print(f"✅ Providers configured: {', '.join(available)}\n")

        my_models = available_models()
        for model in my_models:
            cost_avg = (model.cost_per_input_token + model.cost_per_output_token) / 2
            print(f"  {model.model_id}")
            print(f"    Quality: {model.expected_quality:.0%}, Cost: ${cost_avg*1000:.4f}/1K tokens")

    # 3. Filtered models
    print("\n" + "=" * 60)
    print("High-Quality Budget Models (quality>85%, cost<$0.001/token)")
    print("=" * 60)

    good_cheap = supported_models(min_quality=0.85, max_cost=0.001)
    if good_cheap:
        for model in good_cheap:
            cost_avg = (model.cost_per_input_token + model.cost_per_output_token) / 2
            print(f"  {model.model_id}")
            print(f"    Quality: {model.expected_quality:.0%}, Cost: ${cost_avg*1000:.4f}/1K tokens")
    else:
        print("  No models match criteria")

    # 4. Provider-specific
    print("\n" + "=" * 60)
    print("OpenAI Models Only")
    print("=" * 60)

    openai_models = supported_models(providers=["openai"])
    for model in openai_models:
        cost_avg = (model.cost_per_input_token + model.cost_per_output_token) / 2
        print(f"  {model.model_name}: {model.expected_quality:.0%} quality, ${cost_avg*1000:.4f}/1K")

    print("\n" + "=" * 60)
    print("Usage Tips")
    print("=" * 60)
    print("1. Use supported_models() to see ALL models Conduit can route to")
    print("2. Use available_models() to see what YOU can use (based on .env)")
    print("3. Filter by quality, cost, or provider to find optimal models")
    print("4. Add more API keys to .env to expand your routing options")


if __name__ == "__main__":
    main()
