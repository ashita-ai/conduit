"""Model Discovery Example - Automatic Provider Detection.

Demonstrates how Conduit automatically detects configured providers
and uses representative models without manual configuration.
"""

import asyncio
import os

from conduit.core.model_discovery import ModelDiscovery
from conduit.core.config import settings
from conduit.engines.router import Router
from conduit.core.models import Query


async def example_auto_discovery():
    """Example 1: Automatic model discovery from API keys."""
    print("=" * 60)
    print("Example 1: Auto-Discovery from API Keys")
    print("=" * 60)

    # ModelDiscovery automatically detects which providers have API keys
    discovery = ModelDiscovery(settings)

    # Get auto-discovered models
    models = discovery.get_models()
    print(f"\n✅ Auto-discovered {len(models)} models:")
    for model in models:
        print(f"  - {model}")

    # Get configured providers
    providers = discovery.get_providers()
    print(f"\n✅ Configured providers: {providers}")

    # Get models grouped by provider
    models_by_provider = discovery.get_models_by_provider()
    print("\n✅ Models by provider:")
    for provider, provider_models in models_by_provider.items():
        print(f"  {provider}:")
        for model in provider_models:
            print(f"    - {model}")


async def example_router_auto_discovery():
    """Example 2: Router with automatic discovery (default behavior)."""
    print("\n" + "=" * 60)
    print("Example 2: Router with Auto-Discovery (Default)")
    print("=" * 60)

    # Router automatically uses ModelDiscovery when models=None
    # This is the recommended approach!
    # router = Router()  # No models argument = auto-discovery
    # (Commented out to avoid downloading sentence transformer model in example)

    print("\n✅ Router(models=None) will use auto-discovered models")
    print("   It detects providers from your API keys and uses 3 models per provider")
    print("   Example: If you have OPENAI_API_KEY and ANTHROPIC_API_KEY set:")
    print("     - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo")
    print("     - Anthropic: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus")

    # Example routing (commented out as it requires API keys)
    # query = Query(text="What is machine learning?")
    # decision = await router.route(query)
    # print(f"\n🎯 Routed to: {decision.selected_model}")


async def example_explicit_models():
    """Example 3: Router with explicit model list (override auto-discovery)."""
    print("\n" + "=" * 60)
    print("Example 3: Explicit Model List Override")
    print("=" * 60)

    # You can still provide explicit models to override auto-discovery
    custom_models = [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
    ]

    # router = Router(models=custom_models)
    # (Commented out to avoid downloading sentence transformer model)
    print(f"\n✅ Router(models={custom_models}) overrides auto-discovery")
    print("   Use this when you want explicit control over the model pool")


async def example_yaml_config():
    """Example 4: YAML config file override."""
    print("\n" + "=" * 60)
    print("Example 4: YAML Config Override")
    print("=" * 60)

    # Create a sample YAML config (models.yaml)
    yaml_content = """# Conduit Model Configuration
models:
  - gpt-4o-mini
  - claude-3-5-sonnet-20241022
  - gemini-1.5-flash
"""

    yaml_path = "models.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ Created {yaml_path}:")
    print(yaml_content)

    # Router will use YAML config if provided
    # router = Router(model_config_path=yaml_path)
    # (Commented out to avoid downloading sentence transformer model)
    print(f"✅ Router(model_config_path='{yaml_path}') uses YAML config")
    print("   YAML takes precedence over auto-discovery")

    # Clean up
    os.remove(yaml_path)
    print(f"✅ Cleaned up {yaml_path}")


async def example_comparison():
    """Example 5: Compare different configuration approaches."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration Approach Comparison")
    print("=" * 60)

    print("\n1️⃣  Auto-Discovery (Recommended):")
    print("   router = Router()")
    print("   → Detects providers from API keys")
    print("   → Uses 3 representative models per provider")
    print("   → Zero configuration needed")
    print("   → Easy to add new providers (just set API key)")

    print("\n2️⃣  Explicit Models:")
    print('   router = Router(models=["gpt-4o-mini", "claude-3-5-sonnet"])')
    print("   → Full control over model list")
    print("   → Good for testing specific models")
    print("   → Requires manual updates")

    print("\n3️⃣  YAML Config:")
    print('   router = Router(model_config_path="models.yaml")')
    print("   → Centralized configuration")
    print("   → Easy to share across environments")
    print("   → Version control friendly")

    print("\n💡 Best Practice:")
    print("   - Development: Use auto-discovery (just set API keys)")
    print("   - Production: Use YAML config for consistency")
    print("   - Testing: Use explicit models for controlled experiments")


async def main():
    """Run all examples."""
    print("\n🚀 Conduit Model Discovery Examples\n")

    # Note: These examples show the API usage
    # Actual routing requires valid API keys
    if not any(
        [
            settings.openai_api_key,
            settings.anthropic_api_key,
            settings.google_api_key,
        ]
    ):
        print("⚠️  No API keys configured - examples show usage patterns only")
        print("   To test with real providers, set API keys in .env:")
        print("     OPENAI_API_KEY=sk-...")
        print("     ANTHROPIC_API_KEY=sk-ant-...")
        print("     GOOGLE_API_KEY=...")
        print()

    await example_auto_discovery()
    await example_router_auto_discovery()
    await example_explicit_models()
    await example_yaml_config()
    await example_comparison()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
