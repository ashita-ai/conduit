"""LangChain integration example for Conduit.

Use Conduit as the LLM backend for LangChain chains and agents. Conduit automatically
routes each request to the optimal model based on query complexity and learned patterns.

What this demonstrates:
    1. Direct LLM calls - Use Conduit as a LangChain LLM
    2. Chain composition - Build LangChain chains with intelligent routing
    3. Cost tracking - See per-query costs and model selections
    4. Learning over time - Watch Conduit optimize routing decisions

Requirements:
    pip install langchain langchain-core

Run:
    uv run python examples/integrations/langchain_integration.py

Example usage in your code:
    >>> from conduit.engines.router import Router
    >>> from langchain.chains import LLMChain
    >>> from langchain.prompts import PromptTemplate
    >>>
    >>> router = Router()
    >>> llm = ConduitLangChainLLM(router)
    >>>
    >>> prompt = PromptTemplate.from_template("Explain {topic} in simple terms")
    >>> chain = LLMChain(llm=llm, prompt=prompt)
    >>>
    >>> result = await chain.ainvoke({"topic": "quantum computing"})
    >>> print(result["text"])
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

# Check for langchain_core dependency
try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.outputs import Generation, LLMResult
except ImportError:
    print("LangChain integration requires langchain-core.")
    print("Install with: pip install langchain langchain-core")
    sys.exit(0)

from pydantic import Field

from conduit.core.models import Query
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router


class ConduitLangChainLLM(BaseLLM):
    """LangChain LLM wrapper for Conduit.

    This class allows Conduit to be used as a drop-in replacement for
    LangChain's LLM classes. Conduit automatically selects the optimal model
    for each query based on learned patterns.

    Attributes:
        router: Conduit instance
        executor: Model executor for running queries
        model_kwargs: Additional kwargs to pass to model calls
    """

    router: Router = Field(description="Conduit instance")
    executor: ModelExecutor | None = Field(
        default=None, description="Model executor (auto-initialized if None)"
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for model calls"
    )

    def __init__(
        self,
        router: Router,
        executor: ModelExecutor | None = None,
        **kwargs: Any,
    ):
        """Initialize Conduit LangChain LLM wrapper.

        Args:
            router: Conduit instance (required)
            executor: Model executor (optional, auto-initialized if None)
            **kwargs: Additional arguments passed to BaseLLM
        """
        super().__init__(router=router, executor=executor, **kwargs)
        if self.executor is None:
            self.executor = ModelExecutor()

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "conduit_router"

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for a list of prompts.

        This is the core method required by BaseLLM.

        Args:
            prompts: List of input prompts
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for run tracking
            **kwargs: Additional arguments

        Returns:
            LLMResult containing generations for each prompt
        """
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate responses for a list of prompts.

        Args:
            prompts: List of input prompts
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for run tracking
            **kwargs: Additional arguments

        Returns:
            LLMResult containing generations for each prompt
        """
        generations = []
        for prompt in prompts:
            text = await self._acall(prompt, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # type: ignore[override]
        **kwargs: Any,
    ) -> str:
        """Synchronous call (not recommended for production).

        Note: This blocks the event loop. Use `_acall` for async operations.

        Args:
            prompt: Input prompt text
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for run tracking
            **kwargs: Additional arguments

        Returns:
            Model response text
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use it
                # This is a limitation - use async methods instead
                raise RuntimeError(
                    "Cannot use sync _call when event loop is running. "
                    "Use _acall or chain.ainvoke() instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._acall(prompt, stop, run_manager, **kwargs)
                )
            finally:
                loop.close()

        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # type: ignore[override]
        **kwargs: Any,
    ) -> str:
        """Asynchronous call (recommended).

        Args:
            prompt: Input prompt text
            stop: Stop sequences (not currently supported by Conduit)
            run_manager: Callback manager for run tracking
            **kwargs: Additional arguments merged with model_kwargs

        Returns:
            Model response text
        """
        from pydantic import BaseModel

        class TextResponse(BaseModel):
            text: str

        # Create query
        query = Query(text=prompt)

        # Route query to optimal model
        decision = await self.router.route(query)

        # Execute query with selected model
        response = await self.executor.execute(
            model=decision.selected_model,
            prompt=prompt,
            result_type=TextResponse,
            query_id=query.id,
        )

        # Parse response JSON to get text
        import json
        try:
            data = json.loads(response.text)
            return data.get("text", response.text)
        except json.JSONDecodeError:
            return response.text

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # type: ignore[override]
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream responses (not currently supported).

        Conduit doesn't support streaming yet. This yields the full
        response as a single chunk.

        Args:
            prompt: Input prompt text
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments

        Yields:
            Response chunks (currently single chunk with full response)
        """
        response = await self._acall(prompt, stop, run_manager, **kwargs)
        yield response


async def example_basic_usage():
    """Direct LLM calls with automatic model selection."""
    print("=" * 80)
    print("1. Direct LLM Calls")
    print("=" * 80)
    print("\nConduit routes each query to the optimal model automatically.")

    router = Router()
    llm = ConduitLangChainLLM(router)

    # Demonstrate different complexity queries
    queries = [
        ("What is 2+2?", "simple"),
        ("Explain the implications of Godel's incompleteness theorems.", "complex"),
    ]

    for query_text, complexity in queries:
        print(f"\n[{complexity.upper()}] {query_text}")
        response = await llm.ainvoke(query_text)
        decision = await router.route(Query(text=query_text))
        print(f"Model: {decision.selected_model}")
        print(f"Response: {response[:120]}{'...' if len(response) > 120 else ''}")


async def example_with_chains():
    """Build LangChain chains with Conduit routing."""
    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        print("\n" + "=" * 80)
        print("2. LangChain Chains")
        print("=" * 80)
        print("\nConduit integrates seamlessly with LangChain chains and prompts.")

        router = Router()
        llm = ConduitLangChainLLM(router)

        # Template for explanations at different levels
        prompt = PromptTemplate.from_template(
            "Explain {topic} to a {audience}. Keep it {length}."
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run chain with different inputs
        examples = [
            {"topic": "recursion", "audience": "5-year-old", "length": "very short"},
            {"topic": "transformer architecture", "audience": "ML engineer", "length": "detailed"},
        ]

        for example in examples:
            print(f"\n[{example['audience'].upper()}] Explaining {example['topic']}")
            result = await chain.ainvoke(example)
            text = result["text"]
            print(f"Response: {text[:150]}{'...' if len(text) > 150 else ''}")

    except ImportError:
        print("\nLangChain not installed. Install with: pip install langchain")


async def example_with_preferences():
    """Watch Conduit learn and optimize routing over time."""
    print("\n" + "=" * 80)
    print("3. Learning Over Time")
    print("=" * 80)
    print("\nConduit uses bandit algorithms to learn which models work best.")
    print("Watch the model selection adapt as more queries are processed.\n")

    router = Router()
    llm = ConduitLangChainLLM(router)

    # Mix of simple and complex queries
    queries = [
        ("What is 5 * 7?", "math"),
        ("Explain gradient descent optimization.", "ml"),
        ("What color is the sky?", "trivia"),
        ("Compare and contrast REST vs GraphQL APIs.", "technical"),
        ("Hello!", "greeting"),
    ]

    print(f"{'Query':<45} {'Model':<30} {'Length'}")
    print("-" * 85)

    for query_text, category in queries:
        response = await llm.ainvoke(query_text)
        decision = await router.route(Query(text=query_text))
        display_query = f"[{category}] {query_text[:35]}{'...' if len(query_text) > 35 else ''}"
        print(f"{display_query:<45} {decision.selected_model:<30} {len(response):>5} chars")

    # Show statistics
    stats = router.hybrid_router.get_stats()
    print(f"\nTotal queries routed: {stats['total_queries']}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LangChain + Conduit Integration Demo")
    print("=" * 80)
    print("\nConduit provides intelligent model routing for LangChain applications.")
    print("Each query is automatically routed to the optimal LLM based on:")
    print("  - Query complexity and token count")
    print("  - Historical performance data")
    print("  - Cost/quality trade-offs")

    await example_basic_usage()
    await example_with_chains()
    await example_with_preferences()

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  1. ConduitLangChainLLM is a drop-in replacement for any LangChain LLM")
    print("  2. Works with chains, agents, and all LangChain components")
    print("  3. Use async methods (ainvoke, agenerate) for best performance")
    print("  4. Conduit learns from each query to improve routing over time")


if __name__ == "__main__":
    asyncio.run(main())

