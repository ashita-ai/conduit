"""LangChain integration example for Conduit.

This example shows how to use Conduit as a custom LLM in LangChain chains.
Conduit automatically learns which model to use for each query, optimizing for
cost and quality.

Requirements:
    pip install langchain langchain-core

Example:
    >>> from conduit.engines.router import Router
    >>> from examples.integrations.langchain_integration import ConduitLangChainLLM
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
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackHandlerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
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
    executor: Optional[ModelExecutor] = Field(
        default=None, description="Model executor (auto-initialized if None)"
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for model calls"
    )

    def __init__(
        self,
        router: Router,
        executor: Optional[ModelExecutor] = None,
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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackHandlerForLLMRun] = None,
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
        # Merge kwargs with model_kwargs
        merged_kwargs = {**self.model_kwargs, **kwargs}

        # Create query
        query = Query(text=prompt)

        # Route query to optimal model
        decision = await self.router.route(query)

        # Execute query with selected model
        response = await self.executor.execute(
            model_id=decision.selected_model,
            prompt=prompt,
            **merged_kwargs,
        )

        return response.text

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackHandlerForLLMRun] = None,
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
    """Basic usage example."""
    print("=" * 80)
    print("LangChain Integration - Basic Usage")
    print("=" * 80)

    # Initialize router
    router = Router()
    llm = ConduitLangChainLLM(router)

    # Simple call
    response = await llm.ainvoke("What is 2+2?")
    print(f"\nResponse: {response}")

    # Show which model was selected
    query = Query(text="What is 2+2?")
    decision = await router.route(query)
    print(f"Selected model: {decision.selected_model}")


async def example_with_chains():
    """Example using LangChain chains."""
    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        print("\n" + "=" * 80)
        print("LangChain Integration - Using Chains")
        print("=" * 80)

        # Initialize router and LLM
        router = Router()
        llm = ConduitLangChainLLM(router)

        # Create prompt template
        prompt = PromptTemplate.from_template(
            "Explain {topic} in simple terms suitable for a beginner."
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run chain
        result = await chain.ainvoke({"topic": "quantum computing"})
        print(f"\nResult: {result['text']}")

        # Show routing decision
        query = Query(text=prompt.format(topic="quantum computing"))
        decision = await router.route(query)
        print(f"\nSelected model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")

    except ImportError:
        print("\nLangChain not installed. Install with: pip install langchain")


async def example_with_preferences():
    """Example showing how Conduit learns preferences over time."""
    print("\n" + "=" * 80)
    print("LangChain Integration - Learning Preferences")
    print("=" * 80)

    router = Router()
    llm = ConduitLangChainLLM(router)

    # Run multiple queries - Conduit learns which models work best
    queries = [
        "What is Python?",
        "Write a hello world program",
        "Explain machine learning",
        "What is 2+2?",
        "Summarize quantum computing",
    ]

    print("\nRunning queries (Conduit learns optimal routing)...")
    for i, query_text in enumerate(queries, 1):
        response = await llm.ainvoke(query_text)
        decision = await router.route(Query(text=query_text))
        print(f"\nQuery {i}: {query_text[:50]}...")
        print(f"  Model: {decision.selected_model}")
        print(f"  Response length: {len(response)} chars")

    # Show statistics
    stats = router.hybrid_router.get_stats()
    print(f"\nRouter Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Model selections: {stats.get('arm_pulls', {})}")


async def main():
    """Run all examples."""
    await example_basic_usage()
    await example_with_chains()
    await example_with_preferences()


if __name__ == "__main__":
    asyncio.run(main())

