"""LlamaIndex integration example for Conduit.

Use Conduit as the LLM backend for LlamaIndex RAG pipelines. Conduit automatically
routes each query to the optimal model based on learned performance patterns.

What this demonstrates:
    1. Basic completion - Direct LLM calls with automatic model selection
    2. RAG pipeline - Vector search + Conduit routing for document Q&A
    3. Chat interface - Multi-turn conversations with optimal model per turn

Requirements:
    pip install llama-index llama-index-embeddings-openai

Run:
    uv run python examples/integrations/llamaindex_integration.py

Example usage in your code:
    >>> from conduit.engines.router import Router
    >>> from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    >>>
    >>> router = Router()
    >>> llm = ConduitLlamaIndexLLM(router)
    >>>
    >>> # Build RAG index
    >>> documents = SimpleDirectoryReader("data").load_data()
    >>> index = VectorStoreIndex.from_documents(documents)
    >>> query_engine = index.as_query_engine(llm=llm)
    >>>
    >>> # Query with automatic model selection
    >>> response = await query_engine.aquery("What is the main topic?")
    >>> print(response)
"""

import asyncio
import sys
from collections.abc import Sequence
from typing import Any

# Check for llama_index dependency
try:
    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
    )
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.llms import CustomLLM
    from llama_index.core.llms.callbacks import llm_completion_callback
except ImportError:
    print("LlamaIndex integration requires llama-index.")
    print("Install with: pip install llama-index llama-index-llms-openai")
    sys.exit(0)

from pydantic import Field

from conduit.core.models import Query
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router


class ConduitLlamaIndexLLM(CustomLLM):
    """LlamaIndex LLM wrapper for Conduit.

    This class allows Conduit to be used as a drop-in replacement for
    LlamaIndex's LLM classes. Conduit automatically selects the optimal model
    for each query based on learned patterns.

    Attributes:
        router: Conduit instance
        executor: Model executor for running queries
        context_window: Maximum context window size
        num_output: Maximum output tokens
    """

    router: Router = Field(description="Conduit instance")
    executor: ModelExecutor | None = Field(
        default=None, description="Model executor (auto-initialized if None)"
    )
    context_window: int = Field(default=128000, description="Maximum context window")
    num_output: int = Field(default=4096, description="Maximum output tokens")

    def __init__(
        self,
        router: Router,
        executor: ModelExecutor | None = None,
        context_window: int = 128000,
        num_output: int = 4096,
        callback_manager: CallbackManager | None = None,
        **kwargs: Any,
    ):
        """Initialize Conduit LlamaIndex LLM wrapper.

        Args:
            router: Conduit instance (required)
            executor: Model executor (optional, auto-initialized if None)
            context_window: Maximum context window size
            num_output: Maximum output tokens
            callback_manager: LlamaIndex callback manager
            **kwargs: Additional arguments passed to CustomLLM
        """
        super().__init__(
            router=router,
            executor=executor,
            context_window=context_window,
            num_output=num_output,
            callback_manager=callback_manager,
            **kwargs,
        )
        if self.executor is None:
            object.__setattr__(self, "executor", ModelExecutor())

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata for LlamaIndex."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name="conduit_router",
            is_chat_model=False,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronous completion.

        Handles both cases where event loop is running or not.

        Args:
            prompt: Input prompt text
            formatted: Whether prompt is pre-formatted
            **kwargs: Additional arguments

        Returns:
            CompletionResponse with model output
        """
        try:
            # Try to import nest_asyncio for nested event loop support
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass  # nest_asyncio not available, try other methods

        try:
            asyncio.get_running_loop()
            # Event loop is running - use thread pool to avoid nested loop issues
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.acomplete(prompt, formatted, **kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running event loop
            return asyncio.run(self.acomplete(prompt, formatted, **kwargs))

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronous completion (recommended).

        Args:
            prompt: Input prompt text
            formatted: Whether prompt is pre-formatted
            **kwargs: Additional arguments

        Returns:
            CompletionResponse with model output and metadata
        """
        # Create query
        query = Query(text=prompt)

        # Route query to optimal model
        decision = await self.router.route(query)

        # Execute query with selected model using PydanticAI
        # For raw text completion, we use a simple string response
        from pydantic import BaseModel

        class TextResponse(BaseModel):
            text: str

        response = await self.executor.execute(
            model=decision.selected_model,
            prompt=prompt,
            result_type=TextResponse,
            query_id=query.id,
        )

        # Parse the JSON response to get the text
        import json

        response_data = json.loads(response.text)
        response_text = response_data.get("text", response.text)

        return CompletionResponse(
            text=response_text,
            additional_kwargs={
                "model_used": decision.selected_model,
                "confidence": decision.confidence,
                "cost": response.cost,
                "latency": response.latency,
            },
        )

    @llm_completion_callback()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Synchronous chat completion.

        Handles both cases where event loop is running or not.
        """
        try:
            # Try to import nest_asyncio for nested event loop support
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass  # nest_asyncio not available, try other methods

        try:
            asyncio.get_running_loop()
            # Event loop is running - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.achat(messages, **kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running event loop
            return asyncio.run(self.achat(messages, **kwargs))

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Asynchronous chat completion.

        Converts chat messages to a single prompt and routes through Conduit.

        Args:
            messages: Sequence of chat messages
            **kwargs: Additional arguments

        Returns:
            ChatResponse with model output
        """
        # Convert messages to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            prompt_parts.append(f"{role}: {msg.content}")
        prompt = "\n".join(prompt_parts)

        # Use completion
        completion = await self.acomplete(prompt, **kwargs)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion.text),
            additional_kwargs=completion.additional_kwargs,
        )

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Stream completion (not currently supported).

        Conduit doesn't support streaming yet. Returns full response.
        """
        return self.complete(prompt, formatted, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        """Stream chat (not currently supported).

        Conduit doesn't support streaming yet. Returns full response.
        """
        return self.chat(messages, **kwargs)


async def example_basic_usage():
    """Basic completion with automatic model selection."""
    print("=" * 80)
    print("1. Basic Completion")
    print("=" * 80)
    print("\nConduit selects the optimal model for each query automatically.")

    router = Router()
    llm = ConduitLlamaIndexLLM(router)

    queries = [
        "What is 2 + 2?",  # Simple - likely routes to fast/cheap model
        "Explain the significance of the Turing test in AI history.",  # Complex - may route to stronger model
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = await llm.acomplete(query)
        print(f"Model: {response.additional_kwargs.get('model_used')}")
        print(f"Response: {response.text[:150]}{'...' if len(response.text) > 150 else ''}")
        print(f"Cost: ${response.additional_kwargs.get('cost', 0):.6f}")


async def example_with_rag():
    """RAG pipeline with vector search and intelligent routing."""
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError:
        print("\nSkipping RAG example - requires llama-index-embeddings-openai")
        print("Install with: pip install llama-index-embeddings-openai")
        return

    print("\n" + "=" * 80)
    print("2. RAG Pipeline (Retrieval-Augmented Generation)")
    print("=" * 80)
    print("\nConduit routes RAG queries to optimal models based on complexity.")

    router = Router()
    llm = ConduitLlamaIndexLLM(router)

    # Configure LlamaIndex to use Conduit
    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding()

    # Knowledge base about ML routing
    documents = [
        Document(
            text="Conduit is an ML-powered LLM routing system that reduces costs 30-50% "
            "while maintaining quality. It uses contextual bandit algorithms to learn "
            "which model works best for different query types over time."
        ),
        Document(
            text="Thompson Sampling is a Bayesian bandit algorithm that balances "
            "exploration (trying new models) and exploitation (using known-good models). "
            "It maintains probability distributions over expected rewards and samples "
            "from them to make decisions. Conduit uses it during cold start."
        ),
        Document(
            text="LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm "
            "that uses ridge regression to model the relationship between query features "
            "(embeddings, token count, complexity) and model performance. It's Conduit's "
            "primary algorithm after warm-up, achieving 30% faster convergence than UCB1."
        ),
        Document(
            text="The hybrid routing strategy in Conduit starts with Thompson Sampling "
            "for exploration during cold start, then transitions to LinUCB once enough "
            "data is collected. This combines robust cold-start with optimal convergence."
        ),
    ]

    print("\nBuilding vector index from 4 documents...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Use async queries to avoid event loop issues
    queries = [
        ("What is Conduit and what problem does it solve?", "overview"),
        ("How does Thompson Sampling balance exploration vs exploitation?", "algorithm"),
        ("Why does Conduit use LinUCB instead of simpler algorithms?", "technical"),
    ]

    for query_text, query_type in queries:
        print(f"\n[{query_type.upper()}] {query_text}")
        try:
            # Use async query to avoid event loop conflicts
            response = await query_engine.aquery(query_text)
            print(f"Response: {str(response)[:200]}...")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {str(e)[:100]}")


async def example_with_chat():
    """Multi-turn chat with context-aware routing."""
    print("\n" + "=" * 80)
    print("3. Multi-Turn Chat")
    print("=" * 80)
    print("\nConduit routes each turn independently based on complexity.")

    router = Router()
    llm = ConduitLlamaIndexLLM(router)

    # Simulate a conversation with varying complexity
    conversations = [
        {
            "messages": [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is 2 + 2?"),
            ],
            "description": "Simple math",
        },
        {
            "messages": [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is 2 + 2?"),
                ChatMessage(role="assistant", content="4"),
                ChatMessage(
                    role="user",
                    content="Now explain why neural networks can approximate any continuous function.",
                ),
            ],
            "description": "Complex follow-up (universal approximation theorem)",
        },
    ]

    for conv in conversations:
        print(f"\n[{conv['description'].upper()}]")
        user_msg = conv["messages"][-1].content
        print(f"User: {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")

        response = await llm.achat(conv["messages"])
        print(f"Model: {response.additional_kwargs.get('model_used')}")
        print(f"Response: {response.message.content[:150]}{'...' if len(response.message.content) > 150 else ''}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LlamaIndex + Conduit Integration Demo")
    print("=" * 80)
    print("\nConduit provides intelligent model routing for LlamaIndex applications.")
    print("Each query is automatically routed to the optimal LLM based on:")
    print("  - Query complexity and token count")
    print("  - Historical performance data")
    print("  - Cost/quality trade-offs")

    await example_basic_usage()
    await example_with_rag()
    await example_with_chat()

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  1. Use ConduitLlamaIndexLLM as a drop-in LLM replacement")
    print("  2. Use async methods (aquery, acomplete, achat) for best results")
    print("  3. Conduit learns from each query to improve routing over time")
    print("\nFor production, see: examples/integrations/fastapi_service.py")


if __name__ == "__main__":
    asyncio.run(main())
