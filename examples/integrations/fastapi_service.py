"""FastAPI service integration example for Conduit.

This example shows how to build a production FastAPI service with Conduit's
ML-powered routing. It demonstrates:
- REST API endpoints for query routing
- Async request handling
- Feedback submission for learning
- Health checks and metrics

Requirements:
    pip install fastapi uvicorn (included in conduit[all])

Run the server:
    uv run uvicorn examples.integrations.fastapi_service:app --reload

Or run the demo (starts server + sample requests):
    uv run python examples/integrations/fastapi_service.py --demo

Endpoints:
    POST /query - Route and execute a query
    POST /feedback - Submit feedback for learning
    GET /health - Health check
    GET /stats - Router statistics
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any

# Check for fastapi dependency
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    print("FastAPI integration requires fastapi.")
    print("Install with: pip install fastapi uvicorn")
    sys.exit(0)

from pydantic import BaseModel, Field

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router
from conduit.feedback import FeedbackCollector, FeedbackEvent, InMemoryFeedbackStore

# Global router, executor, and feedback collector (initialized in lifespan)
router: Router | None = None
executor: ModelExecutor | None = None
collector: FeedbackCollector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global router, executor, collector

    print("Initializing Conduit router...")
    router = Router()
    executor = ModelExecutor()

    # Initialize feedback collector with in-memory store
    # For production, use RedisFeedbackStore or PostgresFeedbackStore
    store = InMemoryFeedbackStore()
    collector = FeedbackCollector(router, store=store)
    print("Conduit router ready with feedback collection")

    yield

    # Cleanup
    if router:
        await router.close()
    print("Conduit router closed")


app = FastAPI(
    title="Conduit API",
    description="ML-powered LLM routing service",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response Models


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    text: str = Field(..., description="Query text", min_length=1)
    user_id: str | None = Field(None, description="Optional user ID for tracking")
    max_cost: float | None = Field(None, description="Maximum cost budget", ge=0.0)
    max_latency: float | None = Field(None, description="Maximum latency in seconds", ge=0.0)
    min_quality: float | None = Field(
        None, description="Minimum quality threshold", ge=0.0, le=1.0
    )
    optimize_for: str = Field(
        default="balanced",
        description="Optimization priority: balanced, quality, cost, speed",
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query_id: str = Field(..., description="Unique query ID")
    model_used: str = Field(..., description="Model that processed the query")
    response: str = Field(..., description="Model response text")
    confidence: float = Field(..., description="Routing confidence score")
    cost: float = Field(..., description="Query cost in dollars")
    latency: float = Field(..., description="Response latency in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint.

    Supports pluggable signal types via the signal_type + payload pattern.
    See docs/FEEDBACK_INTEGRATION.md for available signal types.
    """

    query_id: str = Field(..., description="Query ID to provide feedback for")
    signal_type: str = Field(
        default="thumbs",
        description="Feedback type: thumbs, rating, task_success, quality_score",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Signal-specific data (e.g., {'value': 'up'} for thumbs)",
    )


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    router_ready: bool
    algorithm: str


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    total_queries: int
    current_phase: str
    cache_stats: dict[str, Any] | None


# Endpoints


@app.post("/query", response_model=QueryResponse)
async def route_query(request: QueryRequest) -> QueryResponse:
    """Route and execute a query using Conduit's ML routing.

    Conduit automatically selects the optimal model based on:
    - Query complexity and features
    - Historical performance data
    - User preferences and constraints
    """
    if not router or not executor:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Build constraints
    constraints = None
    if request.max_cost or request.max_latency or request.min_quality:
        constraints = QueryConstraints(
            max_cost=request.max_cost,
            max_latency=request.max_latency,
            min_quality=request.min_quality,
        )

    # Build preferences
    preferences = UserPreferences(optimize_for=request.optimize_for)  # type: ignore[arg-type]

    # Create query
    query = Query(
        text=request.text,
        user_id=request.user_id,
        constraints=constraints,
        preferences=preferences,
    )

    # Route query
    decision = await router.route(query)

    # Execute with selected model
    from pydantic import BaseModel as PydanticBaseModel

    class TextOutput(PydanticBaseModel):
        text: str

    try:
        response = await executor.execute(
            model=decision.selected_model,
            prompt=query.text,
            result_type=TextOutput,
            query_id=query.id,
            timeout=request.max_latency or 60.0,
        )

        # Parse response
        import json

        response_data = json.loads(response.text)
        response_text = response_data.get("text", response.text)

        # Track query for delayed feedback
        if collector:
            await collector.track(decision, cost=response.cost, latency=response.latency)

        return QueryResponse(
            query_id=decision.query_id,  # Use decision.query_id for feedback tracking
            model_used=decision.selected_model,
            response=response_text,
            confidence=decision.confidence,
            cost=response.cost,
            latency=response.latency,
            metadata={
                "reasoning": decision.reasoning,
                "fallback_chain": decision.fallback_chain,
                **decision.metadata,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution failed: {e}") from e


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback to improve Conduit's routing decisions.

    Feedback helps Conduit learn which models work best for different
    query types. Supported signal types:

    - thumbs: {"value": "up"} or {"value": "down"}
    - rating: {"rating": 1-5}
    - task_success: {"success": true/false}
    - quality_score: {"score": 0.0-1.0}
    """
    if not collector:
        raise HTTPException(status_code=503, detail="Feedback collector not initialized")

    event = FeedbackEvent(
        query_id=request.query_id,
        signal_type=request.signal_type,
        payload=request.payload,
    )

    try:
        result = await collector.record(event)
        return {
            "status": "recorded",
            "query_id": request.query_id,
            "signal_type": request.signal_type,
            "reward": result.reward if result.reward is not None else None,
            "message": result.message if hasattr(result, "message") else "Feedback recorded",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}") from e


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health."""
    return HealthResponse(
        status="healthy" if router else "initializing",
        router_ready=router is not None,
        algorithm=router.algorithm if router else "unknown",
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get router statistics."""
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    stats = router.hybrid_router.get_stats()

    return StatsResponse(
        total_queries=stats.get("total_queries", 0),
        current_phase=stats.get("current_phase", "unknown"),
        cache_stats=router.get_cache_stats(),
    )


# Demo mode


async def demo_requests():
    """Run demo requests against the API."""
    try:
        import httpx
    except ImportError:
        print("Demo requires httpx. Install with: pip install httpx")
        return

    base_url = "http://127.0.0.1:8000"

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Wait for server to start
        print("Waiting for server to start...")
        for _ in range(10):
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(0.5)

        print("\n" + "=" * 80)
        print("FastAPI + Conduit Demo")
        print("=" * 80)

        # Health check
        print("\n[1] Health Check")
        response = await client.get(f"{base_url}/health")
        print(f"    Status: {response.json()}")

        # Query examples (using simple queries for fast demo)
        queries = [
            {"text": "What is 2+2?", "optimize_for": "cost"},
            {"text": "What is the capital of France?", "optimize_for": "quality"},
            {"text": "Translate 'hello' to French", "optimize_for": "speed"},
        ]

        for i, query in enumerate(queries, 2):
            print(f"\n[{i}] Query: {query['text'][:50]}...")
            response = await client.post(f"{base_url}/query", json=query)

            if response.status_code == 200:
                data = response.json()
                print(f"    Model: {data['model_used']}")
                print(f"    Confidence: {data['confidence']:.2f}")
                print(f"    Cost: ${data['cost']:.6f}")
                print(f"    Response: {data['response'][:80]}...")
            else:
                print(f"    Error: {response.text}")

        # Stats
        print(f"\n[{len(queries) + 2}] Router Stats")
        response = await client.get(f"{base_url}/stats")
        print(f"    Stats: {response.json()}")


def run_demo():
    """Run the demo with server."""
    import subprocess
    import time

    print("Starting FastAPI server...")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "examples.integrations.fastapi_service:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        time.sleep(4)  # Wait for server to start (increased from 2s)
        asyncio.run(demo_requests())
    finally:
        server.terminate()
        server.wait()
        print("\nServer stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conduit FastAPI Service")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo mode with sample requests"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start the server (default)"
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        # Default: just print info about the service
        print("=" * 80)
        print("Conduit FastAPI Service")
        print("=" * 80)
        print("\nThis example demonstrates a production FastAPI service with Conduit.")
        print("\nTo start the server:")
        print("    uv run uvicorn examples.integrations.fastapi_service:app --reload")
        print("\nOr run the demo (starts server + sample requests):")
        print("    uv run python examples/integrations/fastapi_service.py --demo")
        print("\nEndpoints:")
        print("    POST /query    - Route and execute a query")
        print("    POST /feedback - Submit feedback for learning")
        print("    GET  /health   - Health check")
        print("    GET  /stats    - Router statistics")
        print("\nAPI docs available at: http://127.0.0.1:8000/docs")
