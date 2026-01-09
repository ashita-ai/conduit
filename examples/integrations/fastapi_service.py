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
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Check for fastapi dependency
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    logger.error("FastAPI integration requires fastapi.")
    logger.error("Install with: pip install fastapi uvicorn")
    sys.exit(0)

from pydantic import BaseModel, Field

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router

# Global router and executor (initialized in lifespan)
router: Router | None = None
executor: ModelExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global router, executor

    logger.info("Initializing Conduit router...")
    router = Router()
    executor = ModelExecutor()
    logger.info("Conduit router ready")

    yield

    # Cleanup
    if router:
        await router.close()
    logger.info("Conduit router closed")


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
    """Request model for feedback endpoint."""

    query_id: str = Field(..., description="Query ID to provide feedback for")
    model_id: str = Field(..., description="Model that was used")
    quality_score: float = Field(..., description="Quality rating 0.0-1.0", ge=0.0, le=1.0)
    met_expectations: bool = Field(default=True, description="Whether response met expectations")
    comments: str | None = Field(None, description="Optional feedback comments")


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

        return QueryResponse(
            query_id=query.id,
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
async def submit_feedback(request: FeedbackRequest) -> dict[str, str]:
    """Submit feedback to improve Conduit's routing decisions.

    Feedback helps Conduit learn which models work best for different
    query types. Both explicit ratings and implicit signals are used.
    """
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # For now, just acknowledge the feedback
    # In production, this would update the bandit algorithm
    return {
        "status": "received",
        "message": f"Feedback for query {request.query_id} recorded",
    }


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
        logger.error("Demo requires httpx. Install with: pip install httpx")
        return

    base_url = "http://127.0.0.1:8000"

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Wait for server to start
        logger.info("Waiting for server to start...")
        for _ in range(10):
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(0.5)

        logger.info("\n" + "=" * 80)
        logger.info("FastAPI + Conduit Demo")
        logger.info("=" * 80)

        # Health check
        logger.info("\n[1] Health Check")
        response = await client.get(f"{base_url}/health")
        logger.info(f"    Status: {response.json()}")

        # Query examples (using simple queries for fast demo)
        queries = [
            {"text": "What is 2+2?", "optimize_for": "cost"},
            {"text": "What is the capital of France?", "optimize_for": "quality"},
            {"text": "Translate 'hello' to French", "optimize_for": "speed"},
        ]

        for i, query in enumerate(queries, 2):
            logger.info(f"\n[{i}] Query: {query['text'][:50]}...")
            response = await client.post(f"{base_url}/query", json=query)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"    Model: {data['model_used']}")
                logger.info(f"    Confidence: {data['confidence']:.2f}")
                logger.info(f"    Cost: ${data['cost']:.6f}")
                logger.info(f"    Response: {data['response'][:80]}...")
            else:
                logger.error(f"    Error: {response.text}")

        # Stats
        logger.info(f"\n[{len(queries) + 2}] Router Stats")
        response = await client.get(f"{base_url}/stats")
        logger.info(f"    Stats: {response.json()}")


def run_demo():
    """Run the demo with server."""
    import subprocess
    import time

    logger.info("Starting FastAPI server...")
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
        logger.info("\nServer stopped")


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
        # Default: just log info about the service
        logger.info("=" * 80)
        logger.info("Conduit FastAPI Service")
        logger.info("=" * 80)
        logger.info("\nThis example demonstrates a production FastAPI service with Conduit.")
        logger.info("\nTo start the server:")
        logger.info("    uv run uvicorn examples.integrations.fastapi_service:app --reload")
        logger.info("\nOr run the demo (starts server + sample requests):")
        logger.info("    uv run python examples/integrations/fastapi_service.py --demo")
        logger.info("\nEndpoints:")
        logger.info("    POST /query    - Route and execute a query")
        logger.info("    POST /feedback - Submit feedback for learning")
        logger.info("    GET  /health   - Health check")
        logger.info("    GET  /stats    - Router statistics")
        logger.info("\nAPI docs available at: http://127.0.0.1:8000/docs")
