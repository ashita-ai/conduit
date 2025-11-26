"""Command-line interface for Conduit."""

import asyncio
import json
import logging
import sys
from typing import Any

import click
import uvicorn
from pydantic import BaseModel

from conduit.api.app import create_app
from conduit.core.config import settings
from conduit.utils.service_factory import create_service

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Conduit - ML-powered LLM routing system."""
    pass


@cli.command()
@click.option(
    "--host",
    default=settings.api_host,
    help="Host to bind to",
    show_default=True,
)
@click.option(
    "--port",
    default=settings.api_port,
    type=int,
    help="Port to bind to",
    show_default=True,
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--log-level",
    default=settings.log_level.lower(),
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Logging level",
    show_default=True,
)
def serve(host: str, port: int, reload: bool, log_level: str) -> None:
    """Start the Conduit API server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Conduit API server on {host}:{port}")

    # Create app
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Query text to route and execute",
)
@click.option(
    "--max-cost",
    type=float,
    help="Maximum cost constraint (in dollars)",
)
@click.option(
    "--max-latency",
    type=float,
    help="Maximum latency constraint (in seconds)",
)
@click.option(
    "--min-quality",
    type=float,
    help="Minimum quality constraint (0.0-1.0)",
)
@click.option(
    "--provider",
    help="Preferred provider (openai, anthropic, google, groq)",
)
@click.option(
    "--user-id",
    help="User identifier for tracking",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
def run(
    query: str,
    max_cost: float | None,
    max_latency: float | None,
    min_quality: float | None,
    provider: str | None,
    user_id: str | None,
    output_json: bool,
) -> None:
    """Run a single query through Conduit routing system."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Less verbose for CLI
        format="%(message)s",
    )

    # Build constraints
    constraints: dict[str, Any] = {}
    if max_cost is not None:
        constraints["max_cost"] = max_cost
    if max_latency is not None:
        constraints["max_latency"] = max_latency
    if min_quality is not None:
        constraints["min_quality"] = min_quality
    if provider is not None:
        constraints["preferred_provider"] = provider

    async def execute() -> None:
        """Execute query asynchronously."""
        try:
            # Create default result type
            class SimpleResult(BaseModel):
                content: str

            service = await create_service(default_result_type=SimpleResult)

            # Execute query
            result = await service.complete(
                prompt=query,
                constraints=constraints if constraints else None,
                user_id=user_id,
            )

            if output_json:
                # JSON output
                output = {
                    "id": result.id,
                    "query_id": result.query_id,
                    "model": result.model,
                    "data": result.data,
                    "metadata": result.metadata,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                # Human-readable output
                click.echo(f"\n{'='*60}")
                click.echo("Routing Results")
                click.echo(f"{'='*60}")
                click.echo(f"Model: {result.model}")
                click.echo(
                    f"Confidence: {result.metadata.get('routing_confidence', 0.0):.2f}"
                )
                click.echo(f"Cost: ${result.metadata.get('cost', 0.0):.6f}")
                click.echo(f"Latency: {result.metadata.get('latency', 0.0):.2f}s")
                click.echo(f"Tokens: {result.metadata.get('tokens', 0)}")
                click.echo(f"\nReasoning: {result.metadata.get('reasoning', 'N/A')}")
                click.echo("\nResponse:")
                content = result.data.get("content", "N/A")
                # Wrap long content
                for line in content.split("\n"):
                    click.echo(f"  {line}")

            # Cleanup
            await service.database.disconnect()

        except Exception as e:
            logger.error(f"Error: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(execute())


@cli.command()
@click.option(
    "--queries",
    type=int,
    default=10,
    help="Number of demo queries to run",
    show_default=True,
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare Conduit vs static routing (requires baseline)",
)
def demo(queries: int, compare: bool) -> None:
    """Run a demonstration of Conduit's routing capabilities."""
    click.echo("Conduit Demo - ML-Powered LLM Routing")
    click.echo("=" * 60)
    click.echo(f"\nRunning {queries} demo queries...\n")

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
    )

    async def run_demo() -> None:
        """Run demo queries."""
        try:

            class SimpleResult(BaseModel):
                content: str

            service = await create_service(default_result_type=SimpleResult)

            # Demo queries
            demo_queries = [
                "What is 2+2?",
                "Write a haiku about coding",
                "Explain photosynthesis briefly",
                "What's the capital of France?",
                "Write a Python function to reverse a string",
            ] * (queries // 5 + 1)
            demo_queries = demo_queries[:queries]

            total_cost = 0.0
            total_latency = 0.0
            model_counts: dict[str, int] = {}

            for i, query_text in enumerate(demo_queries, 1):
                click.echo(f"[{i}/{queries}] {query_text[:50]}...")

                try:
                    result = await service.complete(
                        prompt=query_text, user_id="demo_user"
                    )

                    cost = result.metadata.get("cost", 0.0)
                    latency = result.metadata.get("latency", 0.0)
                    model = result.model

                    total_cost += cost
                    total_latency += latency
                    model_counts[model] = model_counts.get(model, 0) + 1

                    click.echo(f"  → {model} (${cost:.6f}, {latency:.2f}s)")

                except Exception as e:
                    click.echo(f"  ✗ Error: {e}")

            # Summary
            click.echo(f"\n{'='*60}")
            click.echo("Demo Summary")
            click.echo(f"{'='*60}")
            click.echo(f"Total Queries: {queries}")
            click.echo(f"Total Cost: ${total_cost:.6f}")
            click.echo(f"Average Cost per Query: ${total_cost/queries:.6f}")
            click.echo(f"Average Latency: {total_latency/queries:.2f}s")
            click.echo("\nModel Distribution:")
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
                percentage = (count / queries) * 100
                click.echo(f"  {model}: {count} ({percentage:.1f}%)")

            if compare:
                click.echo(f"\n{'='*60}")
                click.echo("Comparison with Static Routing")
                click.echo(f"{'='*60}")
                click.echo(
                    "Note: Static routing comparison requires baseline implementation"
                )
                click.echo("This feature is coming in Phase 2")

            await service.database.disconnect()

        except Exception as e:
            logger.error(f"Demo error: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(run_demo())


@cli.command()
def version() -> None:
    """Show version information."""
    click.echo("Conduit v0.1.0")


if __name__ == "__main__":
    cli()
