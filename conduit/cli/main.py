"""Command-line interface for Conduit."""

import logging
import sys

import click
import uvicorn

from conduit.api.app import create_app
from conduit.core.config import settings

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
def version() -> None:
    """Show version information."""
    click.echo("Conduit v0.1.0")


if __name__ == "__main__":
    cli()

