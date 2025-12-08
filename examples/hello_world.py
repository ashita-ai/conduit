"""Hello World - Absolute Minimum Conduit Example.

The simplest possible example - 5 lines to route and execute a query.

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Expected output:
    Route to: o4-mini (confidence: 73%)

If you see an API key error, create a .env file with your key.
"""

import asyncio

from conduit.core.models import Query
from conduit.engines.router import Router


async def main():
    router = Router()
    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
