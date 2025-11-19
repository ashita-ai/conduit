"""Hello World - Absolute Minimum Conduit Example.

The simplest possible example - 5 lines to route and execute a query.
"""

import asyncio

from conduit.engines.router import Router


async def main():
    router = Router()
    from conduit.core.models import Query

    decision = await router.route(Query(text="What is 2+2?"))
    print(f"Route to: {decision.selected_model} (confidence: {decision.confidence:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
