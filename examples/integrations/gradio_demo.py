"""Gradio UI integration example for Conduit.

This example shows how to build an interactive demo UI with Conduit's
ML-powered routing using Gradio. It demonstrates:
- Real-time model selection visualization
- Cost and quality tradeoff controls
- Learning progress tracking
- Query history with model attribution

Requirements:
    pip install gradio

Run:
    python examples/integrations/gradio_demo.py

Then open: http://127.0.0.1:7860
"""

import asyncio
import sys
from datetime import datetime

# Check for gradio dependency
try:
    import gradio as gr
except ImportError:
    print("Gradio integration requires gradio.")
    print("Install with: pip install gradio")
    sys.exit(0)

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.router import Router

# Global router
router: Router | None = None


def initialize_router():
    """Initialize the Conduit router."""
    global router
    if router is None:
        router = Router()
    return router


async def route_query_async(
    query_text: str,
    optimize_for: str,
    max_cost: float | None,
) -> tuple[str, str, float, str, str]:
    """Route a query and return results.

    Returns:
        Tuple of (model_used, confidence_str, confidence_float, reasoning, timestamp)
    """
    if not query_text.strip():
        return "N/A", "0.00", 0.0, "Please enter a query", ""

    initialize_router()

    # Build constraints
    constraints = None
    if max_cost and max_cost > 0:
        constraints = QueryConstraints(max_cost=max_cost)

    # Build preferences
    preferences = UserPreferences(optimize_for=optimize_for)  # type: ignore[arg-type]

    # Create and route query
    query = Query(
        text=query_text,
        constraints=constraints,
        preferences=preferences,
    )

    decision = await router.route(query)

    timestamp = datetime.now().strftime("%H:%M:%S")

    return (
        decision.selected_model,
        f"{decision.confidence:.2f}",
        decision.confidence,
        decision.reasoning,
        timestamp,
    )


def route_query(
    query_text: str,
    optimize_for: str,
    max_cost: float | None,
    history: list,
) -> tuple[str, str, float, str, list]:
    """Synchronous wrapper for route_query_async.

    Returns:
        Tuple of (model_used, confidence_str, confidence_float, reasoning, updated_history)
    """
    result = asyncio.run(route_query_async(query_text, optimize_for, max_cost))
    model_used, confidence_str, confidence_float, reasoning, timestamp = result

    # Update history
    if query_text.strip():
        history = history or []
        history.append(
            {
                "time": timestamp,
                "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                "model": model_used,
                "confidence": confidence_str,
            }
        )
        # Keep last 10 entries
        history = history[-10:]

    return model_used, confidence_str, confidence_float, reasoning, history


def format_history(history: list) -> str:
    """Format history for display."""
    if not history:
        return "No queries yet"

    lines = ["| Time | Query | Model | Confidence |", "|------|-------|-------|------------|"]
    for entry in reversed(history):
        lines.append(
            f"| {entry['time']} | {entry['query']} | {entry['model']} | {entry['confidence']} |"
        )
    return "\n".join(lines)


def get_stats() -> str:
    """Get router statistics."""
    if router is None:
        return "Router not initialized"

    stats = router.hybrid_router.get_stats()

    lines = [
        "## Router Statistics",
        "",
        f"**Total Queries:** {stats.get('total_queries', 0)}",
        f"**Current Phase:** {stats.get('current_phase', 'unknown')}",
        f"**Algorithm:** {router.algorithm}",
        "",
        "### Model Selection Counts",
    ]

    arm_pulls = stats.get("arm_pulls", {})
    if arm_pulls:
        for model, count in sorted(arm_pulls.items(), key=lambda x: -x[1]):
            lines.append(f"- {model}: {count}")
    else:
        lines.append("- No data yet")

    cache_stats = router.get_cache_stats()
    if cache_stats:
        lines.extend(
            [
                "",
                "### Cache Statistics",
                f"- Hits: {cache_stats.get('hits', 0)}",
                f"- Misses: {cache_stats.get('misses', 0)}",
                f"- Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%",
            ]
        )

    return "\n".join(lines)


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="Conduit Demo",
    ) as demo:
        gr.Markdown(
            """
            # Conduit Demo

            **ML-powered LLM routing** - Conduit learns which model works best for each query type.

            Enter a query below to see which model Conduit selects and why.
            """
        )

        # State for history
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your query here...",
                    lines=3,
                )

                with gr.Row():
                    optimize_dropdown = gr.Dropdown(
                        choices=["balanced", "quality", "cost", "speed"],
                        value="balanced",
                        label="Optimize For",
                    )
                    max_cost_input = gr.Number(
                        label="Max Cost ($)",
                        value=None,
                        minimum=0,
                        maximum=1.0,
                    )

                route_btn = gr.Button("Route Query", variant="primary")

            with gr.Column(scale=1):
                # Output section
                model_output = gr.Textbox(label="Selected Model", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
                confidence_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    label="Confidence Level",
                    interactive=False,
                )

        # Reasoning section
        reasoning_output = gr.Textbox(
            label="Routing Reasoning",
            lines=2,
            interactive=False,
        )

        # History and stats tabs
        with gr.Tabs():
            with gr.TabItem("Query History"):
                history_output = gr.Markdown("No queries yet")

            with gr.TabItem("Router Stats"):
                stats_output = gr.Markdown("Click 'Refresh Stats' to see statistics")
                refresh_btn = gr.Button("Refresh Stats")

        # Example queries
        gr.Examples(
            examples=[
                ["What is 2+2?", "cost", None],
                ["Explain quantum computing in detail", "quality", None],
                ["Write a haiku about programming", "balanced", None],
                ["Translate 'hello world' to Spanish", "speed", None],
                ["Analyze the time complexity of quicksort", "quality", 0.01],
            ],
            inputs=[query_input, optimize_dropdown, max_cost_input],
        )

        # Event handlers
        def on_route(query, optimize, max_cost, history):
            model, conf_str, conf_float, reasoning, new_history = route_query(
                query, optimize, max_cost, history
            )
            history_md = format_history(new_history)
            return model, conf_str, conf_float, reasoning, new_history, history_md

        route_btn.click(
            fn=on_route,
            inputs=[query_input, optimize_dropdown, max_cost_input, history_state],
            outputs=[
                model_output,
                confidence_output,
                confidence_slider,
                reasoning_output,
                history_state,
                history_output,
            ],
        )

        # Also trigger on Enter key
        query_input.submit(
            fn=on_route,
            inputs=[query_input, optimize_dropdown, max_cost_input, history_state],
            outputs=[
                model_output,
                confidence_output,
                confidence_slider,
                reasoning_output,
                history_state,
                history_output,
            ],
        )

        refresh_btn.click(fn=get_stats, outputs=[stats_output])

        gr.Markdown(
            """
            ---
            **How it works:**
            - Conduit uses Thompson Sampling (Bayesian bandit) to balance exploration and exploitation
            - Query features (complexity, domain, length) influence model selection
            - The system learns from feedback to improve routing over time
            - Different optimization presets adjust the quality/cost/latency tradeoffs
            """
        )

    return demo


def main():
    """Run the Gradio demo."""
    print("=" * 80)
    print("Conduit Gradio Demo")
    print("=" * 80)
    print("\nInitializing router...")

    initialize_router()

    print("Starting Gradio interface...")
    print("Open http://127.0.0.1:7860 in your browser\n")

    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
