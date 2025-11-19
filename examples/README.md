# Conduit Examples

Usage examples demonstrating Conduit's ML-powered routing capabilities.

## Prerequisites

```bash
# Set environment variables
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# Optional: Configure database for feedback loop
export DATABASE_URL=postgresql://...
```

## Examples

### Basic Routing
Simple query routing with automatic model selection:

```bash
python examples/basic_routing.py
```

**What it demonstrates**:
- Creating queries
- ML-powered model selection
- Feature extraction (embeddings, complexity, domain)
- Routing confidence scores

### Constrained Routing
Routing with cost, latency, and quality constraints:

```bash
python examples/with_constraints.py
```

**What it demonstrates**:
- Cost optimization (prefer cheaper models)
- Quality guarantees (minimum quality thresholds)
- Latency constraints (fast response requirements)
- Provider preferences (route to specific providers)

### Feedback Loop (TODO - Phase 2)
Improving routing through user feedback:

```bash
python examples/feedback_loop.py
```

**What it demonstrates**:
- Explicit feedback submission
- Model state updates (Thompson Sampling)
- Quality score tracking
- Routing improvement over time

## Running Examples

All examples are standalone Python scripts:

```bash
# From project root
cd /path/to/conduit

# Activate virtual environment
source .venv/bin/activate

# Run any example
python examples/basic_routing.py
```

## Next Steps

1. **Explore Core Concepts**: See `docs/ARCHITECTURE.md` for system design
2. **API Integration**: See `conduit/api/` for FastAPI service usage
3. **Testing**: Run `pytest tests/` to see comprehensive test suite
4. **Development**: See `AGENTS.md` for development guidelines
