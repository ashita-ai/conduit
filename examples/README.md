# Conduit Examples

Usage examples demonstrating Conduit's ML-powered routing capabilities.

## Prerequisites

```bash
# Set environment variables (at least one LLM provider required)
export OPENAI_API_KEY=your_key_here
# OR
export ANTHROPIC_API_KEY=your_key_here

# Optional: Configure database for feedback loop (recommended)
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_ANON_KEY=your_anon_key_here
```

## Quick Start with CLI

The easiest way to try Conduit is using the CLI commands:

```bash
# Run a single query
conduit run --query "What is 2+2?"

# Run with constraints
conduit run --query "Write a poem" --max-cost 0.001 --min-quality 0.8

# Run a demo (10 queries by default)
conduit demo

# Run demo with more queries
conduit demo --queries 50

# JSON output for scripting
conduit run --query "Hello" --json
```

## Examples

### Simple Router (NEW)
Basic usage of the Router class (as shown in README):

```bash
python examples/simple_router.py
```

**What it demonstrates**:
- Simple Router() instantiation with defaults
- Basic query routing without execution
- RoutingDecision results and metadata
- Feature extraction and confidence scores

**Key Code**:
```python
from conduit.engines import Router
from conduit.core.models import Query

router = Router()
query = Query(text="What is 2+2?")
decision = await router.route(query)
print(f"Model: {decision.selected_model}")
print(f"Confidence: {decision.confidence:.2f}")
```

### Basic Routing
Complete routing service with execution and feedback:

```bash
python examples/basic_routing.py
```

**What it demonstrates**:
- Using `RoutingService` for complete routing + execution
- ML-powered model selection
- Feature extraction (embeddings, complexity, domain)
- Routing confidence scores and execution metrics
- Cost and latency tracking

**Key Code**:
```python
from conduit.utils.service_factory import create_service

service = await create_service()
result = await service.complete(prompt="What is 2+2?")
print(f"Model: {result.model}")
print(f"Cost: ${result.metadata['cost']:.6f}")
```

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

**Key Code**:
```python
result = await service.complete(
    prompt="Write a poem",
    constraints={
        "max_cost": 0.001,
        "min_quality": 0.8,
        "max_latency": 2.0,
    }
)
```

## Running Examples

All examples are standalone Python scripts:

```bash
# From project root
cd /path/to/conduit

# Activate virtual environment
source .venv/bin/activate

# Ensure API keys are set
export OPENAI_API_KEY=sk-...

# Run any example
python examples/basic_routing.py
python examples/with_constraints.py
```

## Using the Service Factory

All examples use `conduit.utils.service_factory.create_service()` which:
- Initializes all components (analyzer, bandit, executor, router)
- Connects to database (if configured)
- Loads model states from database
- Returns a ready-to-use `RoutingService` instance

```python
from conduit.utils.service_factory import create_service
from pydantic import BaseModel

class MyResult(BaseModel):
    content: str

service = await create_service(default_result_type=MyResult)
result = await service.complete(prompt="Your query here")
```

## Next Steps

1. **CLI Commands**: Try `conduit run --help` and `conduit demo --help`
2. **API Integration**: See `conduit/api/` for FastAPI service usage
3. **Architecture**: See `docs/ARCHITECTURE.md` for system design
4. **Testing**: Run `pytest tests/` to see comprehensive test suite
5. **Development**: See `AGENTS.md` for development guidelines
