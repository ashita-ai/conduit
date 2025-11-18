# Conduit

ML-powered LLM routing system that learns optimal model selection for cost, latency, and quality optimization.

## Overview

Conduit uses contextual bandits (Thompson Sampling) to intelligently route queries to the optimal LLM model based on learned patterns from usage data. Unlike static rule-based routers, Conduit continuously improves routing decisions through feedback loops.

## Key Features

- **ML-Driven Routing**: Learns from usage patterns vs static IF/ELSE rules
- **Multi-Objective Optimization**: Balance cost, latency, and quality constraints
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq via PydanticAI
- **Feedback Loop**: Improves from user ratings and quality metrics
- **Cost Prediction**: Estimate costs before execution
- **A/B Testing**: Built-in experimentation framework

## Quick Start

```python
from conduit import Router
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    answer: str
    confidence: float

router = Router(
    models=["gpt-4o-mini", "gpt-4o", "claude-opus-4"],
    optimize_for="cost_quality_balanced"
)

# Automatic intelligent routing
response = await router.complete(
    prompt="What's 2+2?",
    result_type=AnalysisResult
)

# Provide feedback to improve routing
await router.feedback(
    response_id=response.id,
    quality_score=0.95
)
```

## Installation

```bash
pip install conduit-router
```

## Tech Stack

- **Python 3.10+**
- **PydanticAI 1.14+** (unified LLM interface)
- **FastAPI** (REST API)
- **PostgreSQL** (routing history)
- **scikit-learn** (ML algorithms)
- **sentence-transformers** (query embeddings)

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=conduit

# Type checking
mypy conduit/

# Linting
ruff check conduit/

# Formatting
black conduit/
```

## Architecture

```
Query → Embedding → ML Routing Engine → LLM Provider → Response
   ↓                                                        ↓
   └─────────────────── Feedback Loop ─────────────────────┘
```

**Routing Process**:
1. Analyze query (embedding, features)
2. ML model predicts optimal route
3. Execute via PydanticAI
4. Collect feedback
5. Update routing model

## Documentation

See `docs/` for detailed design specifications and implementation guides.

## License

MIT License - see LICENSE file for details

## Status

**Phase**: Initial development
**Version**: 0.0.1-alpha
