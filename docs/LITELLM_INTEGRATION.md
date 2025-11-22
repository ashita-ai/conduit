# LiteLLM Integration Strategy

**Document Purpose**: Strategic analysis and implementation plans for integrating Conduit with LiteLLM
**Last Updated**: 2025-11-21
**Status**: Planning phase - GitHub issues created

---

## Executive Summary

Conduit can integrate with LiteLLM in two complementary ways:

1. **Path 1: Conduit as LiteLLM Routing Strategy** (RECOMMENDED - Issue #9)
   - Conduit plugs into LiteLLM as a native routing plugin
   - Target: LiteLLM's 12K+ star ecosystem
   - Effort: 1-2 days
   - Package: `conduit-litellm` (separate)

2. **Path 2: LiteLLM as Conduit Execution Backend** (Issue #8)
   - Conduit uses LiteLLM for API calls instead of PydanticAI
   - Target: Conduit users wanting 100+ providers
   - Effort: 2-3 days
   - Package: `conduit[litellm]` (optional dependency)

**Recommendation**: Build Path 1 first for maximum strategic impact.

---

## Background: Why LiteLLM Integration?

### LiteLLM Overview
- **What**: Unified API for 100+ LLM providers (OpenAI, Anthropic, Azure, Google, Cohere, etc.)
- **Popularity**: 12K+ GitHub stars, widely adopted in production
- **Features**: Retries, fallbacks, rate limiting, cost tracking, load balancing
- **Routing**: Rule-based only (cheapest, fastest, round-robin, least-busy)

### Conduit Overview
- **What**: ML-powered LLM routing using contextual bandits (Thompson Sampling)
- **Unique Value**: Learns which model works best for YOUR workload
- **Providers**: 5 via PydanticAI (OpenAI, Anthropic, Google, Groq, Ollama)
- **Routing**: ML-based learning that improves with usage

### Complementary Strengths

| Capability | LiteLLM | Conduit |
|------------|---------|---------|
| **Providers** | 100+ (✅ massive) | 5 (❌ limited) |
| **Routing** | Rule-based (❌ static) | ML-based (✅ learning) |
| **Infrastructure** | Battle-tested (✅) | New (⚠️) |
| **Intelligence** | None (❌) | Contextual bandits (✅) |
| **Cost Tracking** | Built-in (✅) | Manual (⚠️) |

**Integration opportunity**: LiteLLM's infrastructure + Conduit's intelligence = best of both worlds

---

## Path 1: Conduit as LiteLLM Routing Strategy (RECOMMENDED)

**GitHub Issue**: [#9](https://github.com/MisfitIdeas/conduit/issues/9)

### Overview

LiteLLM provides `CustomRoutingStrategyBase` API for plugging in custom routing logic. Conduit can implement this interface to become a native routing plugin.

### Technical Architecture

#### LiteLLM's Custom Routing API

```python
from litellm.router import CustomRoutingStrategyBase
from typing import Optional, List, Dict, Union

class CustomRoutingStrategy(CustomRoutingStrategyBase):
    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Select optimal deployment from litellm.router.model_list

        Returns: An element from model_list
        """
        # Custom routing logic here
        pass
```

**Reference**: [LiteLLM Router Docs](https://docs.litellm.ai/docs/routing), [Custom Routing Issue #4302](https://github.com/BerriAI/litellm/issues/4302)

#### Conduit Implementation

```python
# conduit_litellm/strategy.py
from litellm.router import CustomRoutingStrategyBase
from conduit.engines.router import ConduitRouter
from conduit.core.models import QueryFeatures, ModelArm, BanditFeedback

class ConduitRoutingStrategy(CustomRoutingStrategyBase):
    """ML-powered routing strategy for LiteLLM."""

    def __init__(
        self,
        conduit_router: Optional[ConduitRouter] = None,
        bandit_algorithm: str = "contextual_thompson",
        **conduit_config
    ):
        self.conduit_router = conduit_router
        self.bandit_algorithm = bandit_algorithm
        self.conduit_config = conduit_config
        self._initialized = False

    async def _initialize_from_litellm(self, router):
        """Convert LiteLLM model_list to Conduit ModelArms on first call."""
        if self._initialized:
            return

        # Convert LiteLLM deployments to Conduit arms
        arms = []
        for deployment in router.model_list:
            arm = ModelArm(
                model_id=deployment["model_info"]["id"],
                provider=deployment["litellm_params"]["model"].split("/")[0],
                cost_per_token=deployment["model_info"].get("cost_per_token", 0.0),
            )
            arms.append(arm)

        # Initialize Conduit router
        if not self.conduit_router:
            from conduit.utils.service_factory import create_router
            self.conduit_router = await create_router(
                arms=arms,
                algorithm=self.bandit_algorithm,
                **self.conduit_config
            )

        self._initialized = True

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """Use Conduit's ML router to select optimal deployment."""
        # Initialize on first call
        await self._initialize_from_litellm(self.router)

        # Extract query text
        query_text = messages[-1]["content"] if messages else (input or "")

        # Analyze query features (embeddings, complexity, domain)
        features = await self.conduit_router.analyzer.analyze_query(query_text)

        # Use ML bandit to select optimal model
        selected_arm = await self.conduit_router.bandit.select_arm(features)

        # Find matching LiteLLM deployment
        for deployment in self.router.model_list:
            if deployment["model_info"]["id"] == selected_arm.model_id:
                # Store context for feedback
                self._store_context(deployment["model_info"]["id"], features)
                return deployment

        # Fallback to default
        return self.router.model_list[0]

    async def record_feedback(
        self,
        deployment_id: str,
        cost: float,
        latency: float,
        quality_score: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Record feedback to update bandit (called after completion)."""
        features = self._get_stored_context(deployment_id)

        # Calculate composite reward
        reward = self._calculate_reward(
            quality=quality_score or (0.0 if error else 1.0),
            cost=cost,
            latency=latency
        )

        # Update bandit
        feedback = BanditFeedback(
            model_id=deployment_id,
            reward=reward,
            cost=cost,
            latency=latency,
            quality_score=quality_score,
            error_occurred=bool(error)
        )
        await self.conduit_router.bandit.update(feedback, features)
```

#### Usage Example

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Initialize LiteLLM router with 100+ providers
litellm_router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4", "api_key": "..."},
            "model_info": {"id": "gpt-4-openai"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "azure/gpt-4", "api_key": "..."},
            "model_info": {"id": "gpt-4-azure"},
        },
        {
            "model_name": "claude-3",
            "litellm_params": {"model": "anthropic/claude-3-opus", "api_key": "..."},
            "model_info": {"id": "claude-3-opus"},
        },
    ]
)

# Plug in Conduit's ML routing (ONE LINE!)
litellm_router.set_custom_routing_strategy(
    ConduitRoutingStrategy(
        bandit_algorithm="contextual_thompson",
        lambda_reg=1.0,
        window_size=1000,
    )
)

# Use LiteLLM normally - Conduit routes intelligently
response = await litellm_router.acompletion(
    model="gpt-4",  # Conduit selects optimal gpt-4 deployment
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Conduit learns from usage automatically
# (via feedback hook if we can tap into LiteLLM response tracking)
```

### Strategic Benefits

#### Market Positioning
- **Value Prop**: "ML-powered routing for LiteLLM"
- **Positioning**: "Make LiteLLM smarter - upgrade from rule-based to ML-based routing"
- **Unique**: Only ML-based routing strategy in LiteLLM ecosystem
- **Distribution**: Tap into 12K+ GitHub stars, massive community

#### Technical Benefits
- **100+ providers**: Instant access to LiteLLM's entire ecosystem
- **Battle-tested infra**: LiteLLM handles retries, fallbacks, rate limiting
- **Cost tracking**: LiteLLM provides cost/latency automatically (helps feedback)
- **Maintenance**: LiteLLM team maintains provider APIs, not us

#### User Benefits
- **Zero friction**: Add one line to existing LiteLLM setup
- **Immediate value**: ML routing without changing code
- **Learning**: Gets smarter with usage (30-50% cost savings)
- **Flexibility**: Choose algorithm (Thompson, LinUCB, UCB1, etc.)

### Implementation Plan

**Total Effort**: 1-2 days (3 phases)

#### Phase 1: Core Integration (1 day)
- [ ] Create `conduit-litellm` package structure
- [ ] Implement `ConduitRoutingStrategy(CustomRoutingStrategyBase)`
- [ ] Convert LiteLLM `model_list` to Conduit `ModelArm` format
- [ ] Implement `async_get_available_deployment()` with ML routing
- [ ] Extract query text from LiteLLM messages/input
- [ ] Store routing context for feedback

#### Phase 2: Feedback Loop (0.5 days)
- [ ] Hook into LiteLLM response tracking (if possible)
- [ ] Implement `record_feedback()` method
- [ ] Calculate composite rewards (quality + cost + latency)
- [ ] Update bandit with feedback
- [ ] Test learning convergence

#### Phase 3: Testing & Docs (0.5 days)
- [ ] Unit tests for `ConduitRoutingStrategy`
- [ ] Integration tests with mock LiteLLM router
- [ ] Example: basic usage with multiple providers
- [ ] Example: performance comparison (ML vs rule-based)
- [ ] README with installation and usage

#### Phase 4: Package & Distribution
- [ ] Publish `conduit-litellm` to PyPI
- [ ] Create docs: `conduit-litellm.readthedocs.io`
- [ ] Submit PR to LiteLLM docs (community plugins section)
- [ ] Blog post: "ML-Powered Routing for LiteLLM"
- [ ] Demo video showing learning curve

### Package Structure

```
conduit-litellm/
├── conduit_litellm/
│   ├── __init__.py
│   ├── strategy.py          # ConduitRoutingStrategy
│   ├── config.py            # Configuration
│   ├── feedback.py          # Feedback integration
│   └── utils.py             # Helper functions
├── tests/
│   ├── test_strategy.py
│   ├── test_feedback.py
│   ├── test_integration.py
│   └── fixtures/
├── examples/
│   ├── basic_usage.py
│   ├── custom_algorithm.py
│   ├── performance_comparison.py
│   └── feedback_loop.py
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   ├── algorithms.md
│   └── troubleshooting.md
├── pyproject.toml
├── README.md
└── LICENSE
```

### Dependencies

```toml
[project]
name = "conduit-litellm"
version = "0.1.0"
description = "ML-powered routing strategy for LiteLLM"
dependencies = [
    "conduit>=0.1.0",      # Core Conduit library
    "litellm>=1.74.9",     # LiteLLM with custom routing support
]
```

### Success Metrics

- [ ] Strategy successfully routes 100+ requests
- [ ] Feedback loop updates bandit correctly
- [ ] Learning: 30%+ cost reduction after 1000 requests
- [ ] Performance: <50ms routing overhead
- [ ] Test coverage: >90%
- [ ] Documentation: 3+ complete examples
- [ ] Community: 5+ LiteLLM users adopt it

---

## Path 2: LiteLLM as Conduit Execution Backend

**GitHub Issue**: [#8](https://github.com/MisfitIdeas/conduit/issues/8)

### Overview

Add LiteLLM as an optional execution backend for Conduit, expanding from 5 providers (PydanticAI) to 100+ providers (LiteLLM).

### Technical Architecture

#### Pluggable Execution Backend

```python
# conduit/core/config.py
class ConduitConfig:
    execution_backend: Literal["pydantic_ai", "litellm"] = "pydantic_ai"

# conduit/engines/executor.py
class LLMExecutor(ABC):
    @abstractmethod
    async def execute(self, model_id: str, query: str) -> ExecutionResult:
        pass

class PydanticAIExecutor(LLMExecutor):
    # Current implementation (keep as-is)
    pass

class LiteLLMExecutor(LLMExecutor):
    async def execute(self, model_id: str, query: str) -> ExecutionResult:
        import litellm

        response = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": query}]
        )

        # Extract metrics for feedback
        return ExecutionResult(
            content=response.choices[0].message.content,
            cost=response._hidden_params.get("response_cost", 0.0),
            latency=response._response_ms / 1000.0,
            tokens=response.usage.total_tokens,
            metadata={"provider": response.model.split("/")[0]}
        )
```

#### Usage Example

```python
from conduit import create_service

# Create service with LiteLLM backend
service = await create_service(
    execution_backend="litellm",  # Use LiteLLM instead of PydanticAI
    default_result_type=MyResult
)

# Conduit's ML router selects model
# LiteLLM executes with any of 100+ providers
result = await service.complete(
    prompt="Explain quantum computing",
    user_id="user_123"
)
```

### Implementation Plan

**Total Effort**: 2-3 days

#### Day 1: Core Integration
- [ ] Create `LiteLLMExecutor` class
- [ ] Add `execution_backend` to `ConduitConfig`
- [ ] Map model IDs (Conduit format → LiteLLM format)
- [ ] Extract cost/latency/tokens from LiteLLM responses
- [ ] Handle LiteLLM-specific errors

#### Day 2: Testing
- [ ] Unit tests for `LiteLLMExecutor`
- [ ] Integration tests with mock LiteLLM
- [ ] Test cost/latency accuracy
- [ ] Test error handling
- [ ] Verify feedback loop works

#### Day 3: Documentation
- [ ] Update README with LiteLLM backend section
- [ ] Create example: `examples/04_production/litellm_backend.py`
- [ ] Document model ID mapping
- [ ] Add troubleshooting guide
- [ ] Update `supported_models()` to show LiteLLM models

### Strategic Benefits

- **100+ providers** vs current 5 (20x expansion)
- **Battle-tested** infrastructure (retries, fallbacks)
- **User choice**: PydanticAI (5) or LiteLLM (100+)
- **Migration path**: Start PydanticAI, grow to LiteLLM

### Success Metrics

- [ ] LiteLLM backend works with 10+ providers
- [ ] Cost/latency tracking within 5% accuracy
- [ ] Test coverage >90%
- [ ] Example demonstrates value
- [ ] Documentation enables self-service

---

## Comparison: Path 1 vs Path 2

| Aspect | Path 1: Routing Strategy | Path 2: Execution Backend |
|--------|-------------------------|---------------------------|
| **Target Users** | LiteLLM users (12K+ stars) | Conduit users |
| **Value Prop** | "Make LiteLLM smarter" | "More LLM providers" |
| **Integration** | LiteLLM uses Conduit | Conduit uses LiteLLM |
| **Package** | `conduit-litellm` (separate) | `conduit[litellm]` (optional) |
| **Distribution** | LiteLLM ecosystem | Conduit users only |
| **Effort** | 1-2 days | 2-3 days |
| **Complexity** | Lower (routing only) | Moderate (execution layer) |
| **Impact** | High (ecosystem play) | Medium (feature addition) |
| **Uniqueness** | Only ML router for LiteLLM | Many alternatives exist |

**Recommendation**: Build Path 1 first, then Path 2 if demand exists.

---

## Market Positioning Strategy

### Path 1 Positioning (Recommended)
- **Tagline**: "ML-powered routing for LiteLLM"
- **Elevator Pitch**: "Upgrade LiteLLM from rule-based routing (cheapest/fastest) to ML-based routing that learns which models work best for YOUR workload. 30-50% cost savings with one line of code."
- **Target**: LiteLLM users wanting smarter routing
- **Competition**: None (only ML router in ecosystem)
- **Distribution**: LiteLLM community, plugins page, blog posts

### Path 2 Positioning
- **Tagline**: "Intelligent LLM routing with 100+ providers"
- **Elevator Pitch**: "Conduit's ML routing with access to 100+ LLM providers through LiteLLM. Get the best of both worlds: intelligent routing + massive provider ecosystem."
- **Target**: Conduit users wanting more providers
- **Competition**: Other routing solutions with more providers
- **Distribution**: Conduit documentation, examples

### Communication Strategy

**Say (Path 1)**:
- "Make LiteLLM smarter with ML routing"
- "Learns which models work best for your workload"
- "30-50% cost savings through intelligent selection"
- "One line to add: `set_custom_routing_strategy(ConduitRoutingStrategy())`"

**Avoid**:
- "Thompson Sampling", "contextual bandits" (too technical)
- "100% optimal" (impossible, we're probabilistic)
- "Revolutionary" or marketing hyperbole

---

## Technical Risks & Mitigations

### Path 1 Risks

**Risk 1: LiteLLM API changes**
- **Impact**: Strategy breaks if `CustomRoutingStrategyBase` API changes
- **Likelihood**: Low (stable API, community would complain)
- **Mitigation**: Pin LiteLLM version, monitor releases, automated tests

**Risk 2: Feedback loop integration**
- **Impact**: Can't learn if we can't get cost/latency from LiteLLM
- **Likelihood**: Medium (depends on LiteLLM's callback/hook system)
- **Mitigation**: Manual `record_feedback()` method as fallback

**Risk 3: Model ID format incompatibility**
- **Impact**: Can't match Conduit models to LiteLLM deployments
- **Likelihood**: Low (model IDs are configurable)
- **Mitigation**: Clear mapping documentation, validation with helpful errors

### Path 2 Risks

**Risk 1: PydanticAI feature loss**
- **Impact**: Lose structured outputs, type safety
- **Likelihood**: High (LiteLLM doesn't have equivalent)
- **Mitigation**: Keep PydanticAI as default, LiteLLM as optional

**Risk 2: Cost tracking differences**
- **Impact**: Inaccurate cost tracking affects learning
- **Likelihood**: Medium (different providers report differently)
- **Mitigation**: Normalize costs, use fallback estimates

**Risk 3: Dependency bloat**
- **Impact**: LiteLLM is large dependency
- **Likelihood**: High (100+ provider integrations)
- **Mitigation**: Make optional (`pip install conduit[litellm]`)

---

## Next Steps

### Immediate Actions (Path 1)
1. ✅ Create GitHub issue #9
2. ⏳ Create `conduit-litellm` repository
3. ⏳ Implement `ConduitRoutingStrategy` (Phase 1)
4. ⏳ Test with mock LiteLLM router
5. ⏳ Add feedback loop (Phase 2)
6. ⏳ Publish to PyPI

### Future Actions (Path 2)
1. ✅ Create GitHub issue #8
2. ⏳ Wait for Path 1 completion
3. ⏳ Assess demand from Conduit users
4. ⏳ Implement if justified

---

## References

### LiteLLM Documentation
- [Router Documentation](https://docs.litellm.ai/docs/routing)
- [Auto Routing](https://docs.litellm.ai/docs/proxy/auto_routing)
- [Router Architecture](https://docs.litellm.ai/docs/router_architecture)
- [Custom Routing Strategy Issue #4302](https://github.com/BerriAI/litellm/issues/4302)
- [Router Source Code](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

### Conduit Documentation
- [Bandit Algorithms](./BANDIT_ALGORITHMS.md)
- [Model Discovery](./MODEL_DISCOVERY.md)
- [Architecture](./ARCHITECTURE.md)
- [AGENTS.md (Development Guide)](../AGENTS.md)

### GitHub Issues
- [Issue #8: LiteLLM Execution Backend](https://github.com/MisfitIdeas/conduit/issues/8)
- [Issue #9: LiteLLM Routing Strategy](https://github.com/MisfitIdeas/conduit/issues/9)

---

**Last Updated**: 2025-11-21
**Next Review**: After Path 1 implementation
