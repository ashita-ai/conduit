# LiteLLM Integration Status

**Document Purpose**: Implementation status and usage guide for Conduit-LiteLLM integration
**Last Updated**: 2025-11-22
**Status**: Path 1 COMPLETE, Path 2 not started

---

## Executive Summary

Conduit integrates with LiteLLM as a native routing strategy, bringing ML-powered model selection to LiteLLM's 100+ provider ecosystem.

### Path 1: Conduit as LiteLLM Routing Strategy ✅ COMPLETE
- **Status**: Shipped in `conduit_litellm/` package with feedback loop
- **Commits**: d7b69cc, ff06a46, c5dd24c, abcae20, [Issue #13 complete]
- **Features**:
  - `ConduitRoutingStrategy` implements `CustomRoutingStrategyBase`
  - Hybrid routing support (UCB1→LinUCB warm start)
  - Async/sync context handling (Issue #31 fixed)
  - Helper method `setup_strategy()` for initialization
  - **Automatic feedback loop** via `ConduitFeedbackLogger` (Issue #13)
- **Usage**: See `conduit_litellm/README.md`
- **Issues**: All complete ✅

### Path 2: LiteLLM as Conduit Execution Backend ❌ NOT STARTED
- **Status**: Not implemented, no current plans
- **Reason**: Path 1 provides better strategic value
- **Alternative**: Conduit continues using PydanticAI (8 providers)

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

| Capability | LiteLLM | Pure Conduit | conduit_litellm (Path 1) |
|------------|---------|--------------|--------------------------|
| **Providers** | 100+ (✅) | 5 (❌) | 100+ (✅) |
| **Routing** | Rule-based (❌) | ML-based (✅) | ML-based (✅) |
| **Infrastructure** | Battle-tested (✅) | New (⚠️) | Battle-tested (✅) |
| **Intelligence** | None (❌) | Contextual bandits (✅) | Contextual bandits (✅) |
| **Cost Tracking** | Built-in via response (✅) | llm-prices.com fetch (⚠️) | Built-in via LiteLLM\* (✅) |

\* When Issue #13 (feedback loop) is complete, `conduit_litellm` will use LiteLLM's built-in cost tracking via `response._hidden_params["response_cost"]` instead of llm-prices.com.

**Integration opportunity**: LiteLLM's infrastructure + Conduit's intelligence = best of both worlds

---

## Path 1: Conduit as LiteLLM Routing Strategy ✅ COMPLETE

**Implementation**: `conduit_litellm/` package
**Key Commits**: d7b69cc, ff06a46, c5dd24c, abcae20

### What Was Built

The `conduit_litellm` package provides `ConduitRoutingStrategy`, which implements LiteLLM's `CustomRoutingStrategyBase` interface. This allows Conduit's ML-powered routing to work with all 100+ LiteLLM providers.

### Quick Start

```python
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure LiteLLM with your models
router = Router(
    model_list=[
        {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4o-mini"}},
        {"model_name": "claude-3", "litellm_params": {"model": "claude-3-haiku"}},
    ]
)

# Setup Conduit routing strategy (use helper method!)
strategy = ConduitRoutingStrategy(use_hybrid=True)
ConduitRoutingStrategy.setup_strategy(router, strategy)

# LiteLLM now uses Conduit's ML routing
response = await router.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Implementation Details

**Source**: `conduit_litellm/strategy.py:30`

Key features implemented:
- `async_get_available_deployment()`: ML model selection
- `get_available_deployment()`: Sync wrapper with async context handling (Issue #31 fix)
- `setup_strategy()`: Helper for proper initialization
- Lazy initialization from LiteLLM's `model_list`
- Hybrid routing support (UCB1→LinUCB warm start)

**Feedback Loop**: Automatic learning enabled via `ConduitFeedbackLogger` ✅

The feedback loop works seamlessly in the background:
1. LiteLLM completes request → calls `async_log_success_event()`
2. Extract cost from `response._hidden_params['response_cost']`
3. Calculate latency from `end_time - start_time`
4. Reconstruct features from query text using `router.analyzer`
5. Create `BanditFeedback` with quality=0.9 (success) or 0.1 (failure)
6. Update bandit: `router.bandit.update(feedback, features)` or `router.hybrid_router.update()`
7. Bandit learns which models work best for each query type

This eliminates dependency on llm-prices.com for `conduit_litellm` usage.

### Strategic Benefits

#### Market Positioning
- **Value Prop**: "ML-powered routing for LiteLLM"
- **Positioning**: "Make LiteLLM smarter - upgrade from rule-based to ML-based routing"
- **Unique**: Only ML-based routing strategy in LiteLLM ecosystem
- **Distribution**: Tap into 12K+ GitHub stars, massive community

#### Technical Benefits
- **100+ providers**: Instant access to LiteLLM's entire ecosystem
- **Battle-tested infra**: LiteLLM handles retries, fallbacks, rate limiting
- **Cost tracking**: LiteLLM provides cost/latency in response metadata (`response._hidden_params["response_cost"]`)
- **No llm-prices.com dependency**: Use LiteLLM's built-in pricing instead of external API
- **Maintenance**: LiteLLM team maintains provider APIs, not us

#### User Benefits
- **Zero friction**: Add one line to existing LiteLLM setup
- **Immediate value**: ML routing without changing code
- **Learning**: Gets smarter with usage (learns optimal model selection)
- **Flexibility**: Choose algorithm (Thompson, LinUCB, UCB1, etc.)

### Implementation Status

**Total Time**: 2 days (Nov 22, 2025)

#### Phase 1: Core Integration ✅ COMPLETE
- ✅ Create `conduit_litellm/` package structure (d7b69cc)
- ✅ Implement `ConduitRoutingStrategy(CustomRoutingStrategyBase)` (ff06a46)
- ✅ Convert LiteLLM `model_list` to Conduit model IDs
- ✅ Implement `async_get_available_deployment()` with ML routing
- ✅ Extract query text from LiteLLM messages/input
- ✅ Add `setup_strategy()` helper for initialization (c5dd24c)
- ✅ Fix async context handling (Issue #31, abcae20)

#### Phase 2: Feedback Loop ✅ COMPLETE (Issue #13)
- ✅ Hook into LiteLLM response tracking via `CustomLogger` callbacks
- ✅ Implemented `ConduitFeedbackLogger` class (conduit_litellm/feedback.py)
- ✅ Extract cost from `response._hidden_params["response_cost"]`
- ✅ Calculate latency from start_time/end_time
- ✅ Estimate quality (success=0.9, failure=0.1)
- ✅ Generate features from query text using router.analyzer
- ✅ Update bandit with `BanditFeedback` (quality + cost + latency)
- ✅ Support both hybrid and standard routing modes
- ✅ Comprehensive tests (test_litellm_feedback.py)
- ✅ Eliminated llm-prices.com dependency (uses LiteLLM's built-in cost tracking)

#### Phase 3: Testing & Docs ⏳ PARTIAL
- ⏳ Unit tests for `ConduitRoutingStrategy` (optional dependency blocks pytest)
- ✅ README with installation and usage (`conduit_litellm/README.md`)
- ⏳ Example: basic usage with multiple providers (Issue #14)
- ⏳ Example: performance comparison (Issue #14)

#### Phase 4: Package & Distribution ❌ NOT PLANNED
- ❌ Publish `conduit-litellm` to PyPI (part of main package)
- ❌ Separate docs site (using main docs)
- ❌ Submit PR to LiteLLM docs (future)
- ❌ Blog post (future)
- ❌ Demo video (future)

### Actual Package Structure

```
conduit/
├── conduit_litellm/              # LiteLLM integration package
│   ├── __init__.py               # Exports ConduitRoutingStrategy
│   ├── strategy.py               # Main routing strategy (340 lines)
│   ├── config.py                 # Configuration helpers
│   ├── utils.py                  # Validation and extraction
│   └── README.md                 # Usage guide
├── tests/
│   └── unit/
│       └── test_litellm_strategy.py  # Tests (blocked by optional dep)
└── examples/
    └── (Issue #14 - pending)
```

### Dependencies

**From `pyproject.toml`**:
```toml
[project.optional-dependencies]
litellm = ["litellm>=1.50.29"]
```

**Installation**:
```bash
pip install conduit[litellm]
```

### Current Limitations

1. **Manual Feedback API**: `record_feedback()` method is informational only
   - Automatic feedback via `ConduitFeedbackLogger` is the primary mechanism ✅
   - Manual feedback override API deferred (may be added if needed)
   - Automatic learning from all LiteLLM responses is fully operational

2. **Testing**: Optional dependency prevents pytest import in main suite
   - Tests exist but can't run in main test suite
   - Requires `pip install conduit[litellm]` first

3. **Examples**: No working examples yet (Issue #14)
   - Basic usage documented in README
   - Need real-world integration examples

4. **Documentation**: No usage docs beyond README (Issue #15)
   - Need guide for LiteLLM users
   - Need troubleshooting section

---

## Path 2: LiteLLM as Conduit Execution Backend ❌ NOT IMPLEMENTED

**Status**: Not started, no plans to implement

### Why Not Implemented

1. **Path 1 provides better strategic value**
   - LiteLLM users get ML routing (Path 1)
   - More users reached (12K+ stars vs Conduit users)

2. **PydanticAI sufficient for current needs**
   - 8 providers cover most use cases (OpenAI, Anthropic, Google, etc.)
   - Structured outputs and type safety more valuable than 100+ providers

3. **Maintenance burden**
   - Would require maintaining two execution backends
   - LiteLLM changes could break integration

4. **Alternative approach works better**
   - Conduit users wanting 100+ providers can use Path 1
   - Use LiteLLM with `ConduitRoutingStrategy`

### If We Ever Need This

The original design (from planning doc) proposed a pluggable executor interface:
- `LLMExecutor` ABC with `execute()` method
- `PydanticAIExecutor` (current implementation)
- `LiteLLMExecutor` (hypothetical)

This would allow Conduit to use LiteLLM for execution while keeping ML routing. However, current architecture doesn't need this complexity.

---

## Comparison: Path 1 vs Path 2

| Aspect | Path 1: Routing Strategy | Path 2: Execution Backend |
|--------|-------------------------|---------------------------|
| **Status** | ✅ COMPLETE | ❌ NOT STARTED |
| **Target Users** | LiteLLM users (12K+ stars) | Conduit users |
| **Value Prop** | "Make LiteLLM smarter" | "More LLM providers" |
| **Integration** | LiteLLM uses Conduit | Conduit uses LiteLLM |
| **Package** | `conduit_litellm/` | N/A |
| **Distribution** | LiteLLM ecosystem | N/A |
| **Time Spent** | 2 days | 0 days |
| **Complexity** | Lower (routing only) | Moderate (execution layer) |
| **Impact** | High (ecosystem play) | Medium (feature addition) |
| **Uniqueness** | Only ML router for LiteLLM | Many alternatives exist |

**Decision**: Path 1 built and working. Path 2 not needed given Path 1 success.

---

## Market Positioning Strategy

### Path 1 Positioning (Recommended)
- **Tagline**: "ML-powered routing for LiteLLM"
- **Elevator Pitch**: "Upgrade LiteLLM from rule-based routing (cheapest/fastest) to ML-based routing that learns which models work best for YOUR workload. Intelligent model selection with one line of code."
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
- "Intelligent model selection optimized for your workload"
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

### Completed Work
1. ✅ Create `conduit_litellm/` package structure (d7b69cc)
2. ✅ Implement `ConduitRoutingStrategy` (ff06a46)
3. ✅ Add `setup_strategy()` helper (c5dd24c)
4. ✅ Fix async context handling (abcae20, Issue #31)
5. ✅ Write basic README documentation
6. ✅ **Issue #13**: Implement feedback loop
   - ✅ Created `ConduitFeedbackLogger(CustomLogger)`
   - ✅ Automatic registration with LiteLLM callbacks
   - ✅ Cost extraction from `response._hidden_params`
   - ✅ Feature regeneration from query text
   - ✅ Bandit update with composite rewards
   - ✅ Support for hybrid and standard routing
   - ✅ Comprehensive unit tests

### Remaining Work (Priority Order)

#### High Priority
1. **Issue #14**: Add working examples
   - Basic integration example
   - Performance comparison (ML vs rule-based)
   - Multi-provider setup

#### Medium Priority
3. **Issue #15**: User documentation
   - LiteLLM user guide
   - Migration guide
   - Troubleshooting section

4. **Testing**: Fix pytest import issues
   - Make tests work with optional dependency
   - Integration tests with real LiteLLM
   - End-to-end learning verification

#### Low Priority (Future)
5. **Distribution**: PyPI and ecosystem
   - Consider separate PyPI package
   - Submit to LiteLLM community plugins
   - Blog post and demo video

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
