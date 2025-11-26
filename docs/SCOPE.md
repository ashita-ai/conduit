# Conduit Scope Boundaries

**Last Updated**: 2025-11-26
**Status**: Pre-1.0 (v0.0.4-alpha)

This document defines what Conduit does and doesn't do. Clear scope boundaries help users make informed decisions and protect long-term maintainability.

---

## What Conduit Does

### Core Routing Engine
- **ML-powered LLM routing** using contextual bandit algorithms (LinUCB, UCB1, Thompson Sampling, Epsilon-Greedy)
- **Multi-objective optimization** balancing quality, cost, and latency
- **Contextual decision-making** using query embeddings and metadata (387-dimensional feature vectors)
- **Adaptive learning** from explicit feedback (user ratings) and implicit signals (errors, latency, retries)

### State Management
- **Persistent algorithm state** across restarts using PostgreSQL
- **Routing history storage** for analysis and debugging
- **Model performance tracking** with cost, quality, and latency metrics
- **Dynamic model discovery** with automatic pricing updates

### Integration Support
- **Provider-agnostic** integration via PydanticAI (OpenAI, Anthropic, Google, any LiteLLM-compatible provider)
- **LiteLLM integration** with automatic feedback loop via `ConduitRoutingStrategy`
- **Optional Redis caching** for response caching and rate limiting (graceful degradation when unavailable)
- **FastAPI REST API** for programmatic routing requests

### Advanced Features
- **Hybrid routing** combining cheap models for classification with expensive models for execution (30% faster convergence)
- **Non-stationarity handling** with sliding windows for concept drift
- **PCA dimensionality reduction** for efficient exploration (75% sample reduction)
- **Arbiter LLM-as-judge evaluation** for automatic quality assessment with configurable sampling

---

## What Conduit Doesn't Do

### Out of Scope (By Design)

**Multi-Tenancy**
- No per-user routing policies
- No tenant isolation or user-specific model selection
- No usage quotas or billing per user
- Rationale: Single-tenant focus keeps complexity manageable. Users needing multi-tenancy should deploy multiple Conduit instances.

**Custom Scoring Functions**
- No user-defined reward functions beyond quality/cost/latency weights
- No pluggable scoring systems
- No custom exploration strategies
- Rationale: Pre-built scoring covers 95% of use cases. Custom scoring adds API surface area and testing complexity.

**Model Hosting**
- No model inference or serving
- No model deployment or versioning
- No GPU management
- Rationale: Conduit routes to existing model APIs, doesn't host them. Use model providers (OpenAI, Anthropic, self-hosted) for inference.

**Prompt Engineering**
- No prompt optimization or A/B testing
- No prompt versioning or templates
- No automatic prompt improvement
- Rationale: Prompt engineering is orthogonal to routing. Use tools like LangSmith, PromptLayer for prompt management.

**Authentication/Authorization**
- No user authentication
- No API key management
- No access control policies
- Rationale: Deploy Conduit behind your existing auth layer (API gateway, reverse proxy). Don't duplicate auth logic.

**Model Training**
- No model fine-tuning
- No custom model training
- No dataset management
- Rationale: Conduit learns routing policies, not model weights. Use model providers for fine-tuning.

**Real-Time Streaming**
- No streaming LLM responses
- No WebSocket support
- Batch-only processing
- Rationale: Streaming adds significant complexity. Most routing decisions are made upfront, not mid-stream.

**Frontend UI**
- No web dashboard
- No admin interface
- API-only (CLI for testing)
- Rationale: UI preferences vary widely. Provide API, let users build UI if needed. Focus on API stability over UI maintenance.

**Distributed Tracing (Deep Integration)**
- Basic OpenTelemetry support only
- No custom span enrichment
- No advanced distributed tracing features
- Rationale: Sufficient observability via standard metrics. Advanced tracing needs are deployment-specific.

**Advanced Constraints**
- No complex constraint solvers (e.g., "route to OpenAI OR Anthropic but not both")
- No multi-step constraint evaluation
- Basic provider/model inclusion/exclusion only
- Rationale: Simple constraints cover most needs. Complex logic can be implemented via pre-filtering before routing.

---

## Gray Areas (May Consider)

These features are not currently planned but might be considered based on user demand:

**Model Failover Strategies**
- Current: Random fallback on error
- Possible: Smart fallback based on provider health checks
- Decision criteria: User demand, implementation complexity

**Cost Budgets**
- Current: No budget enforcement
- Possible: Per-request or per-period cost limits
- Decision criteria: Real user cost management needs vs configuration complexity

**A/B Testing Support**
- Current: Single routing policy
- Possible: Split traffic between policies for comparison
- Decision criteria: Whether users can't do this outside Conduit

**Prompt Caching**
- Current: Full response caching only
- Possible: Prompt prefix caching for repeated prefixes
- Decision criteria: Provider API support stabilization

---

## Decision Framework

When evaluating new features, we ask:

1. **Alignment**: Does this align with "ML-powered routing for cost/quality/latency optimization"?
2. **Maintainability**: Can we maintain this long-term without increasing complexity burden?
3. **Orthogonality**: Should this be a separate tool/library instead?
4. **User Demand**: Do multiple users need this, or is it one-off?
5. **Alternatives**: Can users solve this outside Conduit with existing tools?

**Example Application**:
- Request: "Add support for custom reward functions"
- Alignment: ✅ Yes (optimization-related)
- Maintainability: ❌ No (testing surface area explodes)
- Orthogonality: ✅ Could be separate extension system
- User Demand: 🤔 Unknown (need validation)
- Alternatives: ✅ Users can post-process routing decisions
- **Decision**: Not in core, consider extension API if demand validated

---

## Scope Changes

This document may evolve as we learn from users. Changes require:
- Clear rationale for scope expansion/contraction
- Assessment against decision framework
- Update to this document with version and date
- Communication in release notes

**How to Propose Scope Changes**:
1. Open GitHub Discussion (not Issue) explaining use case
2. Demonstrate why existing features don't solve it
3. Show evidence of broader user need (not just your use case)
4. Wait for maintainer response before investing time in PR

---

## Resources

- **Architecture**: See `docs/ARCHITECTURE.md` for technical design
- **Configuration**: See `docs/CONFIGURATION.md` for available options
- **FAQ**: See `docs/FAQ.md` for common questions
- **Examples**: See `examples/` directory for usage patterns

---

**Questions about scope?** Open a GitHub Discussion (not Issue) for clarification.
