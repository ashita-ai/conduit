# Conduit Documentation

Comprehensive technical documentation for understanding and working with Conduit.

## Core Concepts

### [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) - Dynamic Pricing & Model Discovery

**Key Topics**:
- Auto-fetch 71+ models from llm-prices.com (24h cache)
- `supported_models()` - see what Conduit can use
- `available_models()` - see what YOU can use (auto-detects API keys)
- Zero maintenance - no hard-coded pricing tables
- Graceful degradation with fallback pricing

**Read this first if**: You want to understand model availability and pricing

### [BANDIT_TRAINING.md](BANDIT_TRAINING.md) - How Learning Works

**Key Topics**:
- Thompson Sampling ≠ LLM Fine-Tuning (fundamentally different!)
- Online learning from feedback (no pre-training needed)
- Learning phases: Cold Start → Learning → Converged
- Feedback signals: Explicit (ratings) + Implicit (behavior)
- The "Data Moat" competitive advantage

**Read this if**: You want to understand how Conduit learns from usage

### [COLD_START.md](COLD_START.md) - The Cold Start Problem

**Key Topics**:
- Problem definition: Making good decisions before learning
- 7 solution approaches with pros/cons
- Recommended strategy: Informed Priors + Contextual Heuristics
- Implementation roadmap
- Expected convergence in 50-100 queries (vs 200 without)

**Read this if**: You want to minimize poor routing during initial queries

### [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Proving Cost Savings

**Key Topics**:
- Multi-baseline approach (Always Premium, Manual Routing, Random)
- 1000-query workload design
- Expected results: 60% savings vs premium, 16% vs manual
- Quality validation methodology
- Report generation

**Read this if**: You want to demonstrate empirical cost savings

## System Architecture

### [ARCHITECTURE.md](ARCHITECTURE.md) - System Design

High-level system architecture and component interactions.

### [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) - Observability Trinity

Detailed documentation of the implicit feedback system:
- Error detection (model failures, empty responses, error patterns)
- Latency tracking (user patience tolerance)
- Retry detection (semantic similarity-based)

### [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) - Algorithm Reference

Comprehensive documentation of all bandit algorithms:
- Contextual Thompson Sampling (Bayesian linear regression)
- LinUCB (ridge regression with UCB)
- Thompson Sampling (Beta distributions)
- UCB1 (upper confidence bounds)
- Epsilon-Greedy (exploration vs exploitation)

### [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) - LiteLLM Integration Strategy

Strategic analysis and implementation plans for LiteLLM integration:
- **Path 1** (Recommended): Conduit as LiteLLM routing strategy plugin
- **Path 2**: LiteLLM as Conduit execution backend
- Market positioning, technical architecture, implementation plans
- Access to 100+ providers through LiteLLM ecosystem

## Design Documents

### [DESIGN_SPEC.md](DESIGN_SPEC.md) - Original Design

Original design specification and requirements.

### [success_metrics.md](success_metrics.md) - Success Criteria

Metrics and KPIs for measuring Conduit's effectiveness.

### [PRICING_UPDATES.md](PRICING_UPDATES.md) - Pricing Infrastructure

Model pricing table and automated sync workflow.

## Quick Navigation

### I want to understand...

**...what models I can use**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md)

**...how pricing stays current**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (Dynamic Pricing section)

**...how Conduit learns**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md)

**...why it doesn't need pre-training**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md) (Thompson Sampling ≠ LLM Fine-Tuning section)

**...how to reduce cold start problems**: → [COLD_START.md](COLD_START.md)

**...how to prove cost savings**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md)

**...how implicit feedback works**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md)

**...the bandit algorithms**: → [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md)

**...LiteLLM integration options**: → [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md)

**...the overall system design**: → [ARCHITECTURE.md](ARCHITECTURE.md)

### I want to implement...

**...model discovery**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (API Reference)

**...API key auto-detection**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (available_models)

**...informed priors**: → [COLD_START.md](COLD_START.md) (Solution 1)

**...contextual heuristics**: → [COLD_START.md](COLD_START.md) (Solution 2)

**...a benchmark comparison**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) (Execution Flow)

**...implicit feedback collection**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) (Usage Examples)

## Key Insights

### Thompson Sampling vs LLM Training

| Aspect | LLMs | Bandits |
|--------|------|---------|
| Training | Offline | Online |
| Data | Labeled corpus | Real-time feedback |
| Cost | $100s-$1000s | Negligible |
| Pre-training | Required | NOT needed |

**Bottom Line**: Zero-shot deployment - no training phase required!

### Cold Start Solutions (Recommended)

**Tier 1** (Implement Now):
1. Informed Priors - Start with reasonable expectations
2. Contextual Heuristics - Use query features for initial routing

**Expected Results**: 50-100 query convergence (vs 200 without)

### Benchmark Strategy

**Three Baselines**:
- Always Premium (GPT-4o): $4.50/1K queries
- Manual Routing: $1.96/1K queries
- Conduit (Learned): $1.65/1K queries

**Value Props**:
- "63% cheaper than always-premium"
- "16% cheaper than manual routing"
- "95%+ quality maintained"

## Documentation Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| LITELLM_INTEGRATION.md | ✅ Complete | 2025-11-21 |
| BANDIT_ALGORITHMS.md | ✅ Complete | 2025-11-21 |
| MODEL_DISCOVERY.md | ✅ Complete | 2025-11-20 |
| BANDIT_TRAINING.md | ✅ Complete | 2025-11-19 |
| COLD_START.md | ✅ Complete | 2025-11-19 |
| BENCHMARK_STRATEGY.md | ✅ Complete | 2025-11-19 |
| IMPLICIT_FEEDBACK.md | ✅ Complete | 2025-11-19 |
| ARCHITECTURE.md | ✅ Complete | 2025-11-18 |
| DESIGN_SPEC.md | ✅ Complete | 2025-11-18 |
| success_metrics.md | ⏳ Needs Update | 2025-11-18 |
| PRICING_UPDATES.md | ⚠️ Deprecated | 2025-11-18 |

## Contributing

When updating documentation:

1. **Keep it practical**: Focus on implementation, not just theory
2. **Use code examples**: Show don't tell
3. **Update README.md**: Add your new doc to the navigation
4. **Cross-reference**: Link related docs
5. **Date your updates**: Keep "Last Updated" current

## Questions?

- **Technical Implementation**: See CLAUDE.md in project root
- **Development Workflow**: See AGENTS.md in project root
- **Strategic Decisions**: See notes/ directory
