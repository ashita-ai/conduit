# Conduit Documentation

Comprehensive technical documentation for understanding and working with Conduit.

---

## Quick Start Paths

### 🚀 New to Conduit?

**Learn the Basics** (15 minutes):
1. [BANDIT_TRAINING.md](BANDIT_TRAINING.md) - Understand how learning works (Thompson Sampling ≠ LLM Training)
2. [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) - See what models are available (71+ models, auto-detected)
3. `../examples/01_quickstart/hello_world.py` - Run your first query (5 lines of code)

**Key Insight**: Zero-shot deployment. No pre-training required!

---

### 🎯 Want to Optimize Performance?

**Reduce Sample Requirements** (30 minutes):
1. [COLD_START.md](COLD_START.md) - Understand the cold start problem
2. [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - Implement UCB1→LinUCB warm start (30% faster)
3. [PCA_GUIDE.md](PCA_GUIDE.md) - Add dimensionality reduction (75% sample reduction)
4. `../examples/04_pca/` - Working examples

**Combined Impact**: 1,500-2,500 queries to production (vs 10,000+ baseline)

---

### 🔌 Want to Integrate?

**LiteLLM Plugin** (20 minutes):
1. [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) - Understand the plugin strategy
2. `../examples/04_litellm/demo.py` - Working integration example
3. `../docker-compose.openwebui.yml` - Full stack with Open WebUI

**Result**: Access 100+ LLM providers with ML-powered routing

---

### 📊 Want to Prove Value?

**Cost Savings Demonstration** (1 hour):
1. [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Multi-baseline methodology
2. Create 1,000-query workload matching your use case
3. Run comparison: Always Premium vs Manual vs Conduit
4. Generate report with empirical cost savings

**Expected Results**: 60% savings vs premium, 16% vs manual routing

---

## Core Concepts

### [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) - Dynamic Pricing & Model Discovery

**Key Topics**:
- Auto-fetch 71+ models from llm-prices.com (24h cache)
- `supported_models()` - see what Conduit can use
- `available_models()` - see what YOU can use (auto-detects API keys)
- Zero maintenance (no hard-coded pricing tables)
- Graceful degradation with fallback pricing

**Read this first if**: You want to understand model availability and pricing

---

### [BANDIT_TRAINING.md](BANDIT_TRAINING.md) - How Learning Works

**Key Topics**:
- Thompson Sampling ≠ LLM Fine-Tuning (fundamentally different!)
- Online learning from feedback (no pre-training needed)
- Learning phases: Cold Start → Learning → Converged
- Feedback signals: Explicit (ratings) + Implicit (behavior)
- The "Data Moat" competitive advantage

**Read this if**: You want to understand how Conduit learns from usage

---

### [COLD_START.md](COLD_START.md) - The Cold Start Problem

**Key Topics**:
- Problem definition: Making good decisions before learning
- 7 solution approaches with pros/cons
- Implemented solutions: Hybrid Routing + PCA (85-90% sample reduction)
- Recommended strategy: Informed Priors + Contextual Heuristics
- Expected convergence in 1,500-2,500 queries (vs 10,000+ without)

**Read this if**: You want to minimize poor routing during initial queries

---

### [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Proving Cost Savings

**Key Topics**:
- Multi-baseline approach (Always Premium, Manual Routing, Random)
- 1000-query workload design
- Expected results: 60% savings vs premium, 16% vs manual
- Quality validation methodology
- Report generation

**Read this if**: You want to demonstrate empirical cost savings

---

## System Architecture

### [ARCHITECTURE.md](ARCHITECTURE.md) - System Design

High-level system architecture and component interactions.

**Covers**:
- Layer architecture (API → Router → Executor → Providers)
- Database schema (PostgreSQL)
- Caching strategy (Redis)
- Feedback collection and integration

---

### [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) - Observability Trinity

Detailed documentation of the implicit feedback system.

**Three Detection Systems**:
1. **Error Detection** - Model failures, empty responses, error patterns
2. **Latency Tracking** - User patience tolerance categorization
3. **Retry Detection** - Semantic similarity-based duplicate detection

**Weighting**: 70% explicit feedback + 30% implicit signals

---

### [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) - Algorithm Reference

Comprehensive documentation of all bandit algorithms.

**Algorithms**:
- **Contextual Thompson Sampling** - Bayesian linear regression (best for diverse queries)
- **LinUCB** - Ridge regression with UCB (best for context-aware routing)
- **Thompson Sampling** - Beta distributions (good exploration/exploitation)
- **UCB1** - Upper confidence bounds (fast, non-contextual)
- **Epsilon-Greedy** - Simple exploration (baseline)
- **Hybrid Routing** - UCB1→LinUCB warm start (30% faster convergence)

**Related**: [COLD_START.md](COLD_START.md) for sample efficiency strategies

---

### [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) - LiteLLM Integration Strategy

Strategic analysis and implementation plans for LiteLLM integration.

**Paths**:
- **Path 1** (Recommended): Conduit as LiteLLM routing strategy plugin
- **Path 2**: LiteLLM as Conduit execution backend

**Benefits**:
- Access to 100+ providers through LiteLLM ecosystem
- OpenAI-compatible API
- Docker Compose setup with Open WebUI

---

## Sample Efficiency Guides

### [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - UCB1→LinUCB Warm Start

**Performance**: 30% faster convergence than pure LinUCB

**Strategy**:
1. Phase 1 (0-2,000 queries): UCB1 (non-contextual, fast exploration)
2. Phase 2 (2,000+ queries): LinUCB (contextual, warm-started from UCB1)

**Sample Requirements**: 2,000-3,000 queries vs 10,000+ for pure LinUCB

---

### [PCA_GUIDE.md](PCA_GUIDE.md) - Dimensionality Reduction

**Performance**: 75% sample reduction for LinUCB convergence

**Compression**: 387 → 67 dimensions (82% reduction)
- 384 embedding dims → 64 PCA components (95%+ variance)
- 3 metadata dims → preserved (tokens, complexity, confidence)

**Sample Requirements**: 17,000 queries vs 68,000 for full features

---

### Combined Impact

| Approach | Sample Requirement | Improvement |
|----------|-------------------|-------------|
| Pure LinUCB (387d) | 10,000-15,000 queries | Baseline |
| Hybrid (387d) | 2,000-3,000 queries | 70-85% reduction |
| PCA (67d) | 3,000-5,000 queries | 60-70% reduction |
| **Hybrid + PCA (67d)** | **1,500-2,500 queries** | **85-90% reduction** |

---

## Quick Navigation

### I want to understand...

**...what models I can use**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md)

**...how pricing stays current**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (Dynamic Pricing section)

**...how Conduit learns**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md)

**...why it doesn't need pre-training**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md) (Thompson Sampling ≠ LLM Fine-Tuning section)

**...how to reduce cold start problems**: → [COLD_START.md](COLD_START.md)

**...sample efficiency (PCA, Hybrid Routing)**: → [HYBRID_ROUTING.md](HYBRID_ROUTING.md) + [PCA_GUIDE.md](PCA_GUIDE.md)

**...how to prove cost savings**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md)

**...how implicit feedback works**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md)

**...the bandit algorithms**: → [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md)

**...LiteLLM integration options**: → [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md)

**...the overall system design**: → [ARCHITECTURE.md](ARCHITECTURE.md)

---

### I want to implement...

**...model discovery**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (API Reference) + `examples/01_quickstart/model_discovery.py`

**...hybrid routing**: → [HYBRID_ROUTING.md](HYBRID_ROUTING.md) + `Router(use_hybrid=True)`

**...PCA reduction**: → [PCA_GUIDE.md](PCA_GUIDE.md) + `examples/04_pca/pca_setup.py`

**...LiteLLM integration**: → [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) + `examples/04_litellm/demo.py`

**...a benchmark comparison**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) (Execution Flow)

**...implicit feedback collection**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) (Usage Examples)

---

## Implementation Paths

### Path 1: Basic Routing (15 minutes)
```
Read: BANDIT_TRAINING.md → MODEL_DISCOVERY.md
Code: examples/01_quickstart/hello_world.py
Test: Run your first query
```

### Path 2: Optimized Routing (1 hour)
```
Read: COLD_START.md → HYBRID_ROUTING.md → PCA_GUIDE.md
Code: examples/04_pca/pca_setup.py (one-time training)
Code: Router(use_hybrid=True, use_pca=True)
Test: Verify 85-90% sample reduction
```

### Path 3: LiteLLM Integration (30 minutes)
```
Read: LITELLM_INTEGRATION.md
Code: examples/04_litellm/demo.py
Deploy: docker-compose.openwebui.yml (full stack)
Test: Access via OpenAI-compatible API
```

### Path 4: Production Deployment (2 hours)
```
Read: ARCHITECTURE.md → IMPLICIT_FEEDBACK.md
Setup: PostgreSQL + Redis
Configure: Environment variables (.env)
Deploy: Docker or direct Python
Monitor: Quality, cost, latency metrics
```

---

## Key Insights

### Thompson Sampling vs LLM Training

| Aspect | LLMs | Bandits |
|--------|------|---------|
| Training | Offline | Online |
| Data | Labeled corpus | Real-time feedback |
| Cost | $100s-$1000s | Negligible |
| Pre-training | Required | NOT needed |

**Bottom Line**: Zero-shot deployment - no training phase required!

---

### Cold Start Solutions (Recommended)

**Implemented** (Production-Ready):
1. **Hybrid Routing** - UCB1→LinUCB warm start (30% faster)
2. **PCA Reduction** - 75% sample reduction

**Combined**: 1,500-2,500 queries to production (vs 10,000+ baseline)

**Future** (See COLD_START.md):
- Informed Priors (industry/domain knowledge)
- Contextual Heuristics (query feature-based routing)
- Transfer Learning (share learnings across customers)

---

### Benchmark Strategy

**Three Baselines**:
- Always Premium (GPT-4o): $4.50/1K queries
- Manual Routing: $1.96/1K queries
- Conduit (Learned): $1.65/1K queries

**Value Props**:
- "63% cheaper than always-premium"
- "16% cheaper than manual routing"
- "95%+ quality maintained"

---

## Documentation Status

| Document | Lines | Last Updated | Status |
|----------|-------|--------------|--------|
| HYBRID_ROUTING.md | 461 | 2025-01-22 | ✅ Complete |
| PCA_GUIDE.md | 574 | 2025-01-22 | ✅ Complete |
| BANDIT_ALGORITHMS.md | 1,084 | 2025-11-22 | ✅ Complete |
| COLD_START.md | 640 | 2025-01-22 | ✅ Complete |
| MODEL_DISCOVERY.md | 584 | 2025-11-20 | ✅ Complete |
| LITELLM_INTEGRATION.md | 591 | 2025-11-21 | ✅ Complete |
| BANDIT_TRAINING.md | 443 | 2025-11-19 | ✅ Complete |
| BENCHMARK_STRATEGY.md | 584 | 2025-11-19 | ✅ Complete |
| IMPLICIT_FEEDBACK.md | 516 | 2025-11-19 | ✅ Complete |
| ARCHITECTURE.md | 897 | 2025-11-18 | ✅ Complete |

---

## Contributing

When updating documentation:

1. **Keep it practical**: Focus on implementation, not just theory
2. **Use code examples**: Show don't tell
3. **Update README.md**: Add your new doc to the navigation
4. **Cross-reference**: Add "See Also" sections linking related docs
5. **Date your updates**: Keep "Last Updated" current

---

## Questions?

- **Technical Implementation**: See `../CLAUDE.md` in project root
- **Development Workflow**: See `../CLAUDE.md` in project root
- **Strategic Decisions**: See `../notes/` directory
