# Success Metrics - Conduit Routing System
**Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: Initial definition (Strategic Decision)
**Source**: Business panel analysis - notes/2025-11-18_business_panel_analysis.md

---

## Purpose

Define measurable success criteria for Conduit's ML-powered routing system across technical performance, customer outcomes, and business objectives.

**Strategic Context**: "Management priority: Define success metrics NOW" - Peter Drucker framework analysis

---

## Technical Success Metrics

### 1. Model Convergence
**Definition**: ML algorithm achieves stable, reliable routing decisions

**Metrics**:
- **Alpha/Beta Stability**: Beta distribution parameters stabilize (change <5% over 100 queries)
- **Convergence Threshold**: Model parameters converge within **1,000 queries** per model
- **Confidence Distribution**: ≥95% of routing decisions have confidence ≥0.6

**Measurement**:
```python
# Check convergence
def is_converged(model_state: ModelState, window_size: int = 100) -> bool:
    """Check if model has converged (parameters stable)."""
    # Compare current alpha/beta to alpha/beta from 100 queries ago
    # Return True if change < 5%
```

**Target**: All models converge within 1,000 queries in production workload

---

### 2. System Performance
**Definition**: Routing infrastructure meets latency and reliability requirements

**Metrics**:
- **p50 Latency**: <100ms for routing decision (embed + bandit + select)
- **p99 Latency**: <200ms for routing decision
- **Availability**: 99.9% uptime (excluding LLM provider outages)
- **Error Rate**: <0.1% routing failures (excluding constraint violations)

**Measurement**:
- Prometheus metrics: `conduit_routing_latency_seconds{quantile="0.99"}`
- Uptime monitoring: 99.9% = max 43 minutes downtime/month

**Target**: Meet all latency and reliability thresholds in production

---

### 3. Data Quality
**Definition**: Training data is sufficient and representative

**Metrics**:
- **Coverage**: ≥100 queries per model before production routing
- **Diversity**: Query embeddings span ≥80% of embedding space
- **Feedback Rate**: ≥30% of queries receive explicit or implicit feedback
- **Signal Quality**: <5% contradictory feedback (same query, different ratings)

**Measurement**:
```sql
-- Check feedback coverage
SELECT
    COUNT(DISTINCT r.id) as total_responses,
    COUNT(DISTINCT f.response_id) as feedback_count,
    (COUNT(DISTINCT f.response_id)::float / COUNT(DISTINCT r.id)) as feedback_rate
FROM responses r
LEFT JOIN feedback f ON r.id = f.response_id
WHERE r.created_at > NOW() - INTERVAL '7 days';
```

**Target**: Maintain ≥30% feedback rate week-over-week

---

## Customer Success Metrics

### 4. Cost Savings
**Definition**: Customer achieves measurable LLM cost reduction

**Metrics**:
- **First Month Savings**: >30% cost reduction vs baseline routing
- **Sustained Savings**: >25% cost reduction maintained beyond month 1
- **Payback Period**: Customer recoups Conduit costs within 30 days

**Baseline Definition**:
- Static routing: GPT-4o for all queries (most common current approach)
- Alternative: Customer's pre-Conduit routing strategy

**Measurement**:
```python
def calculate_savings(customer_id: str, period_days: int = 30) -> float:
    """Calculate cost savings vs static GPT-4o baseline."""
    actual_cost = sum(response.cost for response in get_responses(customer_id, period_days))

    # Baseline: GPT-4o for all queries
    baseline_cost = len(get_queries(customer_id, period_days)) * GPT4O_AVERAGE_COST

    savings_pct = (baseline_cost - actual_cost) / baseline_cost
    return savings_pct
```

**Target**: >30% savings for ≥80% of customers in first 30 days

---

### 5. Quality Maintenance
**Definition**: Cost savings don't come at expense of response quality

**Metrics**:
- **Quality Baseline**: 95% of queries meet or exceed quality threshold
- **User Satisfaction**: Average user_rating ≥4.0/5.0 stars
- **Expectation Met Rate**: ≥90% of feedback has met_expectations=True
- **Quality Degradation**: <5% decrease in avg quality_score vs month 1

**Quality Threshold**:
- Explicit: quality_score ≥0.7 (user-provided)
- Implicit: No retry detected AND no error occurred AND latency accepted

**Measurement**:
```sql
-- Check quality maintenance
SELECT
    AVG(quality_score) as avg_quality,
    AVG(user_rating) as avg_rating,
    SUM(CASE WHEN met_expectations THEN 1 ELSE 0 END)::float / COUNT(*) as expectation_rate
FROM feedback
WHERE created_at > NOW() - INTERVAL '30 days';
```

**Target**: Maintain quality baseline while achieving cost savings

---

### 6. Time to Value
**Definition**: How quickly customer sees benefits from Conduit

**Metrics**:
- **First Insight**: Routing analytics available within 24 hours
- **First Optimization**: Measurable cost savings within 7 days
- **Full Convergence**: Stable routing performance within 30 days
- **ROI Positive**: Cost savings exceed Conduit fees within 30 days

**Measurement**:
- Track time from first query to first cost savings detection
- Monitor adoption curve: queries/day growth rate

**Target**: 80% of customers see cost savings within 7 days

---

## Business Success Metrics

### 7. Customer Retention
**Definition**: Customers continue using Conduit beyond trial period

**Metrics**:
- **30-Day Retention**: ≥80% of trial customers convert to paid
- **90-Day Retention**: ≥90% of paid customers remain active
- **Churn Rate**: <5% monthly churn (excluding seasonal/project-based usage)
- **Expansion**: ≥30% of customers increase query volume month-over-month

**Measurement**:
- Cohort analysis: Track customers by signup month
- Define "active": ≥100 queries/month

**Target**: <5% monthly churn, 30% expansion rate

---

### 8. Product-Market Fit
**Definition**: Strong demand and customer satisfaction signals

**Metrics**:
- **NPS Score**: ≥50 (promoters - detractors)
- **Organic Growth**: ≥30% of new customers from referrals
- **Usage Intensity**: Average customer processes ≥1,000 queries/week
- **Feature Adoption**: ≥60% of customers use constraint-based routing

**Measurement**:
- Quarterly NPS surveys
- Track referral source in signup flow
- Monitor feature usage in analytics

**Target**: NPS ≥50, 30% referral growth rate

---

## Quality Gate Thresholds

### Launch Gates (Before Production Release)

**Phase 1 (MVP)**: Rule-based router + basic ML
- ✅ Technical: p99 latency <200ms
- ✅ Technical: Error rate <0.1%
- ⏳ Data: ≥100 queries per model in test dataset
- ⏳ Quality: 95% test queries have confidence ≥0.6

**Phase 2 (API)**: FastAPI endpoints + implicit feedback
- ⏳ Technical: All Phase 1 metrics maintained
- ⏳ Customer: ≥5 beta customers with 30-day retention
- ⏳ Business: Beta customers report >25% cost savings
- ⏳ Quality: Implicit feedback system operational

**Phase 3 (Scale)**: Production-ready infrastructure
- ⏳ Technical: 99.9% availability over 30 days
- ⏳ Customer: >30% cost savings for ≥80% of customers
- ⏳ Business: <5% monthly churn rate
- ⏳ Quality: NPS ≥50 from customer surveys

---

## Monitoring & Alerting

### Critical Alerts (Page Immediately)
- p99 latency >500ms for >5 minutes
- Error rate >1% for >5 minutes
- Availability <99% over 1 hour
- Database connection pool exhausted

### Warning Alerts (Investigate Within 1 Hour)
- p99 latency >300ms for >15 minutes
- Average quality_score <0.6 for new feedback
- Feedback rate drops below 20% week-over-week
- Customer churn detected (no queries in 7 days)

### Informational Alerts (Daily Digest)
- Model convergence status updates
- Cost savings leaderboard (top performing customers)
- Feature adoption trends
- System performance summary

---

## Reporting Cadence

### Real-Time Dashboards
- System health (latency, errors, availability)
- Active query volume and routing decisions
- Model confidence distribution
- Current cost savings vs baseline

### Daily Reports
- Yesterday's query volume and cost savings
- Model convergence progress
- Quality score distribution
- Customer usage patterns

### Weekly Reviews
- Customer success metrics (savings, quality, retention)
- Model performance trends
- Feature adoption rates
- Incident post-mortems

### Monthly Business Reviews
- Customer retention and churn analysis
- Product-market fit indicators (NPS, referrals)
- Strategic metric trends vs targets
- Roadmap prioritization based on data

---

## Success Metric Hierarchy

```
Business Outcomes (What customers pay for)
├─ Cost Savings (>30% reduction)
├─ Quality Maintenance (95% meet threshold)
└─ Time to Value (<7 days to first savings)
    │
    ├─ Customer Engagement (Feedback, retention)
    │   └─ Technical Performance (Latency, reliability)
    │       └─ Model Quality (Convergence, confidence)
    │           └─ Data Quality (Coverage, feedback rate)
```

**Priority**: Focus on business outcomes first, then work backward to technical metrics

---

## Appendix: Calculation Examples

### Cost Savings Calculation
```python
# Example: Customer with 10,000 queries/month
queries = 10_000

# Baseline: All queries to GPT-4o
baseline_cost = queries * 0.015  # $0.015 avg per query
# = $150/month

# Conduit routing (example distribution)
# 60% routed to gpt-4o-mini ($0.0003/query)
# 30% routed to gpt-4o ($0.015/query)
# 10% routed to claude-sonnet-4 ($0.009/query)
conduit_cost = (
    (queries * 0.6 * 0.0003) +
    (queries * 0.3 * 0.015) +
    (queries * 0.1 * 0.009)
)
# = $1.80 + $45 + $9 = $55.80/month

savings = (baseline_cost - conduit_cost) / baseline_cost
# = ($150 - $55.80) / $150 = 62.8% savings ✅

# Exceeds 30% target
```

### Quality Baseline Calculation
```python
# Example: 100 feedback entries
feedback_entries = 100

# Quality distribution
high_quality = 75  # quality_score >= 0.7
medium_quality = 20  # quality_score 0.4-0.7
low_quality = 5  # quality_score < 0.4

quality_baseline_met = high_quality / feedback_entries
# = 75 / 100 = 75%

# Falls short of 95% target ⚠️
# Action: Investigate low quality queries, adjust routing
```

---

**Review Schedule**: Update metrics quarterly based on customer feedback and business needs
**Next Review**: 2026-02-18
