# Pre-1.0 Launch Roadmap

**Goal**: Ship production-ready Conduit 1.0 to PyPI and launch on HN/Reddit

**Status**: 5 tasks remaining before 1.0 launch

---

## Work Order

### 1. #102 - Restructure README for HN launch ⏳
**Priority**: CRITICAL | **Difficulty**: Beginner | **Estimate**: 2-3 hours

**Why**: First impressions matter for user adoption. Better README = more beta testers.

**Deliverables**:
- [ ] Lead with value: "Cut LLM costs 30-50% with 5 lines of code"
- [ ] Add "Why Conduit?" section with quantified benefits
- [ ] Create "Starter vs Full" paths (quick start vs production)
- [ ] Reduce jargon, add analogies
- [ ] Add "When NOT to use" section
- [ ] Add comparison table: Conduit vs LiteLLM vs LangChain vs RouteLLM

**Success Criteria**: External reviewer can evaluate in <5 minutes without API keys

---

### 2. #113 - Add load testing suite with locust or k6
**Priority**: HIGH | **Difficulty**: Intermediate | **Estimate**: 3-4 hours

**Why**: Validate production performance claims before launch.

**Deliverables**:
- [ ] Choose tool: locust or k6
- [ ] Create load test scenarios (100 QPS, 1000 QPS, burst)
- [ ] Test routing latency under load
- [ ] Test database connection pool behavior
- [ ] Test Redis cache performance
- [ ] Document baseline performance in README
- [ ] Add to CI for regression detection

**Success Criteria**: Can sustain 1000 QPS with <50ms p99 latency

---

### 3. #128 - Add decision audit log for compliance and debugging
**Priority**: HIGH | **Difficulty**: Intermediate | **Estimate**: 4-5 hours

**Why**: Enterprise compliance and production debugging.

**Deliverables**:
- [ ] Create `audit_log` table schema
- [ ] Log every routing decision (model selected, confidence, metadata)
- [ ] Add query API for audit log retrieval
- [ ] Add retention policy configuration
- [ ] Document compliance use cases
- [ ] Add performance tests (audit logging overhead)

**Success Criteria**: Audit log adds <2ms overhead to routing

---

### 4. #117 - Add graceful shutdown handling
**Priority**: MEDIUM | **Difficulty**: Intermediate | **Estimate**: 2-3 hours

**Why**: Production safety - prevents weight loss on container restarts.

**Deliverables**:
- [ ] Register SIGTERM/SIGINT signal handlers
- [ ] Call `router.close()` on shutdown (saves final state)
- [ ] Add graceful drain period (finish in-flight requests)
- [ ] Test with Docker/k8s deployments
- [ ] Document shutdown behavior in OPERATIONS.md

**Success Criteria**: Zero weight loss on graceful shutdown

---

### 5. #111 - Establish release automation process
**Priority**: CRITICAL | **Difficulty**: Intermediate | **Estimate**: 4-6 hours

**Why**: Can't ship 1.0 without automated releases to PyPI.

**Deliverables**:
- [ ] Create `.github/workflows/release.yml`
- [ ] Configure PyPI trusted publisher
- [ ] Add version bump tooling (semantic-release or bump2version)
- [ ] Create `docs/RELEASING.md` process guide
- [ ] Test release to TestPyPI
- [ ] Automate changelog generation
- [ ] Create release checklist template

**Success Criteria**: Can publish to PyPI with one command

---

## Timeline Estimate

**Total**: ~16-21 hours of work

**Aggressive**: 1 week (3-4 hours/day)
**Comfortable**: 2 weeks (2 hours/day)

---

## Launch Readiness Checklist

**Technical**:
- [x] 100% test pass rate (579/579 tests)
- [x] 81% code coverage
- [x] CI/CD pipeline automated
- [x] State persistence working
- [x] Integration tests comprehensive
- [ ] Load testing complete
- [ ] Audit logging implemented
- [ ] Graceful shutdown working
- [ ] Release automation working

**Documentation**:
- [ ] README restructured for HN launch
- [ ] OPERATIONS.md for production deployments
- [ ] RELEASING.md for maintainers
- [ ] Performance benchmarks documented
- [ ] Comparison to alternatives clear

**Community**:
- [ ] HN launch post drafted
- [ ] Reddit r/LocalLLaMA post drafted
- [ ] Twitter announcement prepared
- [ ] Discord/GitHub Discussions ready

---

## Post-1.0 (Deferred)

These can wait until after 1.0 launch:

- #114 - Chaos testing (can do post-launch with user feedback)
- #99 - Zero-config demo (nice-to-have)
- #100 - Benchmark scripts (nice-to-have)
- #103 - HN launch prep docs (do during #102)
- All LONG-TERM issues (defer to 1.1+)

---

**Last Updated**: 2025-11-27
**Next Task**: #102 - README restructure
