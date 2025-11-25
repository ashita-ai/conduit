# Project Implementation Plan & Todo

## 🎯 Priority 1: Critical Fixes & Stability (Current Sprint)

### State Persistence (Issue #76)
- [ ] Implement `StatePersistenceService` to save/load bandit state
- [ ] Add Redis/PostgreSQL backend for persistence
- [ ] Ensure state is saved on shutdown and loaded on startup
- [ ] Verify hybrid router transition state is persisted

### Router Test Failures (Issue #77)
- [ ] Investigate 21 failing router tests
- [ ] Fix API contract mismatches in tests
- [ ] Ensure all tests pass with `uv run pytest tests/unit/test_router.py`

### Database Integration (Issue #6)
- [ ] Fix missing `tests/unit/test_database.py`
- [ ] Address 27 failing tests in database and model registry
- [ ] Verify database connection pooling and error handling

## 🚀 Priority 2: Refactoring & Technical Debt

### Service Layer Refactoring (Issue #79)
- [ ] Decouple `Router` from direct database access
- [ ] Create dedicated `RoutingService` and `FeedbackService`
- [ ] Implement `ResultType` registry for standardized outputs

### Configuration Refactoring (Issue #80)
- [x] Move hardcoded pricing to `conduit.yaml`
- [ ] Split `Settings` into logical groups (Database, Redis, ML, etc.)
- [ ] Add validation for all config sections

### Reward Calculation Simplification (Issue #81)
- [ ] Extract reward logic to `conduit.core.rewards`
- [ ] Add unit tests for reward calculation
- [ ] Document normalization functions clearly

## 🔮 Priority 3: Enhancements & Features

### Observability (Issue #46)
- [ ] Add OpenTelemetry tracing for routing requests
- [ ] Implement metrics for bandit performance (reward, regret)
- [ ] Add structured logging for all components

### Documentation
- [ ] Consolidate LiteLLM integration docs (Issue #15)
- [ ] Add "How to Contribute" guide
- [ ] Update API documentation with new endpoints

## ✅ Completed Items
- [x] Clean up duplicate files (Issue #78)
- [x] Remove deprecated `use_hybrid` parameter
- [x] Consolidate examples
- [x] Add context-specific priors
- [x] Implement failure scenario examples
