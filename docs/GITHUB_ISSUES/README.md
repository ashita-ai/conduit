# GitHub Issues for Advanced Bandit Algorithms

This directory contains detailed GitHub issue templates for implementing advanced bandit algorithms in Conduit.

## Issue Templates

### 1. [Dueling Bandits](./issue_dueling_bandits.md) - **HIGH PRIORITY**
**Status**: Ready to implement
**Priority**: High
**Complexity**: Medium

Implements Contextual Dueling Bandits (FGTS.CDB) for learning from pairwise preference feedback instead of absolute scores. This is the most immediately useful algorithm for production LLM routing.

**Key Benefits**:
- Easier feedback collection (pairwise comparisons vs absolute scores)
- Designed specifically for LLM routing
- Better user experience (no need for explicit quality scores)

### 2. [NeuralUCB](./issue_neural_ucb.md) - **MEDIUM PRIORITY**
**Status**: Ready to implement
**Priority**: Medium
**Complexity**: High

Implements NeuralUCB for capturing non-linear relationships between query features and model performance. Use when linear models (LinUCB) plateau.

**Key Benefits**:
- Captures complex feature interactions
- Better accuracy when linear models plateau
- More expressive reward modeling

**When to Use**: After LinUCB performance plateaus, when you have sufficient data

### 3. [PILOT-Style Routing](./issue_pilot_style.md) - **ADVANCED / FUTURE**
**Status**: Ready to implement
**Priority**: Advanced / Future
**Complexity**: Very High

Implements PILOT (Preference-Prior Informed LinUCB) for combining offline preference data with online bandit learning. Solves cold start problem.

**Key Benefits**:
- Faster cold start (50-80% reduction in queries to convergence)
- Leverages existing preference data
- Better generalization to unseen models

**When to Use**: When you have offline preference data, want faster cold start

## How to Use These Issues

1. **Copy issue content**: Open the markdown file and copy the content
2. **Create GitHub issue**: Go to https://github.com/[your-org]/conduit/issues/new
3. **Paste content**: Paste the markdown content into the issue
4. **Add labels**: Use the suggested labels from each issue
5. **Assign priority**: Set milestone/priority based on issue priority

## Implementation Order Recommendation

1. **First**: Dueling Bandits (easiest to collect feedback, immediate value)
2. **Second**: NeuralUCB (if LinUCB plateaus, have sufficient data)
3. **Third**: PILOT (if you have offline preference data, want faster cold start)

## Research Context

All algorithms are based on recent research papers (2024-2025) specifically for LLM routing:
- Dueling Bandits: arxiv:2510.00841
- NeuralUCB: arxiv:2012.01780
- PILOT: arxiv:2508.21141

See individual issue files for detailed references and implementation notes.

## Questions?

If you have questions about any of these algorithms or implementation approach:
1. Check the research papers linked in each issue
2. Review existing bandit implementations in `conduit/engines/bandits/`
3. See `docs/BANDIT_ALGORITHMS.md` for algorithm documentation

