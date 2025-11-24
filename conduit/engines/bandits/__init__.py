"""Bandit algorithms for LLM routing.

This module contains multi-armed bandit implementations for intelligent
model selection. All algorithms share a common interface and work with
Conduit's QueryFeatures.

Available Algorithms:
- DuelingBandit: Pairwise preference learning (contextual, high sample efficiency)
- ContextualThompsonSamplingBandit: Bayesian linear regression with Thompson Sampling (contextual, optimal regret)
- ThompsonSamplingBandit: Bayesian probability matching (optimal regret)
- LinUCBBandit: Contextual linear UCB (uses query features, proven for LLM routing)
- UCB1Bandit: Upper Confidence Bound (optimal regret, fastest convergence)
- EpsilonGreedyBandit: Simple exploration-exploitation (suboptimal regret)
- RandomBaseline: Uniform random selection (lower bound)
- OracleBaseline: Perfect knowledge (upper bound, zero regret)
- AlwaysBestBaseline: Always use highest quality model
- AlwaysCheapestBaseline: Always use lowest cost model
"""

from .base import BanditAlgorithm, BanditFeedback, ModelArm
from .baselines import (
    AlwaysBestBaseline,
    AlwaysCheapestBaseline,
    OracleBaseline,
    RandomBaseline,
)
from .contextual_thompson_sampling import ContextualThompsonSamplingBandit
from .dueling import DuelingBandit, DuelingFeedback
from .epsilon_greedy import EpsilonGreedyBandit
from .linucb import LinUCBBandit
from .thompson_sampling import ThompsonSamplingBandit
from .ucb import UCB1Bandit

__all__ = [
    "BanditAlgorithm",
    "BanditFeedback",
    "ModelArm",
    "DuelingBandit",
    "DuelingFeedback",
    "ContextualThompsonSamplingBandit",
    "ThompsonSamplingBandit",
    "LinUCBBandit",
    "UCB1Bandit",
    "EpsilonGreedyBandit",
    "RandomBaseline",
    "OracleBaseline",
    "AlwaysBestBaseline",
    "AlwaysCheapestBaseline",
]
