"""Bandit algorithms for LLM routing.

This module contains multi-armed bandit implementations for intelligent
model selection. All algorithms share a common interface and work with
Conduit's QueryFeatures.

Available Algorithms:
- ThompsonSamplingBandit: Bayesian probability matching (optimal regret)
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
from .epsilon_greedy import EpsilonGreedyBandit
from .thompson_sampling import ThompsonSamplingBandit
from .ucb import UCB1Bandit

__all__ = [
    "BanditAlgorithm",
    "BanditFeedback",
    "ModelArm",
    "ThompsonSamplingBandit",
    "UCB1Bandit",
    "EpsilonGreedyBandit",
    "RandomBaseline",
    "OracleBaseline",
    "AlwaysBestBaseline",
    "AlwaysCheapestBaseline",
]
