"""reward module — track B owns weights, gates, and aggregation.

public api:
    compute_reward(ctx)   -> RewardBreakdown   # called at terminal step
    compute_shaping(...)  -> dict[str, float]  # called each turn

track A imports from here ONLY through these two functions. all other
internals are private to track B.
"""

from __future__ import annotations

from nyaya_mitra.rewards.aggregator import compute_reward, make_env_reward_fn
from nyaya_mitra.rewards.context import RewardContext, Turn
from nyaya_mitra.rewards.kb_protocol import KnowledgeBase
from nyaya_mitra.rewards.shaping import compute_shaping
from nyaya_mitra.rewards.types import RewardBreakdown

__all__ = [
    "KnowledgeBase",
    "RewardBreakdown",
    "RewardContext",
    "Turn",
    "compute_reward",
    "compute_shaping",
    "make_env_reward_fn",
]
