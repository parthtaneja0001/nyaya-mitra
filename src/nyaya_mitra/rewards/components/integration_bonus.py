"""bonus when a profile has both an eligible scheme and an applicable framework
and the agent identified both at precision ≥ 0.5 and recall ≥ 0.5.

range: {0, 1}. binary. only awarded on integrated profiles. reads
scheme_precision/recall and legal_precision/recall from the running breakdown.
"""

from __future__ import annotations

from nyaya_mitra.interface.reward_keys import (
    LEGAL_PRECISION,
    LEGAL_RECALL,
    SCHEME_PRECISION,
    SCHEME_RECALL,
)
from nyaya_mitra.rewards.context import RewardContext

_THRESHOLD = 0.5


def compute(ctx: RewardContext, breakdown: dict[str, float]) -> float:
    truth = ctx.profile.derived_ground_truth
    if not truth.eligible_schemes or not truth.applicable_frameworks:
        return 0.0
    sp = breakdown.get(SCHEME_PRECISION, 0.0)
    sr = breakdown.get(SCHEME_RECALL, 0.0)
    lp = breakdown.get(LEGAL_PRECISION, 0.0)
    lr = breakdown.get(LEGAL_RECALL, 0.0)
    if sp >= _THRESHOLD and sr >= _THRESHOLD and lp >= _THRESHOLD and lr >= _THRESHOLD:
        return 1.0
    return 0.0
