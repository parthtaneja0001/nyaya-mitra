"""of suggested schemes, what fraction the citizen actually qualifies for.

range: [0, 1]. empty plan returns 1.0 (no false positives).
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext) -> float:
    suggested = {s.scheme_id for s in ctx.plan.schemes}
    if not suggested:
        return 1.0
    eligible = set(ctx.profile.derived_ground_truth.eligible_schemes)
    correct = suggested & eligible
    return len(correct) / len(suggested)
