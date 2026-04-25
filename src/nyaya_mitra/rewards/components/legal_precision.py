"""of suggested legal routes, what fraction actually applies.

range: [0, 1]. empty plan returns 1.0.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext) -> float:
    suggested = {r.framework_id for r in ctx.plan.legal_routes}
    if not suggested:
        return 1.0
    applicable = set(ctx.profile.derived_ground_truth.applicable_frameworks)
    correct = suggested & applicable
    return len(correct) / len(suggested)
