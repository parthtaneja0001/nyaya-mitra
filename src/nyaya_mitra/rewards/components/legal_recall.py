"""of applicable legal frameworks, what fraction the agent identified.

range: [0, 1]. no applicable frameworks returns 1.0.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext) -> float:
    applicable = set(ctx.profile.derived_ground_truth.applicable_frameworks)
    if not applicable:
        return 1.0
    suggested = {r.framework_id for r in ctx.plan.legal_routes}
    found = suggested & applicable
    return len(found) / len(applicable)
