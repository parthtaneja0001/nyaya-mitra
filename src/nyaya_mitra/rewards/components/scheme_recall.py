"""of qualifying schemes, what fraction the agent identified.

range: [0, 1]. no qualifying schemes returns 1.0 (no false negatives possible).
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext) -> float:
    eligible = set(ctx.profile.derived_ground_truth.eligible_schemes)
    if not eligible:
        return 1.0
    suggested = {s.scheme_id for s in ctx.plan.schemes}
    found = suggested & eligible
    return len(found) / len(eligible)
