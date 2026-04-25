"""turn efficiency: fewer turns is better, but only when fact coverage is ≥ 0.5.

range: [0, 1]. zero turns or zero max_turns returns 0.0. coverage_factor reads
fact_coverage from the running breakdown — components run in deterministic order
and fact_coverage is computed before this one.

formula: max(0, 1 - turns/max_turns) * coverage_factor
where coverage_factor = 1.0 if breakdown[fact_coverage] >= 0.5 else 0.0
"""

from __future__ import annotations

from nyaya_mitra.interface.reward_keys import FACT_COVERAGE
from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext, breakdown: dict[str, float]) -> float:
    if ctx.max_turns <= 0:
        return 0.0
    coverage = breakdown.get(FACT_COVERAGE, 0.0)
    if coverage < 0.5:
        return 0.0
    used = ctx.turns_used
    raw = max(0.0, 1.0 - (used / ctx.max_turns))
    return raw
