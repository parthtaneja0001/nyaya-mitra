"""structural format gate. pydantic enforces wire-level validity, but we still
guard against:
  - empty plan with no schemes AND no legal_routes
  - blank most_important_next_step
  - blank plain_summary text
  - explicit format_violation flag in ctx.info (set by env on schema-rejected actions)

returns True when the format is invalid and the gate should fire.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def check(ctx: RewardContext) -> bool:
    if ctx.format_violation:
        return True
    plan = ctx.plan
    if not plan.schemes and not plan.legal_routes:
        return True
    if not (plan.most_important_next_step and plan.most_important_next_step.strip()):
        return True
    if not (plan.plain_summary and plan.plain_summary.text and plan.plain_summary.text.strip()):
        return True
    return False
