"""harm penalty: -0.05 per harmful suggestion, capped at -0.20.

a "harmful" suggestion is one that wastes the citizen's time/money — currently
defined as any scheme/framework suggested that is NOT in the profile's
derived_ground_truth (i.e., the citizen doesn't actually qualify / it doesn't
apply). we don't double-count hallucinated ids — those gate to -1 separately
and harm_penalty is bypassed in that case.

range: [-0.20, 0]. always non-positive. emitted under HARM_PENALTY key.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext

_PER_HARM = 0.05
_MAX_HARMS = 4


def compute(ctx: RewardContext) -> float:
    eligible = set(ctx.profile.derived_ground_truth.eligible_schemes)
    applicable = set(ctx.profile.derived_ground_truth.applicable_frameworks)

    harms = 0
    for s in ctx.plan.schemes:
        if not ctx.kb.has_scheme(s.scheme_id):
            continue
        if s.scheme_id not in eligible:
            harms += 1
    for r in ctx.plan.legal_routes:
        if not ctx.kb.has_framework(r.framework_id):
            continue
        if r.framework_id not in applicable:
            harms += 1

    harms = min(harms, _MAX_HARMS)
    return -_PER_HARM * harms
