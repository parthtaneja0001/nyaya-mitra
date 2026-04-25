"""fraction of relevant ground-truth facts the agent uncovered.

range: [0, 1]. relevant facts = union of fact ids referenced by the eligibility
checkers for the profile's eligible schemes plus the applicability checkers for
its applicable frameworks. zero relevant facts returns 1.0.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def compute(ctx: RewardContext) -> float:
    relevant: set[str] = set()
    truth = ctx.profile.derived_ground_truth
    for sid in truth.eligible_schemes:
        relevant |= ctx.kb.relevant_facts_for_scheme(sid)
    for fid in truth.applicable_frameworks:
        relevant |= ctx.kb.relevant_facts_for_framework(fid)
    if not relevant:
        return 1.0
    elicited = ctx.elicited_facts & relevant
    return len(elicited) / len(relevant)
