"""contradiction gate. fires when the plan rests on facts the citizen never
revealed (or which the extractor explicitly negated).

current implementation requires every fact in scheme rationale_facts to be
present in elicited_facts. when track A surfaces negated_facts via per-turn
info, we additionally fail when a rationale_fact appears in the union of
negated_facts across all citizen turns.

returns True when contradiction is detected.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def _negated_facts(ctx: RewardContext) -> set[str]:
    out: set[str] = set()
    for turn in ctx.transcript:
        info = turn.info or {}
        for f in info.get("negated_facts", []) or []:
            if isinstance(f, str):
                out.add(f)
    return out


def check(ctx: RewardContext) -> bool:
    elicited = ctx.elicited_facts
    negated = _negated_facts(ctx)

    for s in ctx.plan.schemes:
        for fact in s.rationale_facts:
            if fact in negated:
                return True
            if fact and fact not in elicited:
                return True
    return False
