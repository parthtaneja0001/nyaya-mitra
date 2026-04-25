"""legal route procedural correctness.

three sub-checks per route, averaged:
  1. forum matches kb (substring match, case-insensitive)
  2. legal_aid_authority matches kb authority for that framework
  3. ordered procedural-step similarity vs kb canonical (bag-of-keywords + order bonus)

range: [0, 1]. no legal routes returns 1.0.
"""

from __future__ import annotations

import re

from nyaya_mitra.rewards.context import RewardContext

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def _step_similarity(suggested: list[str], canonical: list[str]) -> float:
    if not canonical:
        return 1.0 if not suggested else 0.0
    if not suggested:
        return 0.0

    bag_score = _bag_overlap(suggested, canonical)
    order_score = _order_alignment(suggested, canonical)
    return 0.5 * bag_score + 0.5 * order_score


def _bag_overlap(suggested: list[str], canonical: list[str]) -> float:
    sug_bag: set[str] = set()
    for s in suggested:
        sug_bag.update(_tokens(s))
    can_bag: set[str] = set()
    for c in canonical:
        can_bag.update(_tokens(c))
    if not can_bag:
        return 1.0
    return len(sug_bag & can_bag) / len(can_bag)


def _order_alignment(suggested: list[str], canonical: list[str]) -> float:
    """fraction of canonical steps appearing in same relative order in suggested.

    a canonical step "matches" a suggested step when they share at least
    half of the canonical step's content tokens. matches must occur in
    non-decreasing order through the suggested list.
    """
    cursor = 0
    matched = 0
    for can in canonical:
        can_tokens = set(_tokens(can))
        if not can_tokens:
            matched += 1
            continue
        threshold = max(1, (len(can_tokens) + 1) // 2)
        for i in range(cursor, len(suggested)):
            sug_tokens = set(_tokens(suggested[i]))
            overlap = len(can_tokens & sug_tokens)
            if overlap >= threshold:
                matched += 1
                cursor = i + 1
                break
    return matched / len(canonical) if canonical else 1.0


def compute(ctx: RewardContext) -> float:
    if not ctx.plan.legal_routes:
        return 1.0
    scores: list[float] = []
    for r in ctx.plan.legal_routes:
        if not ctx.kb.has_framework(r.framework_id):
            scores.append(0.0)
            continue
        sub: list[float] = []

        kb_forum = ctx.kb.forum_for_framework(r.framework_id) or ""
        if kb_forum:
            forum_match = kb_forum.lower() in r.forum.lower() or r.forum.lower() in kb_forum.lower()
            sub.append(1.0 if forum_match else 0.0)

        kb_authority = ctx.kb.legal_aid_authority_for_framework(r.framework_id)
        if kb_authority is not None:
            sub.append(1.0 if r.free_legal_aid_contact.authority == kb_authority else 0.0)

        canonical_steps = ctx.kb.procedural_steps_for_framework(r.framework_id)
        sub.append(_step_similarity(r.procedural_steps, canonical_steps))

        scores.append(sum(sub) / len(sub) if sub else 0.0)
    return sum(scores) / len(scores)
