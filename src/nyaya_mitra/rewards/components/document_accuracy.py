"""for each suggested item, jaccard between listed documents and the kb-canonical list.

range: [0, 1]. averaged over all suggested schemes + legal routes whose ids exist in kb.
empty plan or all-unknown ids returns 1.0 (vacuously).

document strings are normalized (lowercase, stripped, punctuation collapsed) before
comparison so minor spelling variants don't tank the score.
"""

from __future__ import annotations

import re

from nyaya_mitra.rewards.context import RewardContext

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _normalize(doc: str) -> str:
    s = doc.lower().strip()
    s = _NON_ALNUM.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _norm_set(docs: list[str]) -> set[str]:
    return {_normalize(d) for d in docs if d.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute(ctx: RewardContext) -> float:
    scores: list[float] = []

    for s in ctx.plan.schemes:
        if not ctx.kb.has_scheme(s.scheme_id):
            continue
        suggested = _norm_set(s.required_documents)
        canonical = _norm_set(ctx.kb.documents_for_scheme(s.scheme_id))
        scores.append(_jaccard(suggested, canonical))

    for r in ctx.plan.legal_routes:
        if not ctx.kb.has_framework(r.framework_id):
            continue
        suggested = _norm_set(r.required_documents)
        canonical = _norm_set(ctx.kb.documents_for_framework(r.framework_id))
        scores.append(_jaccard(suggested, canonical))

    if not scores:
        return 1.0
    return sum(scores) / len(scores)
