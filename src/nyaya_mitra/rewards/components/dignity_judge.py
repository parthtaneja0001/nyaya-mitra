"""dignity judge: tone, blame avoidance, literacy match. capped at 5% by aggregator.

LLM-judged when a judge callable is wired in. when no judge is configured (the
default in tests, ci, and most training runs to keep it deterministic), this
returns a deterministic heuristic score so the component always has a defined
value and the breakdown stays stable.

range: [0, 1]. always within [0, 1] regardless of judge output (clamped).

the heuristic checks:
  - any EXPLAIN to a low-literacy citizen avoids legal jargon
  - no advisor turn contains blame keywords toward the citizen
  - at least one EXPLAIN exists when the citizen literacy is low

a judge callable can be installed by setting set_judge(fn) where
fn: Callable[[RewardContext], float] returns a 0..1 score.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from nyaya_mitra.interface import Explain
from nyaya_mitra.rewards.context import RewardContext

_JARGON = re.compile(
    r"\b(crpc|cpc|ipc|article|section|magistrate|adjudication|ex[- ]?parte|pendente lite|"
    r"affidavit|writ|rejoinder|interlocutory|injunction|maintainability)\b",
    re.IGNORECASE,
)
_BLAME = re.compile(
    r"\b(your fault|you should have|why didn't you|you were wrong|you are responsible)\b",
    re.IGNORECASE,
)


_judge: Callable[[RewardContext], float] | None = None


def set_judge(fn: Callable[[RewardContext], float] | None) -> None:
    """install or clear the LLM judge. None reverts to the deterministic heuristic."""
    global _judge
    _judge = fn


def _heuristic(ctx: RewardContext) -> float:
    score = 1.0
    literacy = ctx.profile.behavior.literacy

    explains = [t.action for t in ctx.transcript if isinstance(t.action, Explain)]
    advisor_text_blob = " ".join(
        _action_text(t.action) for t in ctx.transcript if t.action is not None
    )

    if literacy == "low":
        if not explains:
            score -= 0.25
        for ex in explains:
            if ex.target_literacy != "low":
                score -= 0.15
                break
            if _JARGON.search(ex.content):
                score -= 0.20
                break

    if _BLAME.search(advisor_text_blob):
        score -= 0.30

    return max(0.0, min(1.0, score))


def _action_text(action) -> str:
    for attr in ("question", "content"):
        v = getattr(action, attr, None)
        if isinstance(v, str):
            return v
    return ""


def compute(ctx: RewardContext) -> float:
    if _judge is not None:
        try:
            raw = float(_judge(ctx))
        except Exception:
            raw = _heuristic(ctx)
        return max(0.0, min(1.0, raw))
    return _heuristic(ctx)
