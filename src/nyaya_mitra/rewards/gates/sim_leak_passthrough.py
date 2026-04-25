"""sim-leak passthrough. NOT a hard gate.

reads info["sim_leak"] from each turn (set by track A's env when a sensitive
fact was revealed without a matching probe). returns the set of turn indices
flagged as leaks so the shaping module can zero out elicitation credit on
those turns. never short-circuits the total.
"""

from __future__ import annotations

from nyaya_mitra.rewards.context import RewardContext


def leaked_turn_indices(ctx: RewardContext) -> set[int]:
    out: set[int] = set()
    for turn in ctx.transcript:
        info = turn.info or {}
        if info.get("sim_leak"):
            out.add(turn.index)
    return out


def total_leak_count(ctx: RewardContext) -> int:
    return len(leaked_turn_indices(ctx))
