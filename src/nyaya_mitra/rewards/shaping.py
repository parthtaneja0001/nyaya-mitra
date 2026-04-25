"""per-turn shaping rewards.

shaping is computed as the env runs (each step) and accumulated into a running
breakdown. at terminal step, the aggregator caps positive shaping at +0.4
across the whole episode (negatives have no cap) and adds it to the total.

PLAN B.2 #4 spec:
  +0.02 when an ASK causes the extractor to add a fact id to elicited_facts
  +0.05 when a PROBE correctly elicits a sensitive fact relevant to the
        ground truth. "correct" means: matching sensitive_topic AND the
        revealed fact projects to the same topic AND not flagged sim_leak.
  -0.03 per turn after turn 15
  -0.10 for an EXPLAIN flagged jargon-heavy by the literacy checker

emits 4 keys (SHAPING_*) into the per-turn delta dict; aggregator sums them.
"""

from __future__ import annotations

import re
from typing import Any

from nyaya_mitra.interface import Ask, Explain, Probe
from nyaya_mitra.interface.reward_keys import (
    SHAPING_ASK_FACT,
    SHAPING_JARGON,
    SHAPING_LATE_TURN,
    SHAPING_PROBE_SENSITIVE,
)
from nyaya_mitra.rewards.components.sensitivity_correctness import (
    _FACT_PREFIX_TOPIC,
    _SENSITIVE_FACT_TOPIC,
)

ASK_FACT_BONUS = 0.02
PROBE_SENSITIVE_BONUS = 0.05
LATE_TURN_PENALTY = -0.03
JARGON_PENALTY = -0.10
LATE_TURN_THRESHOLD = 15
POSITIVE_SHAPING_CAP = 0.4

_JARGON = re.compile(
    r"\b(crpc|cpc|ipc|article|section|magistrate|adjudication|ex[- ]?parte|pendente lite|"
    r"affidavit|writ|rejoinder|interlocutory|injunction|maintainability)\b",
    re.IGNORECASE,
)


def _fact_topic(fact_id: str) -> str | None:
    if fact_id in _SENSITIVE_FACT_TOPIC:
        return _SENSITIVE_FACT_TOPIC[fact_id]
    for prefix, topic in _FACT_PREFIX_TOPIC:
        if fact_id.startswith(prefix):
            return topic
    return None


def compute_shaping(
    turn_index: int,
    action: Any,
    revealed_this_turn: list[str],
    sim_leak: bool,
    citizen_literacy: str,
) -> dict[str, float]:
    """compute the shaping deltas for one advisor turn.

    called by the bootstrap glue right after the env applies an advisor action
    and the citizen responds. arguments are framework-neutral so this never
    needs to know about TurnRecord shapes.
    """
    out: dict[str, float] = {
        SHAPING_ASK_FACT: 0.0,
        SHAPING_PROBE_SENSITIVE: 0.0,
        SHAPING_LATE_TURN: 0.0,
        SHAPING_JARGON: 0.0,
    }

    if turn_index >= LATE_TURN_THRESHOLD:
        out[SHAPING_LATE_TURN] = LATE_TURN_PENALTY

    if isinstance(action, Ask) and revealed_this_turn:
        out[SHAPING_ASK_FACT] = ASK_FACT_BONUS

    if isinstance(action, Probe) and revealed_this_turn and not sim_leak:
        for fact_id in revealed_this_turn:
            topic = _fact_topic(fact_id)
            if topic and topic == action.sensitive_topic:
                out[SHAPING_PROBE_SENSITIVE] = PROBE_SENSITIVE_BONUS
                break

    if isinstance(action, Explain):
        if citizen_literacy == "low" and _JARGON.search(action.content):
            out[SHAPING_JARGON] = JARGON_PENALTY
        elif action.target_literacy != citizen_literacy:
            pass

    return out


def cap_positive_shaping(running: dict[str, float]) -> dict[str, float]:
    """cap the SUM of positive shaping at +0.4. negatives are uncapped.

    proportionally scales positive entries when the cap is exceeded so the
    breakdown still sums correctly and individual key magnitudes stay
    interpretable.
    """
    out = dict(running)
    positives = {k: v for k, v in out.items() if v > 0 and k.startswith("shaping_")}
    pos_sum = sum(positives.values())
    if pos_sum <= POSITIVE_SHAPING_CAP:
        return out
    scale = POSITIVE_SHAPING_CAP / pos_sum
    for k in positives:
        out[k] = positives[k] * scale
    return out
