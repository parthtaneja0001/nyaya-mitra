"""of profile sensitive facts, what fraction were elicited via a matching probe.

range: [0, 1]. sensitive facts come from profile.situation_specific.sensitive_facts;
the topic mapping below maps fact suffixes to AdvisorAction.SensitiveTopic values.
no sensitive facts returns 1.0.

a sensitive fact is "correctly elicited" if at some turn it appeared in
elicited_facts AND a Probe with the matching sensitive_topic preceded that
appearance OR happened on the same turn (the env appends advisor turns first,
so a same-turn probe is valid). turns flagged as sim_leak don't count.
"""

from __future__ import annotations

from nyaya_mitra.interface import Probe
from nyaya_mitra.rewards.context import RewardContext

_SENSITIVE_FACT_TOPIC: dict[str, str] = {
    "caste": "caste",
    "dv": "dv",
    "dv_history": "dv",
    "dv_present": "dv",
    "disability": "disability",
    "immigration": "immigration",
    "hiv": "hiv_status",
    "hiv_status": "hiv_status",
    "orientation": "sexual_orientation",
    "sexual_orientation": "sexual_orientation",
    "mental_health": "mental_health",
    "mental": "mental_health",
}

_FACT_PREFIX_TOPIC: list[tuple[str, str]] = [
    ("caste_", "caste"),
    ("dv_", "dv"),
    ("disability_", "disability"),
    ("immigration_", "immigration"),
    ("hiv_", "hiv_status"),
    ("orientation_", "sexual_orientation"),
    ("mental_", "mental_health"),
]


def _expected_topics(profile_sensitive: dict[str, object]) -> dict[str, str]:
    """maps fact-id-shaped key → sensitive_topic. profile sensitive_facts use
    keys like 'dv_present' or 'caste_sc'; we project to the matching topic."""
    out: dict[str, str] = {}
    for k, v in profile_sensitive.items():
        if not v:
            continue
        if k in _SENSITIVE_FACT_TOPIC:
            out[k] = _SENSITIVE_FACT_TOPIC[k]
            continue
        for prefix, topic in _FACT_PREFIX_TOPIC:
            if k.startswith(prefix):
                out[k] = topic
                break
    return out


def compute(ctx: RewardContext) -> float:
    sensitive = ctx.profile.situation_specific.sensitive_facts
    expected = _expected_topics(sensitive)
    if not expected:
        return 1.0

    correctly_probed: set[str] = set()
    for fact_id, topic in expected.items():
        for turn in ctx.transcript:
            if not isinstance(turn.action, Probe):
                continue
            if turn.action.sensitive_topic != topic:
                continue
            if fact_id in ctx.elicited_facts:
                correctly_probed.add(fact_id)
                break

    return len(correctly_probed) / len(expected)
