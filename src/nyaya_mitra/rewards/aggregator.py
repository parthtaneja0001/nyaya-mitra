"""aggregator. composes components, gates, shaping into a final breakdown.

public api:
    compute_reward(ctx)               -> RewardBreakdown
    make_env_reward_fn(kb, *, ...)    -> Callable matching track A's RewardFn shape

execution order at terminal step:
    1. compute deterministic components in dependency order
       (precision/recall/document/procedural/fact_coverage feed integration_bonus
        and turn_efficiency reads fact_coverage; sensitivity/dignity are independent)
    2. compute harm_penalty (separate additive, not in soft budget)
    3. evaluate gates (format → hallucination → contradiction)
    4. read shaping_running from ctx.info if track A passed it through
       (the env can carry per-turn shaping deltas accumulated during the episode)
    5. apply positive-shaping cap
    6. zero out elicitation-shaping credit on leaked turns (sim_leak passthrough)
    7. final total: weighted sum of soft components + harm + capped shaping,
       OR -1.0 if any hard gate fired.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nyaya_mitra.interface import ActionPlan, CitizenObservation, CitizenProfile
from nyaya_mitra.interface.reward_keys import (
    ALL_KEYS,
    COMPONENT_KEYS,
    DIGNITY_JUDGE,
    DOCUMENT_ACCURACY,
    FACT_COVERAGE,
    GATE_CONTRADICTION,
    GATE_FORMAT_VIOLATION,
    GATE_HALLUCINATION,
    GATE_KEYS,
    GATE_SIM_LEAK,
    HARM_PENALTY,
    INTEGRATION_BONUS,
    LEGAL_PRECISION,
    LEGAL_RECALL,
    PROCEDURAL_CORRECTNESS,
    SCHEME_PRECISION,
    SCHEME_RECALL,
    SENSITIVITY_CORRECTNESS,
    SHAPING_ASK_FACT,
    SHAPING_KEYS,
    SHAPING_PROBE_SENSITIVE,
    TOTAL,
    TURN_EFFICIENCY,
)
from nyaya_mitra.rewards.components import (
    compute_dignity_judge,
    compute_document_accuracy,
    compute_fact_coverage,
    compute_harm_penalty,
    compute_integration_bonus,
    compute_legal_precision,
    compute_legal_recall,
    compute_procedural_correctness,
    compute_scheme_precision,
    compute_scheme_recall,
    compute_sensitivity_correctness,
    compute_turn_efficiency,
)
from nyaya_mitra.rewards.context import RewardContext, Turn
from nyaya_mitra.rewards.gates import (
    check_contradiction,
    check_format,
    check_hallucination,
    leaked_turn_indices,
)
from nyaya_mitra.rewards.kb_protocol import KnowledgeBase
from nyaya_mitra.rewards.shaping import cap_positive_shaping
from nyaya_mitra.rewards.types import RewardBreakdown
from nyaya_mitra.rewards.weights import GATE_FAIL_TOTAL, WEIGHTS


def _empty_breakdown() -> RewardBreakdown:
    return {k: 0.0 for k in ALL_KEYS}


def _apply_sim_leak_passthrough(
    shaping: dict[str, float], leaked_turn_count: int, leaked_any: bool
) -> dict[str, float]:
    """zero out elicitation-shaping credit when leaks occurred. policy:
    if any leaks happened, scale ask_fact and probe_sensitive shaping
    proportional to (1 - leaked_fraction_of_advisor_turns) — but we don't have
    advisor-turn count here, so we keep it simple: if leaked_any is true we
    zero those two keys. negative shaping (late_turn, jargon) is preserved.
    """
    if not leaked_any:
        return shaping
    out = dict(shaping)
    out[SHAPING_ASK_FACT] = 0.0
    out[SHAPING_PROBE_SENSITIVE] = 0.0
    return out


def compute_reward(ctx: RewardContext) -> RewardBreakdown:
    """canonical entrypoint. returns a breakdown with all ALL_KEYS present."""
    breakdown = _empty_breakdown()

    breakdown[SCHEME_PRECISION] = compute_scheme_precision(ctx)
    breakdown[SCHEME_RECALL] = compute_scheme_recall(ctx)
    breakdown[LEGAL_PRECISION] = compute_legal_precision(ctx)
    breakdown[LEGAL_RECALL] = compute_legal_recall(ctx)
    breakdown[DOCUMENT_ACCURACY] = compute_document_accuracy(ctx)
    breakdown[PROCEDURAL_CORRECTNESS] = compute_procedural_correctness(ctx)
    breakdown[FACT_COVERAGE] = compute_fact_coverage(ctx)
    breakdown[SENSITIVITY_CORRECTNESS] = compute_sensitivity_correctness(ctx)
    breakdown[INTEGRATION_BONUS] = compute_integration_bonus(ctx, breakdown)
    breakdown[TURN_EFFICIENCY] = compute_turn_efficiency(ctx, breakdown)
    breakdown[DIGNITY_JUDGE] = compute_dignity_judge(ctx)
    breakdown[HARM_PENALTY] = compute_harm_penalty(ctx)

    format_failed = check_format(ctx)
    halluc_failed = check_hallucination(ctx)
    contradict_failed = check_contradiction(ctx)
    breakdown[GATE_FORMAT_VIOLATION] = 1.0 if format_failed else 0.0
    breakdown[GATE_HALLUCINATION] = 1.0 if halluc_failed else 0.0
    breakdown[GATE_CONTRADICTION] = 1.0 if contradict_failed else 0.0

    leaked = leaked_turn_indices(ctx)
    breakdown[GATE_SIM_LEAK] = float(len(leaked))

    shaping_running: dict[str, float] = {k: 0.0 for k in SHAPING_KEYS}
    raw_shaping = ctx.info.get("shaping_running") if isinstance(ctx.info, dict) else None
    if isinstance(raw_shaping, dict):
        for k in SHAPING_KEYS:
            v = raw_shaping.get(k, 0.0)
            try:
                shaping_running[k] = float(v)
            except (TypeError, ValueError):
                shaping_running[k] = 0.0
    shaping_running = _apply_sim_leak_passthrough(
        shaping_running, leaked_turn_count=len(leaked), leaked_any=bool(leaked)
    )
    shaping_capped = cap_positive_shaping(shaping_running)
    for k in SHAPING_KEYS:
        breakdown[k] = shaping_capped.get(k, 0.0)

    if format_failed or halluc_failed or contradict_failed:
        breakdown[TOTAL] = GATE_FAIL_TOTAL
        return breakdown

    soft = sum(WEIGHTS[k] * breakdown[k] for k in WEIGHTS)
    additive = breakdown[HARM_PENALTY] + sum(breakdown[k] for k in SHAPING_KEYS)
    breakdown[TOTAL] = soft + additive
    return breakdown


# track A's env passes (profile, plan, transcript, elicited_facts) into reward_fn.
# transcript is a list of TurnRecord-like objects with .actor/.payload/.revealed —
# we duck-type it and translate into the Turn dataclass we own.

_TurnLike = Any  # track A's TurnRecord; we only read attributes


def _translate_transcript(transcript: list[_TurnLike]) -> list[Turn]:
    out: list[Turn] = []
    for i, rec in enumerate(transcript):
        actor = getattr(rec, "actor", None)
        payload = getattr(rec, "payload", {}) or {}
        revealed = list(getattr(rec, "revealed", []) or [])
        negated = list(getattr(rec, "negated", []) or [])
        if actor == "advisor":
            action = _action_from_payload(payload)
            out.append(Turn(index=i, action=action, citizen_observation=None, info=dict(payload)))
        elif actor == "citizen":
            obs_info = dict(payload)
            obs_info["revealed"] = revealed
            obs_info["negated_facts"] = negated
            out.append(
                Turn(
                    index=i,
                    action=None,
                    citizen_observation=_observation_from_payload(payload),
                    info=obs_info,
                )
            )
    return out


def _action_from_payload(payload: dict[str, Any]):
    from nyaya_mitra.interface import Ask, Explain, Finalize, Probe

    t = payload.get("type")
    try:
        if t == "ASK":
            return Ask.model_validate(payload)
        if t == "PROBE":
            return Probe.model_validate(payload)
        if t == "EXPLAIN":
            return Explain.model_validate(payload)
        if t == "FINALIZE":
            return Finalize.model_validate(payload)
    except Exception:
        return None
    return None


def _observation_from_payload(payload: dict[str, Any]) -> CitizenObservation | None:
    if "utterance" not in payload:
        return None
    return None  # full obs not needed downstream; components only inspect actions


def make_env_reward_fn(
    kb: KnowledgeBase,
    *,
    extra_info: Callable[[Any], dict[str, Any]] | None = None,
    max_turns: int = 20,
) -> Callable[[CitizenProfile, ActionPlan, list[Any], set[str]], RewardBreakdown]:
    """build a callable matching track A's env reward signature.

    extra_info: optional hook that lets a bootstrap pass per-episode info into
    the context (e.g. shaping_running accumulated by the env each step). called
    with the env's transcript and expected to return a dict.
    """

    def _fn(
        profile: CitizenProfile,
        plan: ActionPlan,
        transcript: list[Any],
        elicited_facts: set[str],
    ) -> RewardBreakdown:
        translated = _translate_transcript(transcript)
        info: dict[str, Any] = {"max_turns": max_turns}
        if extra_info is not None:
            try:
                info.update(extra_info(transcript) or {})
            except Exception:
                pass
        ctx = RewardContext(
            profile=profile,
            plan=plan,
            transcript=translated,
            elicited_facts=set(elicited_facts),
            kb=kb,
            info=info,
        )
        return compute_reward(ctx)

    return _fn


# re-export for the integration test stub.
__all__ = [
    "ALL_KEYS",
    "COMPONENT_KEYS",
    "GATE_KEYS",
    "SHAPING_KEYS",
    "compute_reward",
    "make_env_reward_fn",
]
