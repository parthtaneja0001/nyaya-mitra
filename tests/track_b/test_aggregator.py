"""aggregator end-to-end tests.

verifies:
  - all reward keys present in the breakdown
  - weight sum == 1.0
  - gates dominate components
  - hand-calculated weighted sum matches
  - shaping is consumed from info
  - sim_leak passthrough zeroes elicitation shaping but not jargon penalty
  - make_env_reward_fn surfaces the right callable
"""

from __future__ import annotations

import math

import pytest

from nyaya_mitra.interface import ALL_KEYS, GATE_KEYS, SHAPING_KEYS, TOTAL
from nyaya_mitra.interface.reward_keys import (
    DIGNITY_JUDGE,
    DOCUMENT_ACCURACY,
    GATE_CONTRADICTION,
    GATE_HALLUCINATION,
    GATE_SIM_LEAK,
    HARM_PENALTY,
    LEGAL_PRECISION,
    LEGAL_RECALL,
    SCHEME_PRECISION,
    SCHEME_RECALL,
    SHAPING_ASK_FACT,
    SHAPING_LATE_TURN,
    SHAPING_PROBE_SENSITIVE,
)
from nyaya_mitra.rewards import compute_reward, make_env_reward_fn
from nyaya_mitra.rewards.weights import GATE_FAIL_TOTAL, WEIGHTS, validate_weights

from .conftest import (
    Ask,
    FakeKB,
    make_ctx,
    make_legal_rec,
    make_plan,
    make_profile,
    make_scheme_rec,
    make_turn,
)


def test_weights_sum_to_one_and_caps_respected():
    # primary invariant. validate_weights() raises on failure.
    validate_weights()
    assert math.isclose(sum(WEIGHTS.values()), 1.0)
    assert WEIGHTS[DIGNITY_JUDGE] <= 0.05 + 1e-9


def test_breakdown_emits_every_key(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    for k in ALL_KEYS:
        assert k in out, f"missing key {k}"
    assert isinstance(out[TOTAL], float)


def test_perfect_plan_total_in_expected_range(kb_basic: FakeKB):
    from nyaya_mitra.interface import Probe

    probe = Probe(question="is there violence at home?", sensitive_topic="dv", language="en")
    ctx = make_ctx(
        profile=make_profile(
            eligible_schemes=["pm_kisan"],
            applicable_frameworks=["domestic_violence_act_2005"],
            sensitive_facts={"dv_present": True},
        ),
        plan=make_plan(
            schemes=[
                make_scheme_rec(
                    "pm_kisan",
                    rationale_facts=["occupation_farmer", "land_small"],
                    documents=["Aadhaar", "Bank account", "Land record"],
                )
            ],
            legal_routes=[
                make_legal_rec(
                    "domestic_violence_act_2005",
                    forum="Magistrate of the First Class",
                    procedural_steps=[
                        "approach protection officer",
                        "file dv-1 form",
                        "magistrate grants protection order",
                    ],
                    documents=["Identity proof", "Address proof"],
                    authority="DLSA",
                    contact_id="dlsa_test",
                )
            ],
        ),
        transcript=[make_turn(0, probe)],
        elicited_facts=["occupation_farmer", "land_small", "gender_female", "dv_present"],
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    assert out[GATE_HALLUCINATION] == 0.0
    assert out[GATE_CONTRADICTION] == 0.0
    assert out[TOTAL] == pytest.approx(1.0, abs=0.02)


def test_hallucination_gate_dominates(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(schemes=[make_scheme_rec("ghost_scheme")]),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    assert out[GATE_HALLUCINATION] == 1.0
    assert out[TOTAL] == GATE_FAIL_TOTAL


def test_contradiction_gate_dominates(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(
            schemes=[make_scheme_rec("pm_kisan", rationale_facts=["nonexistent_fact"])],
        ),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    assert out[GATE_CONTRADICTION] == 1.0
    assert out[TOTAL] == GATE_FAIL_TOTAL


def test_format_gate_dominates(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(),
        plan=make_plan(),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    assert out[TOTAL] == GATE_FAIL_TOTAL


def test_total_is_weighted_sum_when_no_gates(kb_basic: FakeKB):
    # construct a ctx where every soft component is exactly 0.5 (or known) and check
    # the math by hand.
    profile = make_profile(eligible_schemes=["pm_kisan", "pmuy"])
    plan = make_plan(schemes=[make_scheme_rec("pm_kisan")])  # 1/1 precision, 1/2 recall
    ctx = make_ctx(
        profile=profile,
        plan=plan,
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    # validate the obvious ones
    assert out[SCHEME_PRECISION] == 1.0
    assert out[SCHEME_RECALL] == 0.5
    assert out[LEGAL_PRECISION] == 1.0
    assert out[LEGAL_RECALL] == 1.0
    # document_accuracy is 0 because pm_kisan has empty docs in the suggestion
    assert out[DOCUMENT_ACCURACY] == 0.0
    expected_soft = sum(WEIGHTS[k] * out[k] for k in WEIGHTS)
    expected = expected_soft + out[HARM_PENALTY] + sum(out[k] for k in SHAPING_KEYS)
    assert out[TOTAL] == pytest.approx(expected)


def test_harm_penalty_subtracts_from_total(kb_basic: FakeKB):
    # pm_kisan suggested but profile has eligible=[pmuy]. wrong but valid id -> harm.
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pmuy"]),
        plan=make_plan(
            schemes=[
                make_scheme_rec("pm_kisan", documents=["Aadhaar", "Bank account", "Land record"])
            ],
        ),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    assert out[HARM_PENALTY] == pytest.approx(-0.05)
    assert out[TOTAL] < sum(WEIGHTS[k] * out[k] for k in WEIGHTS)


def test_shaping_is_consumed_from_info(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
        kb=kb_basic,
        info={
            "max_turns": 20,
            "shaping_running": {
                SHAPING_ASK_FACT: 0.10,
                SHAPING_PROBE_SENSITIVE: 0.05,
                SHAPING_LATE_TURN: -0.06,
                "shaping_jargon": 0.0,
            },
        },
    )
    out = compute_reward(ctx)
    assert out[SHAPING_ASK_FACT] == pytest.approx(0.10)
    assert out[SHAPING_PROBE_SENSITIVE] == pytest.approx(0.05)
    assert out[SHAPING_LATE_TURN] == pytest.approx(-0.06)


def test_shaping_positive_cap_applied(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
        kb=kb_basic,
        info={
            "max_turns": 20,
            "shaping_running": {
                SHAPING_ASK_FACT: 0.5,
                SHAPING_PROBE_SENSITIVE: 0.3,
                SHAPING_LATE_TURN: -0.1,
                "shaping_jargon": 0.0,
            },
        },
    )
    out = compute_reward(ctx)
    pos_sum = sum(out[k] for k in SHAPING_KEYS if out[k] > 0)
    assert pos_sum == pytest.approx(0.4)


def test_sim_leak_zeroes_elicitation_shaping(kb_basic: FakeKB):
    ask = Ask(question="...", language="en")
    leaked_turn = make_turn(0, ask, info={"sim_leak": True})
    ctx = make_ctx(
        profile=make_profile(eligible_schemes=["pm_kisan"]),
        plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
        kb=kb_basic,
        transcript=[leaked_turn],
        info={
            "max_turns": 20,
            "shaping_running": {
                SHAPING_ASK_FACT: 0.10,
                SHAPING_PROBE_SENSITIVE: 0.20,
                SHAPING_LATE_TURN: -0.05,
                "shaping_jargon": -0.10,
            },
        },
    )
    out = compute_reward(ctx)
    assert out[SHAPING_ASK_FACT] == 0.0
    assert out[SHAPING_PROBE_SENSITIVE] == 0.0
    # negative shaping is preserved
    assert out[SHAPING_LATE_TURN] == pytest.approx(-0.05)
    assert out[GATE_SIM_LEAK] == 1.0


def test_make_env_reward_fn_signature(kb_basic: FakeKB):
    fn = make_env_reward_fn(kb_basic)
    profile = make_profile(eligible_schemes=["pm_kisan"])
    plan = make_plan(schemes=[make_scheme_rec("pm_kisan")])
    out = fn(profile, plan, [], set())
    for k in ALL_KEYS:
        assert k in out


def test_translator_surfaces_turnrecord_negated_through_contradiction_gate(kb_basic: FakeKB):
    """track A's env writes per-turn TurnRecord.negated. the translator must copy
    that into Turn.info['negated_facts'] so the contradiction gate fires when a
    plan's rationale_facts include something the citizen explicitly negated."""
    from dataclasses import dataclass, field
    from typing import Any

    from nyaya_mitra.interface.reward_keys import GATE_CONTRADICTION

    @dataclass
    class FakeTurnRecord:
        actor: str
        payload: dict[str, Any]
        revealed: list[str] = field(default_factory=list)
        negated: list[str] = field(default_factory=list)

    transcript = [
        FakeTurnRecord(
            actor="citizen",
            payload={"utterance": "i am not a farmer"},
            revealed=[],
            negated=["occupation_farmer"],
        )
    ]
    profile = make_profile(eligible_schemes=["pm_kisan"])
    plan = make_plan(
        schemes=[make_scheme_rec("pm_kisan", rationale_facts=["occupation_farmer"])],
    )
    fn = make_env_reward_fn(kb_basic)
    out = fn(profile, plan, transcript, {"occupation_farmer"})
    assert out[GATE_CONTRADICTION] == 1.0
    assert out[TOTAL] == GATE_FAIL_TOTAL


def test_translator_handles_missing_negated_attr_gracefully(kb_basic: FakeKB):
    """legacy TurnRecord shapes without a 'negated' attribute must still translate."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class LegacyTurnRecord:
        actor: str
        payload: dict[str, Any]
        revealed: list[str] = field(default_factory=list)

    transcript = [
        LegacyTurnRecord(
            actor="citizen",
            payload={"utterance": "ok"},
            revealed=["occupation_farmer"],
        )
    ]
    profile = make_profile(eligible_schemes=["pm_kisan"])
    plan = make_plan(
        schemes=[make_scheme_rec("pm_kisan", rationale_facts=["occupation_farmer"])],
    )
    fn = make_env_reward_fn(kb_basic)
    out = fn(profile, plan, transcript, {"occupation_farmer"})
    # no negation surfaced -> contradiction gate stays off
    from nyaya_mitra.interface.reward_keys import GATE_CONTRADICTION

    assert out[GATE_CONTRADICTION] == 0.0


def test_no_dignity_dominance_with_max_judge(kb_basic: FakeKB):
    """if a malicious LLM-judge stamps every plan as 1.0, the agent still cannot
    score higher than 0.05 from dignity alone — proves no single LLM component
    dominates."""
    from nyaya_mitra.rewards.components.dignity_judge import set_judge

    set_judge(lambda _ctx: 1.0)
    try:
        # everything else minimum: empty plan would gate on format, so use a
        # plan that exists but is wrong on every soft component.
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(schemes=[make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        out = compute_reward(ctx)
    finally:
        set_judge(None)
    # dignity contributes at most 0.05 to the total; subtract harm = 0.05
    contribution = WEIGHTS[DIGNITY_JUDGE] * out[DIGNITY_JUDGE]
    assert contribution == pytest.approx(0.05)


def test_gate_keys_are_subset_of_breakdown(kb_basic: FakeKB):
    ctx = make_ctx(
        profile=make_profile(),
        plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
        kb=kb_basic,
    )
    out = compute_reward(ctx)
    for k in GATE_KEYS:
        assert k in out
