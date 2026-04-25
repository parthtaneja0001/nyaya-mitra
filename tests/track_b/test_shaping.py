"""per-turn shaping rewards + positive-shaping cap."""

from __future__ import annotations

from nyaya_mitra.interface import Ask, Explain, Probe
from nyaya_mitra.interface.reward_keys import (
    SHAPING_ASK_FACT,
    SHAPING_JARGON,
    SHAPING_LATE_TURN,
    SHAPING_PROBE_SENSITIVE,
)
from nyaya_mitra.rewards.shaping import (
    ASK_FACT_BONUS,
    JARGON_PENALTY,
    LATE_TURN_PENALTY,
    POSITIVE_SHAPING_CAP,
    PROBE_SENSITIVE_BONUS,
    cap_positive_shaping,
    compute_shaping,
)


def test_ask_with_new_fact_yields_bonus():
    ask = Ask(question="...", language="en")
    out = compute_shaping(0, ask, ["occupation_farmer"], False, "medium")
    assert out[SHAPING_ASK_FACT] == ASK_FACT_BONUS
    assert out[SHAPING_PROBE_SENSITIVE] == 0.0


def test_ask_no_facts_no_bonus():
    ask = Ask(question="...", language="en")
    out = compute_shaping(0, ask, [], False, "medium")
    assert out[SHAPING_ASK_FACT] == 0.0


def test_probe_correct_topic_yields_bonus():
    probe = Probe(question="...", sensitive_topic="dv", language="en")
    out = compute_shaping(2, probe, ["dv_present"], False, "medium")
    assert out[SHAPING_PROBE_SENSITIVE] == PROBE_SENSITIVE_BONUS


def test_probe_wrong_topic_no_bonus():
    probe = Probe(question="...", sensitive_topic="caste", language="en")
    out = compute_shaping(2, probe, ["dv_present"], False, "medium")
    assert out[SHAPING_PROBE_SENSITIVE] == 0.0


def test_probe_during_sim_leak_no_bonus():
    probe = Probe(question="...", sensitive_topic="dv", language="en")
    out = compute_shaping(2, probe, ["dv_present"], True, "medium")
    assert out[SHAPING_PROBE_SENSITIVE] == 0.0


def test_late_turn_penalty():
    ask = Ask(question="...", language="en")
    out = compute_shaping(15, ask, [], False, "medium")
    assert out[SHAPING_LATE_TURN] == LATE_TURN_PENALTY


def test_early_turn_no_late_penalty():
    ask = Ask(question="...", language="en")
    out = compute_shaping(14, ask, [], False, "medium")
    assert out[SHAPING_LATE_TURN] == 0.0


def test_explain_jargon_low_literacy_penalty():
    explain = Explain(content="see section 125 of crpc", target_literacy="low", language="en")
    out = compute_shaping(0, explain, [], False, "low")
    assert out[SHAPING_JARGON] == JARGON_PENALTY


def test_explain_no_jargon_no_penalty():
    explain = Explain(
        content="we will fill the form together", target_literacy="low", language="en"
    )
    out = compute_shaping(0, explain, [], False, "low")
    assert out[SHAPING_JARGON] == 0.0


def test_explain_jargon_high_literacy_no_penalty():
    # we only penalize jargon for low-literacy citizens
    explain = Explain(content="see section 125 of crpc", target_literacy="high", language="en")
    out = compute_shaping(0, explain, [], False, "high")
    assert out[SHAPING_JARGON] == 0.0


def test_positive_cap_no_op_under_threshold():
    running = {
        SHAPING_ASK_FACT: 0.10,
        SHAPING_PROBE_SENSITIVE: 0.20,
        SHAPING_LATE_TURN: -0.30,
        SHAPING_JARGON: 0.0,
    }
    capped = cap_positive_shaping(running)
    assert capped == running


def test_positive_cap_scales_proportionally_when_exceeded():
    # 0.5 + 0.3 = 0.8 > 0.4 -> scale by 0.4/0.8 = 0.5
    running = {
        SHAPING_ASK_FACT: 0.5,
        SHAPING_PROBE_SENSITIVE: 0.3,
        SHAPING_LATE_TURN: -0.5,
        SHAPING_JARGON: 0.0,
    }
    capped = cap_positive_shaping(running)
    assert capped[SHAPING_ASK_FACT] == 0.25
    assert capped[SHAPING_PROBE_SENSITIVE] == 0.15
    assert capped[SHAPING_LATE_TURN] == -0.5
    pos_sum = sum(v for v in capped.values() if v > 0 and v != -0.5)
    assert pos_sum <= POSITIVE_SHAPING_CAP + 1e-9
