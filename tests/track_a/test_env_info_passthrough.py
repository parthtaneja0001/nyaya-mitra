"""env info passthrough for rewards integration: shaping_fn ctor arg, info[shaping_running],
info[negated_facts], info[format_violation]. matches reward_design.md "what track A passes through"."""

from __future__ import annotations

import pytest

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.interface import (
    ActionPlan,
    Ask,
    Behavior,
    CitizenProfile,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    SituationSpecific,
)
from nyaya_mitra.knowledge.loader import KnowledgeBase


def _minimal_plan() -> ActionPlan:
    return ActionPlan(
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="x",
                forum="magistrate",
                procedural_steps=["a"],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority="DLSA", contact_id="dlsa_ludhiana"
                ),
                required_documents=["b"],
            )
        ],
        most_important_next_step="contact dlsa",
        plain_summary=PlainSummary(language="en", text="..."),
    )


def _profile_no_kb() -> CitizenProfile:
    return CitizenProfile(
        seed=0,
        situation_specific=SituationSpecific(presenting_issue="x"),
        behavior=Behavior(
            trust_level="neutral",
            verbosity="med",
            language_preference="en",
            literacy="low",
            initial_vague_query="hello",
        ),
    )


@pytest.fixture
def env() -> NyayaMitraEnv:
    return NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())


def test_terminal_info_emits_format_violation_default_false(env: NyayaMitraEnv):
    env.reset(seed=1)
    from nyaya_mitra.interface import Finalize

    res = env.step(Finalize(plan=_minimal_plan()))
    assert res.info["format_violation"] is False


def test_terminal_info_emits_shaping_running(env: NyayaMitraEnv):
    env.reset(seed=1)
    from nyaya_mitra.interface import Finalize

    res = env.step(Finalize(plan=_minimal_plan()))
    assert "shaping_running" in res.info
    assert isinstance(res.info["shaping_running"], dict)


def test_step_info_emits_negated_facts_key(env: NyayaMitraEnv):
    env.reset(seed=1)
    res = env.step(Ask(question="anything else?", language="en"))
    assert "negated_facts" in res.info
    assert isinstance(res.info["negated_facts"], list)


def test_terminal_info_emits_negated_facts_key(env: NyayaMitraEnv):
    env.reset(seed=1)
    from nyaya_mitra.interface import Finalize

    res = env.step(Finalize(plan=_minimal_plan()))
    assert "negated_facts" in res.info
    assert isinstance(res.info["negated_facts"], list)


def test_shaping_fn_default_none_runs_clean():
    """absent shaping_fn must not break the loop; shaping_running stays empty."""
    env = NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor(), shaping_fn=None)
    env.reset(seed=1)
    env.step(Ask(question="hi", language="en"))
    from nyaya_mitra.interface import Finalize

    res = env.step(Finalize(plan=_minimal_plan()))
    assert res.info["shaping_running"] == {}


def test_shaping_fn_called_per_step_and_accumulated():
    calls: list[tuple] = []

    def fake_shaping(turn_index, action, revealed, sim_leak, literacy):
        calls.append((turn_index, type(action).__name__, list(revealed), sim_leak, literacy))
        return {"shaping_ask_fact": 0.02}

    env = NyayaMitraEnv(
        KnowledgeBase(),
        CitizenSimulator(),
        FactExtractor(),
        shaping_fn=fake_shaping,
    )
    env.reset(seed=1)
    env.step(Ask(question="q1", language="en"))
    env.step(Ask(question="q2", language="en"))
    from nyaya_mitra.interface import Finalize

    res = env.step(Finalize(plan=_minimal_plan()))
    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2
    assert res.info["shaping_running"]["shaping_ask_fact"] == pytest.approx(0.04)


def test_shaping_fn_receives_citizen_literacy():
    seen: list[str] = []

    def fake_shaping(turn_index, action, revealed, sim_leak, literacy):
        seen.append(literacy)
        return {}

    env = NyayaMitraEnv(
        KnowledgeBase(),
        CitizenSimulator(),
        FactExtractor(),
        shaping_fn=fake_shaping,
    )
    env.reset(seed=1)
    env.step(Ask(question="hi", language="en"))
    assert seen == ["low"]


def test_extractor_extract_negations_detects_explicit_negation():
    ex = FactExtractor()
    profile = _profile_no_kb()
    out = ex.extract_negations(profile, "I'm not a farmer, never have been")
    assert "occupation_farmer" in out


def test_extractor_extract_negations_skips_positive_mentions():
    ex = FactExtractor()
    profile = _profile_no_kb()
    out = ex.extract_negations(profile, "I am a farmer with a small plot")
    assert out == []


def test_extractor_extract_strips_negated_from_revealed():
    ex = FactExtractor()
    profile = _profile_no_kb()
    revealed = ex.extract(profile, "I'm not a farmer", set())
    assert "occupation_farmer" not in revealed


def test_state_includes_shaping_running_and_negated_facts(
    env: NyayaMitraEnv, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NYAYA_DEBUG", "1")
    env.reset(seed=1)
    s = env.state()
    assert "shaping_running" in s
    assert "negated_facts" in s
