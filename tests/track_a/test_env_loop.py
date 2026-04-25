"""end-to-end env loop with the toy kb and a hand-coded advisor that finalizes immediately."""

from __future__ import annotations

import pytest

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.interface import (
    ActionPlan,
    ApplicationPath,
    Finalize,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    SchemeRecommendation,
)
from nyaya_mitra.knowledge.loader import KnowledgeBase


@pytest.fixture
def env() -> NyayaMitraEnv:
    return NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())


def _hand_coded_plan() -> ActionPlan:
    return ActionPlan(
        schemes=[
            SchemeRecommendation(
                scheme_id="pm_kisan",
                rationale_facts=["occupation_farmer"],
                required_documents=["Aadhaar", "Bank account details", "Land record"],
                application_path=ApplicationPath(
                    online_url="https://pmkisan.gov.in/",
                    offline_office="village patwari",
                    offline_steps=["visit patwari", "submit aadhaar + land record"],
                ),
            ),
        ],
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="ongoing dv at home",
                forum="magistrate of the first class",
                procedural_steps=[
                    "approach protection officer",
                    "file dv-1 application",
                ],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority="DLSA", contact_id="dlsa_ludhiana"
                ),
                required_documents=["identity proof", "address proof"],
            ),
        ],
        most_important_next_step="contact dlsa ludhiana today",
        plain_summary=PlainSummary(
            language="en",
            text="we will route you to dlsa for the dv matter and help you apply for pm-kisan.",
        ),
    )


def test_kb_loads_with_toy_data():
    kb = KnowledgeBase()
    assert "pm_kisan" in kb.scheme_ids()
    assert "pmuy" in kb.scheme_ids()
    assert "domestic_violence_act_2005" in kb.framework_ids()
    assert "dlsa_ludhiana" in kb.all_contact_ids()


def test_kb_validates_against_schemas():
    from nyaya_mitra.knowledge.validators import validate_kb

    errors = validate_kb()
    assert errors == [], f"kb schema violations: {errors}"


def test_reset_returns_observation(env: NyayaMitraEnv):
    obs = env.reset(seed=1)
    assert obs.turn == 0
    assert obs.max_turns == 20
    assert obs.citizen_utterance


def test_finalize_immediately_terminates(env: NyayaMitraEnv):
    env.reset(seed=1)
    result = env.step(Finalize(plan=_hand_coded_plan()))
    assert result.done is True
    assert result.observation is None
    assert result.info["phase"] == "terminal"
    assert "reward_breakdown" in result.info
    assert result.info["reward_breakdown"]["scheme_precision"] == 0.0


def test_state_locked_without_debug(env: NyayaMitraEnv, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NYAYA_DEBUG", raising=False)
    env.reset(seed=1)
    with pytest.raises(RuntimeError, match="NYAYA_DEBUG"):
        env.state()


def test_state_returns_snapshot_with_debug(env: NyayaMitraEnv, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NYAYA_DEBUG", "1")
    env.reset(seed=1)
    snap = env.state()
    assert "profile" in snap
    assert "elicited_facts" in snap
    assert "transcript" in snap


def test_derived_ground_truth_runs_kb_checkers(env: NyayaMitraEnv, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NYAYA_DEBUG", "1")
    env.reset(seed=0)
    truth = env.state()["profile"]["derived_ground_truth"]
    assert set(truth["eligible_schemes"]) == {"pm_kisan", "pmuy", "mgnrega"}
    assert set(truth["applicable_frameworks"]) == {"domestic_violence_act_2005"}


def test_step_after_done_raises(env: NyayaMitraEnv):
    env.reset(seed=1)
    env.step(Finalize(plan=_hand_coded_plan()))
    with pytest.raises(RuntimeError, match="already done"):
        env.step(Finalize(plan=_hand_coded_plan()))


def test_reward_fn_is_called_when_provided(env: NyayaMitraEnv):
    received: dict = {}

    def fake_reward_fn(profile, plan, transcript, elicited):
        received["called"] = True
        received["plan_id"] = plan.schemes[0].scheme_id
        return {"scheme_precision": 1.0, "total": 0.5}

    env.reward_fn = fake_reward_fn
    env.reset(seed=1)
    result = env.step(Finalize(plan=_hand_coded_plan()))
    assert received["called"]
    assert received["plan_id"] == "pm_kisan"
    assert result.info["reward_breakdown"]["scheme_precision"] == 1.0
    assert result.reward == 0.5
