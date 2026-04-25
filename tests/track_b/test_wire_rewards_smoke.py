"""smoke test for scripts/wire_rewards.build_env. confirms my reward + shaping fns
wire cleanly into track-a's env and a full episode produces a complete breakdown.

if this test starts failing it almost always means the env <-> rewards seam drifted —
either RewardFn or ShapingFn signature changed on one side, or the breakdown shape
diverged from interface.ALL_KEYS.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from nyaya_mitra.interface import (
    ALL_KEYS,
    ActionPlan,
    ApplicationPath,
    Ask,
    Finalize,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    SchemeRecommendation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _import_wire_rewards():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module("scripts.wire_rewards")
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


def _hand_coded_plan() -> ActionPlan:
    return ActionPlan(
        schemes=[
            SchemeRecommendation(
                scheme_id="pm_kisan",
                rationale_facts=["occupation_farmer"],
                required_documents=["Aadhaar"],
                application_path=ApplicationPath(),
            )
        ],
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="dv at home",
                forum="magistrate",
                procedural_steps=["file dv-1"],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority="DLSA", contact_id="dlsa_ludhiana"
                ),
                required_documents=["id"],
            )
        ],
        most_important_next_step="contact dlsa",
        plain_summary=PlainSummary(language="en", text="we will help"),
    )


def test_build_env_returns_wired_env_with_full_breakdown():
    wire = _import_wire_rewards()
    env = wire.build_env()
    env.reset(seed=1)
    res_step = env.step(Ask(question="tell me more", language="en"))
    assert not res_step.done
    res_term = env.step(Finalize(plan=_hand_coded_plan()))
    assert res_term.done
    assert "reward_breakdown" in res_term.info
    breakdown = res_term.info["reward_breakdown"]
    for k in ALL_KEYS:
        assert k in breakdown, f"missing reward key {k}"
    assert isinstance(breakdown["total"], float)


def test_build_env_shaping_running_accumulates():
    wire = _import_wire_rewards()
    env = wire.build_env()
    env.reset(seed=1)
    env.step(Ask(question="anything more?", language="en"))
    env.step(Ask(question="and?", language="en"))
    res = env.step(Finalize(plan=_hand_coded_plan()))
    assert "shaping_running" in res.info
    assert isinstance(res.info["shaping_running"], dict)


def test_build_env_respects_max_turns_override():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=4)
    assert env.max_turns == 4


def test_build_env_relevant_facts_override_passes_through():
    """override flows through DuckTypedKB into the adapter."""
    wire = _import_wire_rewards()
    custom = {"pm_kisan": {"occupation_farmer"}}
    env = wire.build_env(relevant_facts=custom)
    env.reset(seed=1)
    res = env.step(Finalize(plan=_hand_coded_plan()))
    assert res.done
