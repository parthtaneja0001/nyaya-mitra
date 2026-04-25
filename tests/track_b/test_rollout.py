"""rollout runner tests. uses scripts.wire_rewards.build_env and a scripted
advisor that finalizes after a fixed number of turns. covers:

- happy path: advisor finalizes -> EpisodeResult.finalized=True, breakdown complete
- truncation: advisor never finalizes -> EpisodeResult.truncated_by_env=True
- advisor exception: surfaced as EpisodeResult.error, episode terminates cleanly
- run_episodes uses env_factory and isolates state across seeds
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
from training.rollout import RolloutState, run_episode, run_episodes

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _import_wire_rewards():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module("scripts.wire_rewards")
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


def _hand_plan() -> ActionPlan:
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


def _finalize_at_turn(target: int):
    """advisor: ask N times then finalize."""

    def _advisor(obs, state: RolloutState):
        if state.turn_index >= target:
            return Finalize(plan=_hand_plan())
        return Ask(question=f"please tell me more (turn {state.turn_index})", language="en")

    return _advisor


def test_run_episode_happy_path():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=10)
    result = run_episode(env, _finalize_at_turn(2), seed=1)
    assert result.error is None
    assert result.finalized is True
    assert not result.truncated_by_env
    assert result.total_reward != 0.0 or result.final_breakdown
    for k in ALL_KEYS:
        assert k in result.final_breakdown


def test_run_episode_collects_turn_logs():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=10)
    result = run_episode(env, _finalize_at_turn(3), seed=2)
    assert result.error is None
    assert len(result.turns) >= 1
    # the final turn must be the FINALIZE
    assert result.turns[-1].action.type == "FINALIZE"
    # earlier turns should be ASK
    if len(result.turns) > 1:
        assert all(t.action.type == "ASK" for t in result.turns[:-1])


def test_run_episode_truncation_when_advisor_never_finalizes():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=4)
    # advisor that always asks; should hit max_turns
    result = run_episode(env, lambda o, s: Ask(question="x", language="en"), seed=3)
    assert result.error is None
    assert result.truncated_by_env is True
    assert result.finalized is False


def test_run_episode_surfaces_advisor_exception():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=10)

    def bad_advisor(obs, state):
        raise RuntimeError("advisor blew up")

    result = run_episode(env, bad_advisor, seed=4)
    assert result.error is not None
    assert "advisor blew up" in result.error
    # episode terminated; no further state expected
    assert result.finalized is False


def test_run_episodes_uses_factory_and_returns_per_seed():
    wire = _import_wire_rewards()
    seeds = [1, 2, 3]
    results = run_episodes(
        env_factory=lambda: wire.build_env(max_turns=6),
        advisor=_finalize_at_turn(1),
        seeds=seeds,
    )
    assert len(results) == 3
    assert [r.seed for r in results] == seeds
    for r in results:
        assert r.error is None
        assert r.finalized is True


def test_run_episodes_calls_on_episode_hook():
    wire = _import_wire_rewards()
    seen: list[int] = []
    run_episodes(
        env_factory=lambda: wire.build_env(max_turns=6),
        advisor=_finalize_at_turn(1),
        seeds=[10, 11],
        on_episode=lambda r: seen.append(r.seed),
    )
    assert seen == [10, 11]


def test_rollout_state_carries_observation_history():
    wire = _import_wire_rewards()
    env = wire.build_env(max_turns=10)

    captured: list[RolloutState] = []

    def advisor(obs, state):
        captured.append(state)
        if state.turn_index >= 3:
            return Finalize(plan=_hand_plan())
        return Ask(question="more", language="en")

    run_episode(env, advisor, seed=1)
    assert len(captured) >= 1
    # history grows monotonically
    sizes = [len(s.history) for s in captured]
    assert sizes == sorted(sizes)
    # last call's state should reflect advancement
    assert captured[-1].turn_index >= captured[0].turn_index
