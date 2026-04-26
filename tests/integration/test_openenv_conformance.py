"""OpenEnv-conformance tests.

Verify that the env subclasses Environment properly, exposes a Rubric,
serves the canonical OpenEnv HTTP routes, and does an end-to-end round-trip
through the Action/Observation envelopes.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric, Sequential, WeightedSum

from nyaya_mitra.env.openenv_env import (
    NyayaAction,
    NyayaEnvironment,
    NyayaObservation,
    NyayaState,
)


def test_action_inherits_openenv_action():
    assert issubclass(NyayaAction, Action)


def test_observation_inherits_openenv_observation():
    assert issubclass(NyayaObservation, Observation)


def test_state_inherits_openenv_state():
    assert issubclass(NyayaState, State)


def test_environment_inherits_openenv_environment():
    assert issubclass(NyayaEnvironment, Environment)


def test_default_env_attaches_rubric_tree():
    env = NyayaEnvironment()
    assert env.rubric is not None
    assert isinstance(env.rubric, Rubric)
    # top-level should be a Sequential of fail-fast gates + WeightedSum of soft components
    assert isinstance(env.rubric, Sequential)
    children = list(env.rubric.named_rubrics())
    # at minimum: the 11 soft components live as descendants of the WeightedSum
    weighted_sum_count = sum(1 for _, c in children if isinstance(c, WeightedSum))
    assert weighted_sum_count >= 1
    # 14+ descendants total (3 gates + their wrapped rubrics + 11 components + WeightedSum)
    assert len(children) >= 14


def test_round_trip_through_openenv_env_records_rubric_scores():
    """end-to-end: reset → step(ASK) → step(FINALIZE).
    after FINALIZE the rubric component leaves should have last_score populated."""
    from nyaya_mitra.knowledge.loader import KnowledgeBase
    from nyaya_mitra.rewards import compute_shaping, make_env_reward_fn
    from nyaya_mitra.rewards.kb_adapter import DuckTypedKB

    kb = KnowledgeBase()
    env = NyayaEnvironment(
        reward_fn=make_env_reward_fn(DuckTypedKB(kb)),
        shaping_fn=compute_shaping,
    )

    obs = env.reset(seed=1)
    assert isinstance(obs, NyayaObservation)
    assert not obs.done

    obs = env.step(
        NyayaAction(advisor={"type": "ASK", "question": "tell me more", "language": "en"})
    )
    assert isinstance(obs, NyayaObservation)
    assert not obs.done

    finalize = NyayaAction(
        advisor={
            "type": "FINALIZE",
            "plan": {
                "schemes": [],
                "legal_routes": [
                    {
                        "framework_id": "domestic_violence_act_2005",
                        "applicable_situation": "x",
                        "forum": "magistrate",
                        "procedural_steps": ["a"],
                        "free_legal_aid_contact": {
                            "authority": "DLSA",
                            "contact_id": "dlsa_ludhiana",
                        },
                        "required_documents": ["b"],
                    }
                ],
                "most_important_next_step": "contact dlsa",
                "plain_summary": {"language": "en", "text": "we will help"},
            },
        }
    )
    obs = env.step(finalize)
    assert obs.done
    assert isinstance(obs.reward, float)
    assert obs.reward_breakdown is not None
    # rubric introspection: at least some component leaves recorded a score
    scored = [
        (n, c.last_score)
        for n, c in env.rubric.named_rubrics()
        if c.last_score is not None and not isinstance(c, (Sequential, WeightedSum))
    ]
    assert len(scored) >= 5


def test_state_property_returns_NyayaState():
    env = NyayaEnvironment()
    env.reset(seed=1)
    s = env.state
    assert isinstance(s, NyayaState)
    assert s.episode_id is not None
    assert s.step_count == 0


def test_state_does_not_leak_profile():
    """the public `state` property must not leak ground-truth profile fields.
    that's what the NYAYA_DEBUG-gated NyayaMitraEnv.state() is for."""
    env = NyayaEnvironment()
    env.reset(seed=1)
    s = env.state
    public = s.model_dump()
    assert "profile" not in public
    assert "derived_ground_truth" not in str(public)


def test_create_app_routes_exposed():
    """canonical OpenEnv server has /reset, /step, /state, /metadata, /schema, /docs."""
    from nyaya_mitra.env.server import app

    paths = {r.path for r in app.routes if hasattr(r, "path")}
    for required in ("/reset", "/step", "/state", "/metadata", "/healthz", "/docs"):
        assert required in paths, f"missing canonical route {required}"


def test_openenv_yaml_canonical_format():
    """openenv.yaml uses spec_version 1, type space, runtime fastapi (matches echo_env)."""
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent.parent
    cfg = yaml.safe_load((repo / "openenv.yaml").read_text(encoding="utf-8"))
    assert cfg["spec_version"] == 1
    assert cfg["type"] == "space"
    assert cfg["runtime"] == "fastapi"
    assert cfg["app"].endswith(":app")
    assert isinstance(cfg["port"], int)
