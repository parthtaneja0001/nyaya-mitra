"""determinism test: same seed → same transcript hash, twice in a row.

guards against accidental nondeterminism creeping into the env (random.choice in
the sim, dict-iteration order escaping into output, etc). hash-checks both:
- the transcript content (utterances + advisor actions)
- the derived ground truth (so kb-checker output is stable)

if this test starts failing, the env is no longer reproducible — the demo's
'this seed produces this transcript' story breaks. STOP and re-sync."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.interface import (
    ActionPlan,
    Ask,
    Finalize,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    Probe,
)
from nyaya_mitra.knowledge.loader import KnowledgeBase


def _hand_coded_plan() -> ActionPlan:
    return ActionPlan(
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="ongoing dv at home",
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


def _run_episode(seed: int, monkeypatch_setenv) -> dict[str, Any]:
    """run a fixed scripted episode and return a hashable snapshot."""
    monkeypatch_setenv("NYAYA_DEBUG", "1")
    env = NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())
    env.reset(seed=seed)
    env.step(Ask(question="tell me about your situation", language="en"))
    env.step(Ask(question="anything else?", language="en"))
    env.step(Probe(question="anything at home?", sensitive_topic="dv", language="en"))
    res = env.step(Finalize(plan=_hand_coded_plan()))

    state = env.state()
    snap = {
        "transcript": state["transcript"],
        "elicited_facts": sorted(state["elicited_facts"]),
        "negated_facts": sorted(state["negated_facts"]),
        "derived_ground_truth": state["profile"]["derived_ground_truth"],
        "reward_breakdown": res.info["reward_breakdown"],
    }
    return snap


def _hash(snap: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(snap, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_same_seed_same_hash(seed: int, monkeypatch: pytest.MonkeyPatch):
    a = _run_episode(seed, monkeypatch.setenv)
    b = _run_episode(seed, monkeypatch.setenv)
    assert _hash(a) == _hash(b), (
        f"seed={seed} produced different transcripts on two runs "
        f"— sim or extractor introduced nondeterminism"
    )


def test_different_seeds_produce_different_hashes(monkeypatch: pytest.MonkeyPatch):
    """sanity: seeds 0 and 1 must differ. catches the case where seed is ignored."""
    a = _run_episode(0, monkeypatch.setenv)
    b = _run_episode(1, monkeypatch.setenv)
    assert _hash(a) != _hash(b), "seed 0 and seed 1 produced the same transcript"


def test_episode_state_clears_on_close(monkeypatch: pytest.MonkeyPatch):
    """env.close() releases internal state so a re-reset starts clean."""
    monkeypatch.setenv("NYAYA_DEBUG", "1")
    env = NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())
    env.reset(seed=0)
    env.step(Ask(question="hi", language="en"))
    env.close()
    assert env.state() == {}
