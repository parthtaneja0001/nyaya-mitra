"""tests for the smart-canned citizen sim. all responses are deterministic and
profile-driven; no llm involved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.interface import (
    Ask,
    Behavior,
    CitizenProfile,
    Explain,
    Probe,
    SituationSpecific,
)


@dataclass
class FakeTurn:
    actor: str
    payload: dict[str, Any] = field(default_factory=dict)
    revealed: list[str] = field(default_factory=list)


def _profile(
    *,
    demographics: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    sensitive: dict[str, Any] | None = None,
    hidden: dict[str, Any] | None = None,
    trust: str = "neutral",
    literacy: str = "medium",
    language: str = "en",
) -> CitizenProfile:
    return CitizenProfile(
        seed=0,
        demographics=demographics or {},
        economic=economic or {},
        family={},
        situation_specific=SituationSpecific(
            presenting_issue="x",
            hidden_facts=hidden or {},
            sensitive_facts=sensitive or {},
        ),
        behavior=Behavior(
            trust_level=trust,
            verbosity="med",
            language_preference=language,
            literacy=literacy,
            initial_vague_query="hi, i need help",
        ),
    )


@pytest.fixture
def sim() -> CitizenSimulator:
    return CitizenSimulator()


def test_initial_utterance_returns_profile_query(sim: CitizenSimulator):
    p = _profile()
    p.behavior.initial_vague_query = "i need help with farm and home"
    assert sim.initial_utterance(p) == "i need help with farm and home"


def test_ask_reveals_first_applicable_fact(sim: CitizenSimulator):
    p = _profile(economic={"occupation": "farmer", "holds_cultivable_land": True})
    out = sim.respond(p, [], Ask(question="tell me more", language="en"))
    assert "farmer" in out.lower()


def test_ask_skips_already_revealed_facts(sim: CitizenSimulator):
    p = _profile(
        demographics={"gender": "female", "age": 30, "residence": "rural"},
        economic={"bpl_household": True, "existing_lpg_in_family": False},
    )
    transcript = [FakeTurn(actor="citizen", revealed=["gender_female", "bpl_household"])]
    out = sim.respond(p, transcript, Ask(question="more?", language="en"))
    assert "BPL" not in out
    assert "woman" not in out.lower()


def test_ask_runs_out_of_facts_returns_exhausted_reply(sim: CitizenSimulator):
    p = _profile()
    transcript = [
        FakeTurn(
            actor="citizen",
            revealed=[
                fid
                for fid, _, _ in __import__(
                    "nyaya_mitra.citizen.simulator", fromlist=["_REVEAL_FACTS"]
                )._REVEAL_FACTS
            ],
        ),
    ]
    out = sim.respond(p, transcript, Ask(question="anything else?", language="en"))
    assert "not sure" in out.lower() or "samajh" in out.lower()


def test_probe_dv_with_dv_present_reveals(sim: CitizenSimulator):
    p = _profile(
        demographics={"gender": "female"},
        sensitive={"dv_present": True},
        trust="open",
    )
    out = sim.respond(
        p, [], Probe(question="anything at home?", sensitive_topic="dv", language="en")
    )
    assert "husband" in out.lower() or "hits" in out.lower()


def test_probe_dv_without_dv_returns_negative(sim: CitizenSimulator):
    p = _profile(demographics={"gender": "female"}, sensitive={})
    out = sim.respond(p, [], Probe(question="any abuse?", sensitive_topic="dv", language="en"))
    assert "no" in out.lower() or "not the case" in out.lower()


def test_wary_citizen_defers_first_probe(sim: CitizenSimulator):
    p = _profile(
        demographics={"gender": "female"},
        sensitive={"dv_present": True},
        trust="wary",
    )
    out = sim.respond(p, [], Probe(question="any abuse?", sensitive_topic="dv", language="en"))
    assert "rather not" in out.lower() or "abhi" in out.lower()


def test_wary_citizen_discloses_after_two_advisor_turns(sim: CitizenSimulator):
    p = _profile(
        demographics={"gender": "female"},
        sensitive={"dv_present": True},
        trust="wary",
    )
    transcript = [
        FakeTurn(actor="advisor", payload={"type": "ASK"}),
        FakeTurn(actor="citizen"),
        FakeTurn(actor="advisor", payload={"type": "ASK"}),
        FakeTurn(actor="citizen"),
    ]
    out = sim.respond(
        p, transcript, Probe(question="anything at home?", sensitive_topic="dv", language="en")
    )
    assert "husband" in out.lower() or "hits" in out.lower()


def test_explain_low_literacy_returns_simple_reply(sim: CitizenSimulator):
    p = _profile(literacy="low")
    out = sim.respond(
        p, [], Explain(content="here is the law", target_literacy="low", language="en")
    )
    assert "what should i do" in out.lower() or "samajh" in out.lower()


def test_explain_high_literacy_returns_acknowledgment(sim: CitizenSimulator):
    p = _profile(literacy="high")
    out = sim.respond(
        p, [], Explain(content="here is the law", target_literacy="medium", language="en")
    )
    assert "follow up" in out.lower() or "dekh" in out.lower()


def test_language_preference_hi_yields_devanagari(sim: CitizenSimulator):
    p = _profile(
        economic={"occupation": "farmer", "holds_cultivable_land": True},
        language="hi",
    )
    out = sim.respond(p, [], Ask(question="bataiye", language="hi"))
    assert "किसान" in out


def test_language_preference_hinglish_yields_latin_script(sim: CitizenSimulator):
    p = _profile(
        economic={"occupation": "farmer", "holds_cultivable_land": True},
        language="hinglish",
    )
    out = sim.respond(p, [], Ask(question="aur batao", language="hinglish"))
    assert "farmer" in out.lower() or "kheti" in out.lower()


def test_probe_unknown_topic_returns_negative(sim: CitizenSimulator):
    p = _profile()
    out = sim.respond(p, [], Probe(question="x", sensitive_topic="hiv_status", language="en"))
    assert out


def test_ask_handles_missing_economic_fields_gracefully(sim: CitizenSimulator):
    p = _profile(economic={})
    out = sim.respond(p, [], Ask(question="tell me", language="en"))
    assert isinstance(out, str) and out
