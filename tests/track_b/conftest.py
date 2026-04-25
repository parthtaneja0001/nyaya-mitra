"""shared fixtures for track-b tests. all builders here so individual tests
stay focused on the specific behavior under test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from nyaya_mitra.interface import (
    ActionPlan,
    ApplicationPath,
    Ask,
    Behavior,
    CitizenObservation,
    CitizenProfile,
    DerivedGroundTruth,
    Explain,
    Finalize,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    Probe,
    SchemeRecommendation,
    SituationSpecific,
)
from nyaya_mitra.rewards.context import RewardContext, Turn


class FakeKB:
    """in-memory kb satisfying the rewards.kb_protocol surface. tests build
    instances via FakeKB.builder() so each test owns its data."""

    def __init__(
        self,
        schemes: dict[str, dict[str, Any]] | None = None,
        frameworks: dict[str, dict[str, Any]] | None = None,
        contacts: set[tuple[str, str]] | None = None,
        relevant_facts: dict[str, set[str]] | None = None,
    ) -> None:
        self._schemes = schemes or {}
        self._frameworks = frameworks or {}
        self._contacts = contacts or set()
        self._relevant = relevant_facts or {}

    def has_scheme(self, scheme_id: str) -> bool:
        return scheme_id in self._schemes

    def has_framework(self, framework_id: str) -> bool:
        return framework_id in self._frameworks

    def has_contact(self, authority: str, contact_id: str) -> bool:
        return (authority, contact_id) in self._contacts

    def documents_for_scheme(self, scheme_id: str) -> list[str]:
        s = self._schemes.get(scheme_id) or {}
        return list(s.get("required_documents") or [])

    def documents_for_framework(self, framework_id: str) -> list[str]:
        f = self._frameworks.get(framework_id) or {}
        return list(f.get("required_documents") or [])

    def procedural_steps_for_framework(self, framework_id: str) -> list[str]:
        f = self._frameworks.get(framework_id) or {}
        return list(f.get("procedural_steps") or [])

    def forum_for_framework(self, framework_id: str) -> str | None:
        f = self._frameworks.get(framework_id) or {}
        v = f.get("forum")
        return v if isinstance(v, str) else None

    def legal_aid_authority_for_framework(self, framework_id: str) -> str | None:
        f = self._frameworks.get(framework_id) or {}
        v = f.get("legal_aid_authority")
        return v if isinstance(v, str) else None

    def relevant_facts_for_scheme(self, scheme_id: str) -> set[str]:
        return set(self._relevant.get(scheme_id, set()))

    def relevant_facts_for_framework(self, framework_id: str) -> set[str]:
        return set(self._relevant.get(framework_id, set()))


def make_profile(
    *,
    eligible_schemes: Sequence[str] = (),
    applicable_frameworks: Sequence[str] = (),
    sensitive_facts: dict[str, Any] | None = None,
    literacy: str = "medium",
    language: str = "en",
    seed: int = 1,
) -> CitizenProfile:
    return CitizenProfile(
        seed=seed,
        demographics={"gender": "female", "state": "punjab"},
        economic={"occupation": "farmer", "holds_cultivable_land": True},
        family={},
        situation_specific=SituationSpecific(
            presenting_issue="test issue",
            hidden_facts={},
            sensitive_facts=dict(sensitive_facts or {}),
        ),
        behavior=Behavior(
            trust_level="neutral",
            verbosity="med",
            language_preference=language,  # type: ignore[arg-type]
            literacy=literacy,  # type: ignore[arg-type]
            initial_vague_query="i need help",
        ),
        derived_ground_truth=DerivedGroundTruth(
            eligible_schemes=list(eligible_schemes),
            applicable_frameworks=list(applicable_frameworks),
        ),
    )


def make_scheme_rec(
    scheme_id: str,
    *,
    rationale_facts: Sequence[str] = (),
    documents: Sequence[str] = (),
) -> SchemeRecommendation:
    return SchemeRecommendation(
        scheme_id=scheme_id,
        rationale_facts=list(rationale_facts),
        required_documents=list(documents),
        application_path=ApplicationPath(),
    )


def make_legal_rec(
    framework_id: str,
    *,
    forum: str = "magistrate",
    procedural_steps: Sequence[str] = (),
    documents: Sequence[str] = (),
    authority: str = "DLSA",
    contact_id: str = "dlsa_test",
) -> LegalRouteRecommendation:
    return LegalRouteRecommendation(
        framework_id=framework_id,
        applicable_situation="x",
        forum=forum,
        procedural_steps=list(procedural_steps),
        free_legal_aid_contact=FreeLegalAidContact(
            authority=authority,  # type: ignore[arg-type]
            contact_id=contact_id,
        ),
        required_documents=list(documents),
    )


def make_plan(
    *,
    schemes: Sequence[SchemeRecommendation] = (),
    legal_routes: Sequence[LegalRouteRecommendation] = (),
    summary_lang: str = "en",
    next_step: str = "contact dlsa today",
    summary_text: str = "we will help with the matter step by step.",
) -> ActionPlan:
    return ActionPlan(
        schemes=list(schemes),
        legal_routes=list(legal_routes),
        most_important_next_step=next_step,
        plain_summary=PlainSummary(language=summary_lang, text=summary_text),
    )


def make_turn(
    index: int,
    action,
    *,
    info: dict[str, Any] | None = None,
) -> Turn:
    return Turn(index=index, action=action, citizen_observation=None, info=dict(info or {}))


def make_citizen_turn(
    index: int,
    *,
    revealed: Sequence[str] = (),
    info: dict[str, Any] | None = None,
) -> Turn:
    base = dict(info or {})
    base.setdefault("revealed", list(revealed))
    return Turn(
        index=index,
        action=None,
        citizen_observation=CitizenObservation(
            citizen_utterance="ok",
            language="en",
            turn=index,
            max_turns=20,
            elicited_facts=list(revealed),
            facts_revealed_this_turn=list(revealed),
        ),
        info=base,
    )


def make_ctx(
    *,
    profile: CitizenProfile,
    plan: ActionPlan,
    transcript: Sequence[Turn] = (),
    elicited_facts: Sequence[str] = (),
    kb: FakeKB | None = None,
    info: dict[str, Any] | None = None,
) -> RewardContext:
    return RewardContext(
        profile=profile,
        plan=plan,
        transcript=list(transcript),
        elicited_facts=set(elicited_facts),
        kb=kb or FakeKB(),
        info=dict(info or {"max_turns": 20}),
    )


@pytest.fixture
def kb_basic() -> FakeKB:
    """small kb with two schemes, one framework, two contacts, plus relevant facts."""
    return FakeKB(
        schemes={
            "pm_kisan": {
                "required_documents": ["Aadhaar", "Bank account", "Land record"],
            },
            "pmuy": {
                "required_documents": ["Aadhaar", "BPL certificate", "Bank account"],
            },
        },
        frameworks={
            "domestic_violence_act_2005": {
                "forum": "Magistrate of the First Class",
                "legal_aid_authority": "DLSA",
                "required_documents": ["Identity proof", "Address proof"],
                "procedural_steps": [
                    "approach protection officer",
                    "file dv-1 form",
                    "magistrate grants protection order",
                ],
            },
        },
        contacts={
            ("DLSA", "dlsa_ludhiana"),
            ("NALSA", "nalsa_central"),
            ("DLSA", "dlsa_test"),
        },
        relevant_facts={
            "pm_kisan": {"occupation_farmer", "land_small"},
            "pmuy": {"gender_female", "bpl_household", "no_lpg"},
            "domestic_violence_act_2005": {"gender_female", "dv_present"},
        },
    )


__all__ = [
    "FakeKB",
    "Ask",
    "Explain",
    "Finalize",
    "Probe",
    "make_citizen_turn",
    "make_ctx",
    "make_legal_rec",
    "make_plan",
    "make_profile",
    "make_scheme_rec",
    "make_turn",
]
