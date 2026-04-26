"""parametrized applicability tests for the 4 newly added frameworks: posh_act_2013,
rti_act_2005, pwd_act_2016, sc_st_atrocities_act_1989. PLAN.md target is >=8 cases per
framework; this is the v1 cut."""

from __future__ import annotations

from typing import Any

import pytest

from nyaya_mitra.interface import Behavior, CitizenProfile, SituationSpecific
from nyaya_mitra.knowledge.applicability import (
    posh_act_2013,
    pwd_act_2016,
    rti_act_2005,
    sc_st_atrocities_act_1989,
)


def _profile(
    *,
    demographics: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    sensitive: dict[str, Any] | None = None,
    hidden: dict[str, Any] | None = None,
) -> CitizenProfile:
    return CitizenProfile(
        seed=0,
        demographics=demographics or {},
        economic=economic or {},
        family={},
        situation_specific=SituationSpecific(
            presenting_issue="",
            hidden_facts=hidden or {},
            sensitive_facts=sensitive or {},
        ),
        behavior=Behavior(
            trust_level="neutral",
            verbosity="med",
            language_preference="en",
            literacy="medium",
            initial_vague_query="",
        ),
    )


@pytest.mark.parametrize(
    "demographics, economic, hidden, expected",
    [
        (
            {"gender": "female"},
            {"formally_employed": True},
            {"sexual_harassment_at_workplace": True},
            True,
        ),
        (
            {"gender": "female"},
            {"is_wage_worker": True},
            {"sexual_harassment_at_workplace": True},
            True,
        ),
        (
            {"gender": "male"},
            {"formally_employed": True},
            {"sexual_harassment_at_workplace": True},
            False,
        ),
        (
            {"gender": "female"},
            {},
            {"sexual_harassment_at_workplace": True},
            False,
        ),
        (
            {"gender": "female"},
            {"formally_employed": True},
            {},
            False,
        ),
    ],
)
def test_posh_act(demographics, economic, hidden, expected):
    p = _profile(demographics=demographics, economic=economic, hidden=hidden)
    ok, _ = posh_act_2013.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "hidden, expected",
    [
        ({"seeks_government_information": True}, True),
        ({"denied_govt_info": True}, True),
        ({}, False),
        ({"seeks_government_information": False}, False),
    ],
)
def test_rti_act(hidden, expected):
    p = _profile(hidden=hidden)
    ok, _ = rti_act_2005.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "sensitive, hidden, expected",
    [
        ({"disability_present": True}, {"disability_discrimination_present": True}, True),
        ({"severe_disability": True}, {"disability_discrimination_present": True}, True),
        ({"disability": "polio"}, {"disability_discrimination_present": True}, True),
        ({"disability_present": True}, {}, False),
        ({}, {"disability_discrimination_present": True}, False),
    ],
)
def test_pwd_act(sensitive, hidden, expected):
    p = _profile(sensitive=sensitive, hidden=hidden)
    ok, _ = pwd_act_2016.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "sensitive, hidden, expected",
    [
        ({"caste": "scheduled_caste"}, {"caste_atrocity_present": True}, True),
        ({"caste": "scheduled_tribe"}, {"caste_based_violence": True}, True),
        ({"caste": "scheduled_caste"}, {"caste_humiliation": True}, True),
        ({"caste": "general"}, {"caste_atrocity_present": True}, False),
        ({}, {"caste_atrocity_present": True}, False),
        ({"caste": "scheduled_caste"}, {}, False),
    ],
)
def test_sc_st_atrocities_act(sensitive, hidden, expected):
    p = _profile(sensitive=sensitive, hidden=hidden)
    ok, _ = sc_st_atrocities_act_1989.check(p)
    assert ok is expected
