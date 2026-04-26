"""parametrized eligibility tests for the 6 newly added schemes (pmjjby, apy, pmmvy,
iggndps, pmkmy, pmegp). PLAN.md target is >=8 cases per scheme; this is the v1 cut."""

from __future__ import annotations

from typing import Any

import pytest

from nyaya_mitra.interface import Behavior, CitizenProfile, SituationSpecific
from nyaya_mitra.knowledge.eligibility import (
    apy,
    iggndps,
    pmegp,
    pmjjby,
    pmkmy,
    pmmvy,
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
    "demographics, economic, expected",
    [
        ({"age": 18}, {"has_bank_account": True}, True),
        ({"age": 50}, {"has_bank_account": True}, True),
        ({"age": 35}, {"has_bank_account": True}, True),
        ({"age": 17}, {"has_bank_account": True}, False),
        ({"age": 51}, {"has_bank_account": True}, False),
        ({"age": 35}, {"has_bank_account": False}, False),
    ],
)
def test_pmjjby(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = pmjjby.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, expected",
    [
        ({"age": 18}, {"has_bank_account": True}, True),
        ({"age": 40}, {"has_bank_account": True}, True),
        ({"age": 28}, {"has_bank_account": True}, True),
        ({"age": 17}, {"has_bank_account": True}, False),
        ({"age": 41}, {"has_bank_account": True}, False),
        ({"age": 28}, {"has_bank_account": False}, False),
        ({"age": 28}, {"has_bank_account": True, "income_tax_payer": True}, False),
    ],
)
def test_apy(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = apy.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, hidden, expected",
    [
        (
            {"gender": "female", "age": 25},
            {"pregnant": True, "first_living_child": True},
            True,
        ),
        (
            {"gender": "female", "age": 25},
            {"recent_delivery": True, "second_child_is_girl": True},
            True,
        ),
        (
            {"gender": "male", "age": 25},
            {"pregnant": True, "first_living_child": True},
            False,
        ),
        (
            {"gender": "female", "age": 18},
            {"pregnant": True, "first_living_child": True},
            False,
        ),
        (
            {"gender": "female", "age": 25},
            {"pregnant": False, "first_living_child": True},
            False,
        ),
        (
            {"gender": "female", "age": 25},
            {"pregnant": True, "first_living_child": False},
            False,
        ),
    ],
)
def test_pmmvy(demographics, hidden, expected):
    economic = {"is_government_employee": False}
    p = _profile(demographics=demographics, economic=economic, hidden=hidden)
    ok, _ = pmmvy.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, sensitive, expected",
    [
        ({"age": 30}, {"bpl_household": True}, {"severe_disability": True}, True),
        ({"age": 30}, {"bpl_household": True}, {"multiple_disability": True}, True),
        ({"age": 17}, {"bpl_household": True}, {"severe_disability": True}, False),
        ({"age": 30}, {"bpl_household": False}, {"severe_disability": True}, False),
        ({"age": 30}, {"bpl_household": True}, {}, False),
    ],
)
def test_iggndps(demographics, economic, sensitive, expected):
    p = _profile(demographics=demographics, economic=economic, sensitive=sensitive)
    ok, _ = iggndps.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, hidden, expected",
    [
        (
            {"age": 30},
            {"occupation": "farmer", "holds_cultivable_land": True, "has_bank_account": True},
            {"land_acres": 1.5},
            True,
        ),
        (
            {"age": 41},
            {"occupation": "farmer", "holds_cultivable_land": True, "has_bank_account": True},
            {"land_acres": 1.5},
            False,
        ),
        (
            {"age": 30},
            {"occupation": "engineer", "has_bank_account": True},
            {},
            False,
        ),
        (
            {"age": 30},
            {"occupation": "farmer", "holds_cultivable_land": True, "has_bank_account": True},
            {"land_acres": 5.0},
            False,
        ),
        (
            {"age": 30},
            {
                "occupation": "farmer",
                "holds_cultivable_land": True,
                "has_bank_account": True,
                "income_tax_payer": True,
            },
            {"land_acres": 1.5},
            False,
        ),
    ],
)
def test_pmkmy(demographics, economic, hidden, expected):
    p = _profile(demographics=demographics, economic=economic, hidden=hidden)
    ok, _ = pmkmy.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, hidden, expected",
    [
        ({"age": 25}, {"planning_new_microenterprise": True, "project_size_lakh": 5}, True),
        ({"age": 25}, {"planning_new_microenterprise": True, "project_size_lakh": 8}, True),
        ({"age": 17}, {"planning_new_microenterprise": True, "project_size_lakh": 5}, False),
        ({"age": 25}, {"planning_new_microenterprise": False}, False),
        (
            {"age": 25},
            {
                "planning_new_microenterprise": True,
                "project_size_lakh": 15,
                "education_level": "below_8th",
            },
            False,
        ),
        (
            {"age": 25},
            {
                "planning_new_microenterprise": True,
                "project_size_lakh": 15,
                "education_level": "10th_pass",
            },
            True,
        ),
    ],
)
def test_pmegp(demographics, hidden, expected):
    p = _profile(demographics=demographics, hidden=hidden)
    ok, _ = pmegp.check(p)
    assert ok is expected
