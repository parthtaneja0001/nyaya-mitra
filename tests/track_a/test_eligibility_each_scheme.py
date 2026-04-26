"""parametrized eligibility tests, one block per scheme.
target is >=8 cases per scheme; this is the v1 cut."""

from __future__ import annotations

from typing import Any

import pytest

from nyaya_mitra.interface import Behavior, CitizenProfile, SituationSpecific
from nyaya_mitra.knowledge.eligibility import (
    ayushman_bharat,
    mgnrega,
    pm_awas_grameen,
    pm_kisan,
    pmsby,
    pmuy,
)


def _profile(
    *,
    demographics: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    family: dict[str, Any] | None = None,
    sensitive: dict[str, Any] | None = None,
    hidden: dict[str, Any] | None = None,
) -> CitizenProfile:
    return CitizenProfile(
        seed=0,
        demographics=demographics or {},
        economic=economic or {},
        family=family or {},
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
    "economic, expected",
    [
        ({"occupation": "farmer", "holds_cultivable_land": True}, True),
        ({"occupation": "kisan", "holds_cultivable_land": True}, True),
        ({"occupation": "farmer", "holds_cultivable_land": False}, False),
        ({"occupation": "engineer", "holds_cultivable_land": True}, False),
        ({"occupation": "farmer", "holds_cultivable_land": True, "income_tax_payer": True}, False),
        ({"occupation": "farmer", "holds_cultivable_land": True, "is_professional": True}, False),
    ],
)
def test_pm_kisan(economic, expected):
    p = _profile(economic=economic)
    ok, _ = pm_kisan.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, expected",
    [
        (
            {"gender": "female", "age": 25},
            {"bpl_household": True, "existing_lpg_in_family": False},
            True,
        ),
        (
            {"gender": "female", "age": 50},
            {"bpl_household": True, "existing_lpg_in_family": False},
            True,
        ),
        (
            {"gender": "male", "age": 25},
            {"bpl_household": True, "existing_lpg_in_family": False},
            False,
        ),
        (
            {"gender": "female", "age": 17},
            {"bpl_household": True, "existing_lpg_in_family": False},
            False,
        ),
        (
            {"gender": "female", "age": 25},
            {"bpl_household": False, "existing_lpg_in_family": False},
            False,
        ),
        (
            {"gender": "female", "age": 25},
            {"bpl_household": True, "existing_lpg_in_family": True},
            False,
        ),
    ],
)
def test_pmuy(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = pmuy.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, expected",
    [
        ({"age": 25, "residence": "rural"}, {"willing_unskilled_work": True}, True),
        ({"age": 25, "residence": "rural"}, {}, True),
        ({"age": 17, "residence": "rural"}, {"willing_unskilled_work": True}, False),
        ({"age": 25, "residence": "urban"}, {"willing_unskilled_work": True}, False),
        ({"age": 25, "residence": "rural"}, {"willing_unskilled_work": False}, False),
    ],
)
def test_mgnrega(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = mgnrega.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, expected",
    [
        ({"residence": "rural"}, {"secc_listed": True, "kuccha_house": True}, True),
        ({"residence": "rural"}, {"secc_listed": True, "houseless": True}, True),
        ({"residence": "urban"}, {"secc_listed": True, "kuccha_house": True}, False),
        ({"residence": "rural"}, {"secc_listed": False, "kuccha_house": True}, False),
        (
            {"residence": "rural"},
            {"secc_listed": True, "owns_pucca_house": True, "kuccha_house": False},
            False,
        ),
        (
            {"residence": "rural"},
            {"secc_listed": True, "kuccha_house": False, "houseless": False},
            False,
        ),
    ],
)
def test_pm_awas_grameen(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = pm_awas_grameen.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "economic, expected",
    [
        ({"secc_listed": True}, True),
        ({"urban_occupational_category": True}, True),
        ({"secc_listed": True, "urban_occupational_category": True}, True),
        ({}, False),
        ({"secc_listed": False, "urban_occupational_category": False}, False),
    ],
)
def test_ayushman_bharat(economic, expected):
    p = _profile(economic=economic)
    ok, _ = ayushman_bharat.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, expected",
    [
        ({"age": 18}, {"has_bank_account": True}, True),
        ({"age": 70}, {"has_bank_account": True}, True),
        ({"age": 35}, {"has_bank_account": True}, True),
        ({"age": 17}, {"has_bank_account": True}, False),
        ({"age": 71}, {"has_bank_account": True}, False),
        ({"age": 35}, {"has_bank_account": False}, False),
    ],
)
def test_pmsby(demographics, economic, expected):
    p = _profile(demographics=demographics, economic=economic)
    ok, _ = pmsby.check(p)
    assert ok is expected
