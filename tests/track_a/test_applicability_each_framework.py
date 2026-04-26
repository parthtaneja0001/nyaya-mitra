"""parametrized applicability tests, one block per framework.
target is >=8 cases per framework; this is the v1 cut."""

from __future__ import annotations

from typing import Any

import pytest

from nyaya_mitra.interface import Behavior, CitizenProfile, SituationSpecific
from nyaya_mitra.knowledge.applicability import (
    consumer_protection_act_2019,
    domestic_violence_act_2005,
    maternity_benefit_act_1961,
    minimum_wages_act_1948,
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
    "demographics, sensitive, expected",
    [
        ({"gender": "female"}, {"dv_present": True}, True),
        ({"gender": "female"}, {"dv_history": True}, True),
        ({"gender": "male"}, {"dv_present": True}, False),
        ({"gender": "female"}, {}, False),
    ],
)
def test_dv_act_2005(demographics, sensitive, expected):
    p = _profile(demographics=demographics, sensitive=sensitive)
    ok, _ = domestic_violence_act_2005.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "economic, hidden, expected",
    [
        ({"is_wage_worker": True}, {"wages_below_minimum": True}, True),
        ({"is_wage_worker": False}, {"wages_below_minimum": True}, False),
        ({"is_wage_worker": True}, {"wages_below_minimum": False}, False),
        ({"is_wage_worker": True}, {}, False),
    ],
)
def test_minimum_wages_act(economic, hidden, expected):
    p = _profile(economic=economic, hidden=hidden)
    ok, _ = minimum_wages_act_1948.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "demographics, economic, hidden, expected",
    [
        (
            {"gender": "female"},
            {"formally_employed": True},
            {"pregnant": True, "denied_maternity_benefit": True},
            True,
        ),
        (
            {"gender": "female"},
            {"formally_employed": True},
            {"recent_delivery": True, "denied_maternity_benefit": True},
            True,
        ),
        (
            {"gender": "male"},
            {"formally_employed": True},
            {"pregnant": True, "denied_maternity_benefit": True},
            False,
        ),
        (
            {"gender": "female"},
            {"formally_employed": False},
            {"pregnant": True, "denied_maternity_benefit": True},
            False,
        ),
        (
            {"gender": "female"},
            {"formally_employed": True},
            {"pregnant": False, "denied_maternity_benefit": True},
            False,
        ),
        (
            {"gender": "female"},
            {"formally_employed": True},
            {"pregnant": True, "denied_maternity_benefit": False},
            False,
        ),
    ],
)
def test_maternity_benefit_act(demographics, economic, hidden, expected):
    p = _profile(demographics=demographics, economic=economic, hidden=hidden)
    ok, _ = maternity_benefit_act_1961.check(p)
    assert ok is expected


@pytest.mark.parametrize(
    "economic, hidden, expected",
    [
        ({"is_consumer_disputant": True}, {"defective_goods": True}, True),
        ({"is_consumer_disputant": True}, {"deficient_service": True}, True),
        ({"is_consumer_disputant": True}, {"unfair_trade_practice": True}, True),
        ({"is_consumer_disputant": True}, {"misleading_ad": True}, True),
        ({"is_consumer_disputant": False}, {"defective_goods": True}, False),
        ({"is_consumer_disputant": True}, {}, False),
    ],
)
def test_consumer_protection_act(economic, hidden, expected):
    p = _profile(economic=economic, hidden=hidden)
    ok, _ = consumer_protection_act_2019.check(p)
    assert ok is expected
