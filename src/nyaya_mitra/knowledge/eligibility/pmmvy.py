"""eligibility checker for pmmvy. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    is_woman = profile.demographics.get("gender") == "female"
    age = int(profile.demographics.get("age") or 0)
    facts = profile.situation_specific.hidden_facts or {}
    pregnant_or_post = bool(facts.get("pregnant") or facts.get("recent_delivery"))
    is_first_or_girl_second = bool(
        facts.get("first_living_child") or facts.get("second_child_is_girl")
    )
    is_govt_employee = bool(profile.economic.get("is_government_employee"))

    if not is_woman:
        return False, ["pmmvy is for women"]
    if age < 19:
        return False, ["under 19"]
    if not pregnant_or_post:
        return False, ["not pregnant or recently delivered"]
    if not is_first_or_girl_second:
        return False, ["only for first live birth (or second if girl)"]
    if is_govt_employee:
        return False, ["government employees excluded (already get comparable benefits)"]
    return True, ["pregnant/postpartum woman 19+, first child or second girl, not govt employee"]
