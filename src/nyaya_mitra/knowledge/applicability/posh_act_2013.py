"""applicability checker for the posh act, 2013. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    is_woman = profile.demographics.get("gender") == "female"
    employed = bool(
        profile.economic.get("formally_employed") or profile.economic.get("is_wage_worker")
    )
    facts = profile.situation_specific.hidden_facts or {}
    sexual_harassment = bool(facts.get("sexual_harassment_at_workplace"))

    if not is_woman:
        return False, ["posh act covers women employees"]
    if not employed:
        return False, ["not in employment"]
    if not sexual_harassment:
        return False, ["no sexual harassment at workplace reported"]
    return True, ["woman employee experiencing sexual harassment at workplace"]
