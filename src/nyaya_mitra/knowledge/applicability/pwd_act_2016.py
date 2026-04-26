"""applicability checker for the rights of persons with disabilities act, 2016. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    sensitive = profile.situation_specific.sensitive_facts or {}
    facts = profile.situation_specific.hidden_facts or {}
    has_disability = bool(
        sensitive.get("disability_present")
        or sensitive.get("severe_disability")
        or sensitive.get("multiple_disability")
        or sensitive.get("disability")
    )
    discrimination = bool(facts.get("disability_discrimination_present"))

    if not has_disability:
        return False, ["no disability declared"]
    if not discrimination:
        return False, ["no discrimination or denial reported"]
    return True, ["person with disability facing discrimination or denial of accommodation"]
