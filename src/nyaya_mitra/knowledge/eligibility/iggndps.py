"""eligibility checker for iggndps. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    age = int(profile.demographics.get("age") or 0)
    bpl = bool(profile.economic.get("bpl_household"))
    sensitive = profile.situation_specific.sensitive_facts or {}
    severe_disability = bool(
        sensitive.get("severe_disability") or sensitive.get("multiple_disability")
    )

    if age < 18:
        return False, ["under 18"]
    if not bpl:
        return False, ["not in BPL household"]
    if not severe_disability:
        return False, ["no certified severe or multiple disability"]
    return True, ["BPL adult with severe / multiple disability certified"]
