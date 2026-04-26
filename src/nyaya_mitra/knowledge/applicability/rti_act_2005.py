"""applicability checker for the rti act, 2005. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    facts = profile.situation_specific.hidden_facts or {}
    seeks_info = bool(facts.get("seeks_government_information") or facts.get("denied_govt_info"))

    if not seeks_info:
        return False, ["citizen is not seeking information from a public authority"]
    return True, ["citizen seeking information held by a public authority"]
