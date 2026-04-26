"""applicability checker for the sc/st (prevention of atrocities) act, 1989. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    sensitive = profile.situation_specific.sensitive_facts or {}
    facts = profile.situation_specific.hidden_facts or {}
    caste = (sensitive.get("caste") or "").lower()
    is_sc_st = caste in ("scheduled_caste", "scheduled_tribe", "sc", "st")
    atrocity_present = bool(
        facts.get("caste_atrocity_present")
        or facts.get("caste_based_violence")
        or facts.get("caste_humiliation")
    )

    if not is_sc_st:
        return False, ["not from SC or ST community"]
    if not atrocity_present:
        return False, ["no caste-based atrocity reported"]
    return True, ["SC/ST member experiencing caste-based atrocity"]
