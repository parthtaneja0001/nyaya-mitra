"""eligibility checker for pmegp. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    age = int(profile.demographics.get("age") or 0)
    facts = profile.situation_specific.hidden_facts or {}
    new_unit = bool(facts.get("planning_new_microenterprise"))
    education = facts.get(
        "education_level"
    )  # "below_8th" | "8th_pass" | "10th_pass" | "12th_pass" | "graduate"
    project_size_lakh = float(facts.get("project_size_lakh", 5))

    if age < 18:
        return False, ["under 18"]
    if not new_unit:
        return False, ["no new micro-enterprise proposal (existing units excluded)"]
    if project_size_lakh > 10 and education in (None, "below_8th"):
        # for projects above 10 lakh in manufacturing or 5 lakh in services need 8th-pass
        return False, ["project above 10 lakh requires minimum 8th-pass education"]
    return True, ["18+ with new micro-enterprise project meeting education threshold"]
