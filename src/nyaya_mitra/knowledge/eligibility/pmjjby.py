"""eligibility checker for pmjjby. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    age = int(profile.demographics.get("age") or 0)
    has_account = bool(profile.economic.get("has_bank_account"))

    if age < 18 or age > 50:
        return False, [f"age {age} outside enrolment band 18-50"]
    if not has_account:
        return False, ["no savings bank or post office account"]
    return True, ["18-50 with savings bank account"]
