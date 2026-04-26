"""eligibility checker for apy. pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    age = int(profile.demographics.get("age") or 0)
    has_account = bool(profile.economic.get("has_bank_account"))
    income_tax_payer = bool(profile.economic.get("income_tax_payer"))

    if age < 18 or age > 40:
        return False, [f"age {age} outside enrolment band 18-40"]
    if not has_account:
        return False, ["no savings bank or post office account"]
    if income_tax_payer:
        return False, ["excluded as income-tax payer"]
    return True, ["18-40, bank account, not an income-tax payer"]
