"""eligibility checker for pmkmy (pm kisan maandhan). pure python, no i/o."""

from __future__ import annotations

from nyaya_mitra.interface import CitizenProfile


def check(profile: CitizenProfile) -> tuple[bool, list[str]]:
    age = int(profile.demographics.get("age") or 0)
    occ = (profile.economic.get("occupation") or "").lower()
    holds_land = bool(profile.economic.get("holds_cultivable_land"))
    has_account = bool(profile.economic.get("has_bank_account"))
    is_taxpayer = bool(profile.economic.get("income_tax_payer"))
    is_pro = bool(profile.economic.get("is_professional"))
    facts = profile.situation_specific.hidden_facts or {}
    land_acres = float(facts.get("land_acres", 99))
    other_pension = bool(profile.economic.get("has_contributory_pension"))

    if age < 18 or age > 40:
        return False, [f"age {age} outside enrolment band 18-40"]
    if not ("farmer" in occ or "kisan" in occ):
        return False, ["not a farmer"]
    if not holds_land:
        return False, ["does not hold cultivable land"]
    if land_acres > 2.0:  # ~5 acres = ~2 hectares; cap is 2 hectares
        return False, [f"landholding {land_acres} acres above small/marginal threshold"]
    if not has_account:
        return False, ["no savings bank account"]
    if is_taxpayer or is_pro:
        return False, ["excluded as income-tax payer / professional"]
    if other_pension:
        return False, ["already in another contributory pension scheme"]
    return True, ["18-40 small/marginal farmer with bank account, no other pension"]
