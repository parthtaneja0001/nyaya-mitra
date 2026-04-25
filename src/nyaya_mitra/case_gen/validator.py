"""validates a generated CitizenProfile.

three classes of failure:

1. schema invalid — pydantic rejects (missing fields, bad enum values).
2. degenerate — profile matches zero schemes AND zero frameworks. nothing to learn.
3. inconsistent — internal conflicts (e.g., age 14 + marital_status='married',
   occupation 'software engineer' + monthly_income 4000).

returns ValidationResult so the generator's reward fn can apply the appropriate
penalty (-1.0 for invalid, no advisor rollout wasted on degenerate profiles).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from nyaya_mitra.interface import CitizenProfile


@dataclass
class ValidationResult:
    valid: bool
    schema_error: str | None = None
    degenerate: bool = False
    inconsistencies: list[str] | None = None


_PROFESSIONAL_OCCUPATIONS = {
    "software engineer",
    "doctor",
    "lawyer",
    "chartered accountant",
    "architect",
    "consultant",
}


class ProfileValidator:
    """validates a profile against schema + KB-eligibility-derived ground truth +
    consistency rules. accepts a `derive_ground_truth` callable so the validator
    stays decoupled from track-a's loader.

    derive_fn signature: (profile) -> tuple[list[scheme_ids], list[framework_ids]].
    if None, the degenerate-check is skipped (still useful for schema + consistency).
    """

    def __init__(
        self,
        derive_fn: callable | None = None,
        *,
        require_at_least_one_match: bool = True,
    ) -> None:
        self._derive = derive_fn
        self._require_match = require_at_least_one_match

    def validate(self, raw: dict) -> ValidationResult:
        try:
            profile = CitizenProfile.model_validate(raw)
        except ValidationError as exc:
            return ValidationResult(valid=False, schema_error=str(exc))

        inconsistencies = list(self._consistency_violations(profile))
        if inconsistencies:
            return ValidationResult(valid=False, inconsistencies=inconsistencies)

        if self._derive is not None:
            try:
                schemes, frameworks = self._derive(profile)
            except Exception as exc:
                return ValidationResult(
                    valid=False,
                    schema_error=f"derive_ground_truth failed: {exc!r}",
                )
            if self._require_match and not (schemes or frameworks):
                return ValidationResult(valid=False, degenerate=True)

        return ValidationResult(valid=True)

    @staticmethod
    def _consistency_violations(profile: CitizenProfile) -> list[str]:
        out: list[str] = []
        age = int(profile.demographics.get("age") or 0)
        marital = (profile.family.get("marital_status") or "").lower()
        occupation = (profile.economic.get("occupation") or "").lower()
        income = profile.economic.get("monthly_income")
        residence = profile.demographics.get("residence")
        is_pro = bool(profile.economic.get("is_professional"))
        bpl = bool(profile.economic.get("bpl_household"))
        income_tax = bool(profile.economic.get("income_tax_payer"))

        if age and age < 18 and marital == "married":
            out.append(f"age {age} with marital_status 'married'")

        if any(p in occupation for p in _PROFESSIONAL_OCCUPATIONS):
            if income is not None and income < 5000:
                out.append(f"professional occupation '{occupation}' with monthly_income {income}")
            if bpl:
                out.append(f"professional occupation '{occupation}' marked BPL")

        if income_tax and bpl:
            out.append("income_tax_payer=True with bpl_household=True")

        if is_pro and bpl:
            out.append("is_professional=True with bpl_household=True")

        if residence not in {"rural", "urban", None}:
            out.append(f"residence must be 'rural' or 'urban', got {residence!r}")

        # cultivable land + urban is fishy unless explicitly farmer
        if (
            profile.economic.get("holds_cultivable_land")
            and residence == "urban"
            and "farmer" not in occupation
            and "kisan" not in occupation
        ):
            out.append("holds_cultivable_land + urban + non-farmer occupation")

        return out


__all__ = ["ProfileValidator", "ValidationResult"]
