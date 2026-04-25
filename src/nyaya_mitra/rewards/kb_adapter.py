"""adapter that turns a duck-typed KB instance into something matching KnowledgeBase protocol.

the adapter only reads public attrs documented in the kb json schemas
(interface/kb_schemas.py). it does NOT import anything from nyaya_mitra.knowledge,
so it stays on track B's side of the seam.

usage (in shared bootstrap or scripts/, never inside src/nyaya_mitra/env/*):
    from nyaya_mitra.knowledge.loader import KnowledgeBase as RealKB
    from nyaya_mitra.rewards import make_env_reward_fn
    from nyaya_mitra.rewards.kb_adapter import DuckTypedKB

    real_kb = RealKB()
    adapter = DuckTypedKB(real_kb)
    env.reward_fn = make_env_reward_fn(adapter)
"""

from __future__ import annotations

from typing import Any, Protocol


class _RawKBLike(Protocol):
    schemes: dict[str, dict[str, Any]]
    frameworks: dict[str, dict[str, Any]]
    dlsa: dict[str, Any]


# fact-id mapping. each entry lists the fact ids the eligibility/applicability
# checker for that scheme/framework actually reads from the profile. used by
# fact_coverage to decide which facts the agent must elicit.
#
# fact ids must match what track A's extractor emits. when a checker reads a
# field that the extractor does not yet emit, the fact id is still listed here
# (so coverage tracks the contract correctly) — track A grows the extractor to
# match. coverage components tolerate missing facts gracefully (return 1.0 when
# the relevant set is empty), so this stays safe ahead of extractor growth.
_DEFAULT_RELEVANT_FACTS: dict[str, set[str]] = {
    # schemes
    "pm_kisan": {"occupation_farmer", "land_small"},
    "pmuy": {"gender_female", "bpl_household", "no_lpg"},
    "ayushman_bharat": {"secc_listed", "urban_occupational_category"},
    "mgnrega": {"adult", "residence_rural", "willing_unskilled_work"},
    "pm_awas_grameen": {"residence_rural", "secc_listed", "kuccha_or_houseless"},
    "pmsby": {"adult_18_70", "has_bank_account"},
    # frameworks
    "domestic_violence_act_2005": {"gender_female", "dv_present"},
    "consumer_protection_act_2019": {"is_consumer", "consumer_grievance"},
    "maternity_benefit_act_1961": {
        "gender_female",
        "formally_employed",
        "pregnant_or_postpartum",
        "denied_maternity_benefit",
    },
    "minimum_wages_act_1948": {"is_wage_worker", "wages_below_minimum"},
}


class DuckTypedKB:
    """wraps a kb-like object (anything with .schemes/.frameworks/.dlsa dicts)
    and exposes the read-only surface the rewards module needs."""

    def __init__(
        self,
        raw: _RawKBLike,
        relevant_facts: dict[str, set[str]] | None = None,
    ) -> None:
        self._raw = raw
        self._relevant = dict(relevant_facts or _DEFAULT_RELEVANT_FACTS)
        self._contact_index = self._build_contact_index(raw.dlsa)

    @staticmethod
    def _build_contact_index(dlsa: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        idx: dict[tuple[str, str], dict[str, Any]] = {}
        nalsa = dlsa.get("NALSA") or {}
        if "contact_id" in nalsa:
            idx[("NALSA", nalsa["contact_id"])] = nalsa
        for slsa in (dlsa.get("SLSAs") or {}).values():
            if "contact_id" in slsa:
                idx[("SLSA", slsa["contact_id"])] = slsa
        for d in (dlsa.get("DLSAs") or {}).values():
            if "contact_id" in d:
                idx[("DLSA", d["contact_id"])] = d
        return idx

    def has_scheme(self, scheme_id: str) -> bool:
        return scheme_id in self._raw.schemes

    def has_framework(self, framework_id: str) -> bool:
        return framework_id in self._raw.frameworks

    def has_contact(self, authority: str, contact_id: str) -> bool:
        return (authority, contact_id) in self._contact_index

    def documents_for_scheme(self, scheme_id: str) -> list[str]:
        s = self._raw.schemes.get(scheme_id) or {}
        return list(s.get("required_documents") or [])

    def documents_for_framework(self, framework_id: str) -> list[str]:
        f = self._raw.frameworks.get(framework_id) or {}
        return list(f.get("required_documents") or [])

    def procedural_steps_for_framework(self, framework_id: str) -> list[str]:
        f = self._raw.frameworks.get(framework_id) or {}
        return list(f.get("procedural_steps") or [])

    def forum_for_framework(self, framework_id: str) -> str | None:
        f = self._raw.frameworks.get(framework_id) or {}
        v = f.get("forum")
        return v if isinstance(v, str) else None

    def legal_aid_authority_for_framework(self, framework_id: str) -> str | None:
        f = self._raw.frameworks.get(framework_id) or {}
        v = f.get("legal_aid_authority")
        return v if isinstance(v, str) else None

    def relevant_facts_for_scheme(self, scheme_id: str) -> set[str]:
        return set(self._relevant.get(scheme_id, set()))

    def relevant_facts_for_framework(self, framework_id: str) -> set[str]:
        return set(self._relevant.get(framework_id, set()))
