"""read-only protocol the rewards module needs from the knowledge base.

track A provides an adapter that satisfies this protocol when wiring
compute_reward into the env. track B never imports nyaya_mitra.knowledge
directly — the contract test enforces it.
"""

from __future__ import annotations

from typing import Protocol


class KnowledgeBase(Protocol):
    """minimal lookup surface needed by reward components and gates."""

    def has_scheme(self, scheme_id: str) -> bool: ...

    def has_framework(self, framework_id: str) -> bool: ...

    def has_contact(self, authority: str, contact_id: str) -> bool: ...

    def documents_for_scheme(self, scheme_id: str) -> list[str]:
        """canonical required-documents list for a scheme. empty if unknown."""
        ...

    def documents_for_framework(self, framework_id: str) -> list[str]:
        """canonical required-documents list for a framework. empty if unknown."""
        ...

    def procedural_steps_for_framework(self, framework_id: str) -> list[str]:
        """canonical ordered procedural steps for a framework. empty if unknown."""
        ...

    def forum_for_framework(self, framework_id: str) -> str | None:
        """canonical forum for a framework. None if unknown."""
        ...

    def legal_aid_authority_for_framework(self, framework_id: str) -> str | None:
        """canonical legal aid authority. None if unknown."""
        ...

    def relevant_facts_for_scheme(self, scheme_id: str) -> set[str]:
        """fact IDs the scheme's eligibility checker references. empty if unknown."""
        ...

    def relevant_facts_for_framework(self, framework_id: str) -> set[str]:
        """fact IDs the framework's applicability checker references. empty if unknown."""
        ...
