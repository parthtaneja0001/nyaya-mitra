"""deterministic fact extractor. NEVER an llm — citizen sim must not be tricked into volunteering ids.

emits two streams from each utterance:
- positive matches → revealed facts (added to elicited)
- negated matches  → negated facts (consumed by track b's contradiction gate)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nyaya_mitra.interface import CitizenProfile


_NEGATION = re.compile(r"\b(not|nahi|kabhi nahi|never|no\b)", re.IGNORECASE)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(farmer|kisan|kheti|farming)\b", re.IGNORECASE), "occupation_farmer"),
    (re.compile(r"\b(woman|women|girl|mahila|aurat|female)\b", re.IGNORECASE), "gender_female"),
    (re.compile(r"\b(bpl|below poverty)\b", re.IGNORECASE), "bpl_household"),
    (
        re.compile(r"\b(small holding|small plot|chhota khet|marginal)\b", re.IGNORECASE),
        "land_small",
    ),
    (
        re.compile(
            r"\b(domestic violence|husband hits|husband beats|ghar mein maar|pati maarta)\b",
            re.IGNORECASE,
        ),
        "dv_present",
    ),
    (re.compile(r"\bpunjab\b", re.IGNORECASE), "state_punjab"),
    (re.compile(r"\bbihar\b", re.IGNORECASE), "state_bihar"),
    (re.compile(r"\b(no lpg|no gas connection|chulha|wood stove)\b", re.IGNORECASE), "no_lpg"),
]


def _is_negated(utterance: str, match: re.Match[str]) -> bool:
    window_start = max(match.start() - 20, 0)
    window = utterance[window_start : match.end()]
    return bool(_NEGATION.search(window))


class FactExtractor:
    def extract(
        self,
        profile: CitizenProfile,
        utterance: str,
        prior_elicited: set[str],
    ) -> list[str]:
        out: list[str] = []
        for pattern, fact_id in _PATTERNS:
            if fact_id in prior_elicited:
                continue
            match = pattern.search(utterance)
            if not match:
                continue
            if _is_negated(utterance, match):
                continue
            out.append(fact_id)
        return out

    def extract_negations(
        self,
        profile: CitizenProfile,
        utterance: str,
    ) -> list[str]:
        """fact ids the utterance explicitly negates. consumed by track b's contradiction gate
        via info["negated_facts"]."""
        out: list[str] = []
        for pattern, fact_id in _PATTERNS:
            match = pattern.search(utterance)
            if not match:
                continue
            if not _is_negated(utterance, match):
                continue
            out.append(fact_id)
        return out
