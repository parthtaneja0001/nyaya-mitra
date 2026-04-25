"""deterministic fact extractor. NEVER an llm — citizen sim must not be tricked into volunteering ids.

emits two streams from each utterance:
- positive matches → revealed facts (added to elicited)
- negated matches  → negated facts (consumed by track b's contradiction gate)

fact ids match the contract in src/nyaya_mitra/rewards/kb_adapter._DEFAULT_RELEVANT_FACTS
and reward_design.md. patterns cover en + hi (devanagari) + hinglish.

absence-polarity facts (no_lpg, kuccha_or_houseless, wages_below_minimum,
denied_maternity_benefit) are facts where the match phrase itself expresses
absence/lack/denial — for these, the negation post-check is skipped (otherwise
"no LPG" would be self-stripping)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nyaya_mitra.interface import CitizenProfile


_NEGATION = re.compile(
    r"(\bnot\b|\bnever\b|\bisn'?t\b|\bdon'?t\b|\bdoesn'?t\b|\bwon'?t\b|\bno\b|\bnahi\b|\bkabhi nahi\b|नहीं)",
    re.IGNORECASE,
)


# (pattern, fact_id, absence_polarity)
_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    (
        re.compile(r"(?:\b(farmer|kisan|kheti|farming)\b|किसान)", re.IGNORECASE),
        "occupation_farmer",
        False,
    ),
    (
        re.compile(
            r"(?:\b(woman|women|girl|mahila|aurat|female)\b|महिला)",
            re.IGNORECASE,
        ),
        "gender_female",
        False,
    ),
    (
        re.compile(r"(?:\b(bpl|below poverty)\b|बीपीएल)", re.IGNORECASE),
        "bpl_household",
        False,
    ),
    (
        re.compile(
            r"(?:\b(small holding|small plot|chhota khet|chhota plot|marginal landholding)\b|छोटा खेत|कम जमीन)",
            re.IGNORECASE,
        ),
        "land_small",
        False,
    ),
    (
        re.compile(
            r"(?:\b(domestic violence|husband hits|husband beats|ghar mein maar|pati maarta|pati maarte)\b|घरेलू हिंसा|पति मारते|पति मारता)",
            re.IGNORECASE,
        ),
        "dv_present",
        False,
    ),
    (re.compile(r"(?:\bpunjab\b|पंजाब)", re.IGNORECASE), "state_punjab", False),
    (re.compile(r"(?:\bbihar\b|बिहार)", re.IGNORECASE), "state_bihar", False),
    (
        re.compile(
            r"(?:\b(no lpg|no gas connection|chulha|wood stove)\b"
            r"|don'?t have (an? )?lpg"
            r"|don'?t have (an? )?gas( connection)?"
            r"|गैस नहीं है"
            r"|gas connection nahi"
            r"|gas nahi)",
            re.IGNORECASE,
        ),
        "no_lpg",
        True,
    ),
    (
        re.compile(
            r"(?:\b(village|gaon|gaav|rural|in a village)\b|गाँव|गांव|देहात)",
            re.IGNORECASE,
        ),
        "residence_rural",
        False,
    ),
    (
        re.compile(
            r"(?:\b(adult|vyask|grown[- ]up)\b|वयस्क|बालिग)",
            re.IGNORECASE,
        ),
        "adult",
        False,
    ),
    (
        re.compile(
            r"(?:\b(18 to 70|18-70|18 se 70|between 18 and 70)\b|18 से 70)",
            re.IGNORECASE,
        ),
        "adult_18_70",
        False,
    ),
    (
        re.compile(
            r"(?:\b(secc|secc[- ]?2011|deprivation list|deprivation roll)\b|SECC list|सूची में हैं)",
            re.IGNORECASE,
        ),
        "secc_listed",
        False,
    ),
    (
        re.compile(
            r"(?:\b(urban occupational|urban occupation)\b|urban category|शहरी श्रेणी)",
            re.IGNORECASE,
        ),
        "urban_occupational_category",
        False,
    ),
    (
        re.compile(
            r"(?:\b(bank account|savings account|savings bank|post office account)\b|बैंक खाता|account hai)",
            re.IGNORECASE,
        ),
        "has_bank_account",
        False,
    ),
    (
        re.compile(
            r"(?:\b(formal(ly)? employed|formal job|formal factory job|salaried)\b|पक्की नौकरी|formal nokri)",
            re.IGNORECASE,
        ),
        "formally_employed",
        False,
    ),
    (
        re.compile(
            r"(?:\b(daily wage|wage labor(er)?|wage worker|dihaadi|dihari)\b|दिहाड़ी|मज़दूर|मजदूर)",
            re.IGNORECASE,
        ),
        "is_wage_worker",
        False,
    ),
    (
        re.compile(
            r"(?:\b(below minimum wage|under minimum wage|less than minimum)\b"
            r"|न्यूनतम से कम"
            r"|minimum wage se kam"
            r"|kam paisa)",
            re.IGNORECASE,
        ),
        "wages_below_minimum",
        True,
    ),
    (
        re.compile(
            r"(?:\b(pregnant|expecting|delivered recently|recent delivery)\b|गर्भवती|हाल ही में जन्म)",
            re.IGNORECASE,
        ),
        "pregnant_or_postpartum",
        False,
    ),
    (
        re.compile(
            r"(?:\b(denied maternity|maternity (leave|benefit)? (was )?(denied|refused))\b"
            r"|refusing (maternity )?leave"
            r"|मातृत्व अवकाश नहीं"
            r"|maternity leave nahi)",
            re.IGNORECASE,
        ),
        "denied_maternity_benefit",
        True,
    ),
    (
        re.compile(
            r"(?:\b(paid for|purchased|bought online|paid the seller)\b|पैसे देकर|paise deke)",
            re.IGNORECASE,
        ),
        "is_consumer",
        False,
    ),
    (
        re.compile(
            r"(?:\b(defective|deficient service|deficient)\b"
            r"|खराब निकला"
            r"|खराब था"
            r"|defective tha)",
            re.IGNORECASE,
        ),
        "consumer_grievance",
        False,
    ),
    (
        re.compile(
            r"(?:\b(willing to (do )?(unskilled )?(manual )?work|ready to work)\b"
            r"|manual work karne"
            r"|मजदूरी का काम)",
            re.IGNORECASE,
        ),
        "willing_unskilled_work",
        False,
    ),
    (
        re.compile(
            r"(?:\b(kuccha|kuccha hut|kuccha house|houseless|homeless)\b|कच्चा घर|कच्ची|बेघर)",
            re.IGNORECASE,
        ),
        "kuccha_or_houseless",
        False,
    ),
]


def _negation_window(utterance: str, match: re.Match[str]) -> str:
    """captures up to 25 chars before and 15 chars after the match for negation scan,
    truncated at the nearest sentence boundary so 'X. I am not Y' doesn't flag X."""
    start = max(match.start() - 25, 0)
    pre = utterance[start : match.start()]
    last_break = max(pre.rfind(". "), pre.rfind(", "), pre.rfind("! "), pre.rfind("? "))
    if last_break != -1:
        pre = pre[last_break + 2 :]
    end = min(match.end() + 15, len(utterance))
    post = utterance[match.end() : end]
    breaks = [post.find(s) for s in (". ", ", ", "! ", "? ") if post.find(s) != -1]
    if breaks:
        post = post[: min(breaks)]
    return pre + utterance[match.start() : match.end()] + post


def _is_negated(utterance: str, match: re.Match[str]) -> bool:
    window = _negation_window(utterance, match)
    return bool(_NEGATION.search(window))


class FactExtractor:
    def extract(
        self,
        profile: CitizenProfile,
        utterance: str,
        prior_elicited: set[str],
    ) -> list[str]:
        out: list[str] = []
        for pattern, fact_id, absence in _PATTERNS:
            if fact_id in prior_elicited:
                continue
            match = pattern.search(utterance)
            if not match:
                continue
            if not absence and _is_negated(utterance, match):
                continue
            out.append(fact_id)
        return out

    def extract_negations(
        self,
        profile: CitizenProfile,
        utterance: str,
    ) -> list[str]:
        """fact ids the utterance explicitly negates. consumed by track b's contradiction gate
        via info["negated_facts"]. absence-polarity facts are never reported as negated
        (the match itself is the fact)."""
        out: list[str] = []
        for pattern, fact_id, absence in _PATTERNS:
            if absence:
                continue
            match = pattern.search(utterance)
            if not match:
                continue
            if not _is_negated(utterance, match):
                continue
            out.append(fact_id)
        return out
