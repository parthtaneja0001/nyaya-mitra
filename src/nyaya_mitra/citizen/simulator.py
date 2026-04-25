"""smart-canned citizen sim. NEVER an llm — deterministic by construction.

surfaces facts from the profile via templated utterances keyed by language and
literacy. wary citizens defer sensitive disclosure until 2 advisor turns of
trust-building. real frozen-llm version replaces this in track A.4."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nyaya_mitra.interface import AdvisorAction, Ask, Explain, Probe

if TYPE_CHECKING:
    from nyaya_mitra.env.episode_state import TurnRecord
    from nyaya_mitra.interface import CitizenProfile


def _has(d: dict | None, key: str) -> bool:
    return bool((d or {}).get(key))


def _farmer(p) -> bool:
    occ = (p.economic.get("occupation") or "").lower()
    return "farmer" in occ or "kisan" in occ


def _land_small(p) -> bool:
    acres = p.situation_specific.hidden_facts.get("land_acres") if p.situation_specific else None
    if acres is not None:
        return float(acres) < 2.0
    return bool(p.economic.get("holds_cultivable_land"))


def _female(p) -> bool:
    return p.demographics.get("gender") == "female"


def _adult(p) -> bool:
    return int(p.demographics.get("age") or 0) >= 18


def _adult_18_70(p) -> bool:
    age = int(p.demographics.get("age") or 0)
    return 18 <= age <= 70


def _willing_unskilled(p) -> bool:
    val = p.economic.get("willing_unskilled_work")
    return True if val is None else bool(val)


def _kuccha_or_houseless(p) -> bool:
    return bool(p.economic.get("kuccha_house") or p.economic.get("houseless"))


def _wages_below_min(p) -> bool:
    return _has(p.situation_specific.hidden_facts, "wages_below_minimum")


def _pregnant_or_post(p) -> bool:
    facts = p.situation_specific.hidden_facts or {}
    return bool(facts.get("pregnant") or facts.get("recent_delivery"))


def _denied_maternity(p) -> bool:
    return _has(p.situation_specific.hidden_facts, "denied_maternity_benefit")


def _consumer_grievance(p) -> bool:
    facts = p.situation_specific.hidden_facts or {}
    return bool(
        facts.get("defective_goods")
        or facts.get("deficient_service")
        or facts.get("unfair_trade_practice")
        or facts.get("misleading_ad")
    )


_REVEAL_FACTS: list[tuple[str, Callable, dict[str, str]]] = [
    (
        "occupation_farmer",
        _farmer,
        {
            "en": "I work as a farmer.",
            "hi": "मैं किसान हूं।",
            "hinglish": "Mai farmer hoon, kheti karte hain.",
        },
    ),
    (
        "gender_female",
        _female,
        {
            "en": "I am a woman.",
            "hi": "मैं महिला हूं।",
            "hinglish": "Mai mahila hoon.",
        },
    ),
    (
        "bpl_household",
        lambda p: _has(p.economic, "bpl_household"),
        {
            "en": "We are a BPL family.",
            "hi": "हम BPL परिवार हैं।",
            "hinglish": "Hum BPL family mein hain.",
        },
    ),
    (
        "residence_rural",
        lambda p: p.demographics.get("residence") == "rural",
        {
            "en": "I live in a village.",
            "hi": "मैं गाँव में रहती हूं।",
            "hinglish": "Gaon mein rehte hain hum.",
        },
    ),
    (
        "no_lpg",
        lambda p: not _has(p.economic, "existing_lpg_in_family"),
        {
            "en": "We don't have an LPG gas connection, I cook on a chulha.",
            "hi": "हमारे यहाँ गैस कनेक्शन नहीं है, चूल्हे पर खाना बनाते हैं।",
            "hinglish": "Gas connection nahi hai, chulha pe khana banate hain.",
        },
    ),
    (
        "land_small",
        _land_small,
        {
            "en": "I have only a small plot of land, marginal really.",
            "hi": "मेरे पास छोटा खेत है, बहुत कम जमीन है।",
            "hinglish": "Chhota plot hai bas, marginal landholding.",
        },
    ),
    (
        "adult",
        _adult,
        {
            "en": "Yes, I am an adult.",
            "hi": "हाँ, मैं वयस्क हूं।",
            "hinglish": "Haan, adult hoon.",
        },
    ),
    (
        "willing_unskilled_work",
        _willing_unskilled,
        {
            "en": "I'm willing to do unskilled manual work if there's pay.",
            "hi": "मेहनत-मजदूरी का काम कर सकती हूं अगर पैसा मिले।",
            "hinglish": "Manual work karne ko ready hoon agar paisa mile.",
        },
    ),
    (
        "kuccha_or_houseless",
        _kuccha_or_houseless,
        {
            "en": "Our house is just a kuccha hut.",
            "hi": "हमारा घर कच्चा है।",
            "hinglish": "Ghar kuccha hai humara.",
        },
    ),
    (
        "secc_listed",
        lambda p: _has(p.economic, "secc_listed"),
        {
            "en": "We are on the SECC 2011 deprivation list.",
            "hi": "हम SECC 2011 की सूची में हैं।",
            "hinglish": "SECC list mein hain hum.",
        },
    ),
    (
        "urban_occupational_category",
        lambda p: _has(p.economic, "urban_occupational_category"),
        {
            "en": "I'm in an urban occupational category that qualifies.",
            "hi": "मैं शहरी पात्र श्रेणी में आती हूं।",
            "hinglish": "Urban occupational category mein hoon.",
        },
    ),
    (
        "has_bank_account",
        lambda p: _has(p.economic, "has_bank_account"),
        {
            "en": "Yes, I have a savings bank account.",
            "hi": "हाँ, मेरे पास बैंक खाता है।",
            "hinglish": "Haan, bank account hai.",
        },
    ),
    (
        "adult_18_70",
        _adult_18_70,
        {
            "en": "I'm in the 18-70 age range.",
            "hi": "मेरी उम्र 18 से 70 के बीच है।",
            "hinglish": "Age 18-70 ke beech hai.",
        },
    ),
    (
        "formally_employed",
        lambda p: _has(p.economic, "formally_employed"),
        {
            "en": "I have a formal factory job.",
            "hi": "मेरी फैक्टरी में पक्की नौकरी है।",
            "hinglish": "Formal factory job hai meri.",
        },
    ),
    (
        "is_wage_worker",
        lambda p: _has(p.economic, "is_wage_worker"),
        {
            "en": "I work as a daily wage laborer.",
            "hi": "मैं दिहाड़ी मजदूर हूं।",
            "hinglish": "Daily wage worker hoon.",
        },
    ),
    (
        "wages_below_minimum",
        _wages_below_min,
        {
            "en": "My employer pays me below the minimum wage notified for our work.",
            "hi": "मालिक न्यूनतम मजदूरी से कम पैसा देता है।",
            "hinglish": "Minimum wage se kam paisa milta hai.",
        },
    ),
    (
        "pregnant_or_postpartum",
        _pregnant_or_post,
        {
            "en": "I am pregnant.",
            "hi": "मैं गर्भवती हूं।",
            "hinglish": "Pregnant hoon main.",
        },
    ),
    (
        "denied_maternity_benefit",
        _denied_maternity,
        {
            "en": "My employer is refusing to grant maternity leave.",
            "hi": "मालिक मातृत्व अवकाश देने से मना कर रहा है।",
            "hinglish": "Maternity leave dene se mana kar rahe hain.",
        },
    ),
    (
        "is_consumer",
        lambda p: _has(p.economic, "is_consumer_disputant"),
        {
            "en": "I had paid for a product or service.",
            "hi": "मैंने पैसे देकर सामान/सेवा खरीदी थी।",
            "hinglish": "Paise deke saaman/service li thi.",
        },
    ),
    (
        "consumer_grievance",
        _consumer_grievance,
        {
            "en": "The product was defective and the seller won't issue a refund.",
            "hi": "सामान खराब था और दुकानदार रिफंड नहीं दे रहा।",
            "hinglish": "Saaman defective tha, refund nahi de rahe.",
        },
    ),
]


_PROBE_DISCLOSURES: dict[str, tuple[Callable, dict[str, str]]] = {
    "dv": (
        lambda p: bool(
            p.situation_specific.sensitive_facts.get("dv_present")
            or p.situation_specific.sensitive_facts.get("dv_history")
        ),
        {
            "en": "Yes... my husband hits me sometimes.",
            "hi": "हाँ... पति कभी कभी मारते हैं मुझे।",
            "hinglish": "Haan... pati maarte hain kabhi kabhi.",
        },
    ),
    "caste": (
        lambda p: _has(p.situation_specific.sensitive_facts, "caste"),
        {
            "en": "I belong to a scheduled caste.",
            "hi": "मैं अनुसूचित जाति से हूं।",
            "hinglish": "SC se hoon main.",
        },
    ),
    "disability": (
        lambda p: _has(p.situation_specific.sensitive_facts, "disability"),
        {
            "en": "I have a disability.",
            "hi": "मुझे विकलांगता है।",
            "hinglish": "Disability hai mujhe.",
        },
    ),
    "immigration": (
        lambda p: _has(p.situation_specific.sensitive_facts, "immigration"),
        {
            "en": "I am a migrant.",
            "hi": "मैं प्रवासी हूं।",
            "hinglish": "Migrant hoon main.",
        },
    ),
}


_NEGATIVE = {
    "en": "No, that's not the case for me.",
    "hi": "नहीं, ऐसा कुछ नहीं है।",
    "hinglish": "Nahi, aisa kuch nahi hai.",
}

_DEFER = {
    "en": "I'd rather not discuss that right now.",
    "hi": "अभी इस बारे में बात नहीं करना चाहती।",
    "hinglish": "Abhi yeh baat nahi karna chahti.",
}

_EXHAUSTED = {
    "en": "I'm not sure what else to add.",
    "hi": "और क्या बताऊं, समझ नहीं आ रहा।",
    "hinglish": "Aur kya bataaun, samajh nahi aa raha.",
}

_EXPLAIN_LOW = {
    "en": "I understand a little. What should I do next?",
    "hi": "थोड़ा समझ आया। आगे क्या करना है?",
    "hinglish": "Thoda samajh aaya. Aage kya karna hai?",
}

_EXPLAIN_HIGH = {
    "en": "Understood, I'll follow up.",
    "hi": "समझ गई, मैं देख लूंगी।",
    "hinglish": "Samajh gaya, dekh leti hoon.",
}


def _lang_pick(profile, table: dict[str, str]) -> str:
    return table.get(profile.behavior.language_preference) or table.get("en") or ""


def _previously_revealed(transcript: list) -> set[str]:
    out: set[str] = set()
    for t in transcript:
        if getattr(t, "actor", None) == "citizen":
            for f in getattr(t, "revealed", None) or []:
                out.add(f)
    return out


class CitizenSimulator:
    def initial_utterance(self, profile: CitizenProfile) -> str:
        return profile.behavior.initial_vague_query

    def respond(
        self,
        profile: CitizenProfile,
        prior_transcript: list[TurnRecord],
        advisor_action: AdvisorAction,
    ) -> str:
        if isinstance(advisor_action, Probe):
            return self._respond_probe(profile, prior_transcript, advisor_action)
        if isinstance(advisor_action, Ask):
            return self._respond_ask(profile, prior_transcript)
        if isinstance(advisor_action, Explain):
            return self._respond_explain(profile)
        return "..."

    def _respond_probe(self, profile, transcript, probe: Probe) -> str:
        entry = _PROBE_DISCLOSURES.get(probe.sensitive_topic)
        if entry is None:
            return _lang_pick(profile, _NEGATIVE)
        predicate, replies = entry
        try:
            applies = predicate(profile)
        except Exception:
            applies = False
        if not applies:
            return _lang_pick(profile, _NEGATIVE)
        if profile.behavior.trust_level == "wary":
            advisor_turns = sum(1 for t in transcript if getattr(t, "actor", None) == "advisor")
            if advisor_turns < 2:
                return _lang_pick(profile, _DEFER)
        return _lang_pick(profile, replies)

    def _respond_ask(self, profile, transcript) -> str:
        already = _previously_revealed(transcript)
        for fact_id, predicate, replies in _REVEAL_FACTS:
            if fact_id in already:
                continue
            try:
                if not predicate(profile):
                    continue
            except Exception:
                continue
            return _lang_pick(profile, replies)
        return _lang_pick(profile, _EXHAUSTED)

    def _respond_explain(self, profile) -> str:
        if profile.behavior.literacy == "low":
            return _lang_pick(profile, _EXPLAIN_LOW)
        return _lang_pick(profile, _EXPLAIN_HIGH)
