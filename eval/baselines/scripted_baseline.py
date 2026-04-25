"""scripted advisor: probes for sensitive facts, asks targeted questions, then
finalizes with a plan derived from elicited_facts via fact->scheme/framework
heuristics.

NEVER an LLM. it's a sanity floor and a CI-friendly baseline. the heuristics
mirror the relevant_facts contract from kb_adapter so the plan it produces is
internally consistent with track-a's eligibility checkers.

intentionally simple: no probabilistic decisions, no learning. if the env
exposes a fact id that maps to a scheme/framework, the advisor includes that
scheme/framework in the plan with a small, KB-faithful set of documents and
procedural steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nyaya_mitra.interface import (
    ActionPlan,
    ApplicationPath,
    Ask,
    Finalize,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    Probe,
    SchemeRecommendation,
)

# fact id -> scheme/framework + sensitive_topic if the fact is sensitive.
# kept here so the baseline doesn't import from rewards/kb_adapter (which it
# could, but staying decoupled keeps the baseline a fair comparison target).
_FACT_TO_SCHEMES: dict[str, list[str]] = {
    "occupation_farmer": ["pm_kisan"],
    "land_small": ["pm_kisan"],
    "gender_female": [],
    "bpl_household": ["pmuy"],
    "no_lpg": ["pmuy"],
    "secc_listed": ["ayushman_bharat", "pm_awas_grameen"],
    "urban_occupational_category": ["ayushman_bharat"],
    "adult": ["mgnrega"],
    "residence_rural": ["mgnrega", "pm_awas_grameen"],
    "willing_unskilled_work": ["mgnrega"],
    "kuccha_or_houseless": ["pm_awas_grameen"],
    "adult_18_70": ["pmsby"],
    "has_bank_account": ["pmsby"],
}

_FACT_TO_FRAMEWORKS: dict[str, list[str]] = {
    "dv_present": ["domestic_violence_act_2005"],
    "is_consumer": ["consumer_protection_act_2019"],
    "consumer_grievance": ["consumer_protection_act_2019"],
    "formally_employed": [],
    "pregnant_or_postpartum": ["maternity_benefit_act_1961"],
    "denied_maternity_benefit": ["maternity_benefit_act_1961"],
    "is_wage_worker": ["minimum_wages_act_1948"],
    "wages_below_minimum": ["minimum_wages_act_1948"],
}

# minimum corroborating fact count to suggest a scheme/framework. avoids
# proposing a scheme on a single soft signal (e.g., "gender_female" alone
# wouldn't suggest pmuy without bpl_household + no_lpg).
_SCHEME_MIN_FACTS: dict[str, int] = {
    "pm_kisan": 1,
    "pmuy": 2,
    "ayushman_bharat": 1,
    "mgnrega": 2,
    "pm_awas_grameen": 2,
    "pmsby": 1,
}

_FRAMEWORK_MIN_FACTS: dict[str, int] = {
    "domestic_violence_act_2005": 1,
    "consumer_protection_act_2019": 1,
    "maternity_benefit_act_1961": 1,
    "minimum_wages_act_1948": 1,
}


def _required_facts_for_scheme(scheme_id: str) -> set[str]:
    return {f for f, ss in _FACT_TO_SCHEMES.items() if scheme_id in ss}


def _required_facts_for_framework(framework_id: str) -> set[str]:
    return {f for f, ss in _FACT_TO_FRAMEWORKS.items() if framework_id in ss}


# minimum-but-correct documents per scheme/framework, taken from the toy KB
# defaults track A ships. real impl reads from KB at construction time; we keep
# this static here so the baseline can run without a kb instance.
_DOCS_SCHEME: dict[str, list[str]] = {
    "pm_kisan": ["Aadhaar", "Bank account details", "Land record (khasra-khatauni)"],
    "pmuy": ["Aadhaar", "BPL ration card", "Bank passbook"],
    "ayushman_bharat": ["Aadhaar", "SECC household identifier"],
    "mgnrega": ["Aadhaar", "Bank account", "Job card application"],
    "pm_awas_grameen": ["Aadhaar", "SECC verification", "Bank account"],
    "pmsby": ["Aadhaar", "Bank account"],
}

_DOCS_FRAMEWORK: dict[str, list[str]] = {
    "domestic_violence_act_2005": [
        "Identity proof of the aggrieved person",
        "Address proof of the shared household",
        "Medical records or photographs of injuries (if any)",
        "Witness statements (if available)",
        "Marriage proof or proof of domestic relationship",
    ],
    "consumer_protection_act_2019": [
        "Bill or invoice",
        "Identity proof",
        "Written communication with seller (if any)",
    ],
    "maternity_benefit_act_1961": [
        "Identity proof",
        "Employment record",
        "Medical certificate confirming pregnancy",
    ],
    "minimum_wages_act_1948": [
        "Identity proof",
        "Employment record",
        "Wage slips for at least 3 months",
    ],
}

_FORUMS: dict[str, str] = {
    "domestic_violence_act_2005": "Magistrate of the First Class (Judicial Magistrate)",
    "consumer_protection_act_2019": "District Consumer Disputes Redressal Commission",
    "maternity_benefit_act_1961": "Inspector under the Maternity Benefit Act",
    "minimum_wages_act_1948": "Authority appointed under section 20 of the Minimum Wages Act",
}

_PROCEDURE: dict[str, list[str]] = {
    "domestic_violence_act_2005": [
        "Approach a Protection Officer or registered service-provider NGO",
        "File Form DV-1 (Domestic Incident Report) before the Magistrate",
        "Magistrate may grant protection, residence, monetary, custody, or compensation orders",
    ],
    "consumer_protection_act_2019": [
        "Send a written notice to the seller seeking redressal",
        "File a complaint with the District Consumer Commission",
        "Attend hearings; obtain order",
    ],
    "maternity_benefit_act_1961": [
        "Notify the employer in writing of the denial",
        "File a complaint with the Inspector",
        "Inspector adjudicates; appeal lies to the Labour Court",
    ],
    "minimum_wages_act_1948": [
        "Document the underpayment with wage slips or witness statements",
        "File a claim under section 20 within 12 months",
        "Authority may direct payment plus compensation",
    ],
}

# the SLSA/DLSA contact id we route to when the citizen state is unknown.
# real impl chooses based on profile.demographics.state — see prompted_baseline.
_DEFAULT_DLSA_CONTACT_ID = "nalsa_central"
_DEFAULT_DLSA_AUTHORITY = "NALSA"


_PROBE_PLAN: list[tuple[str, str, str]] = [
    # (sensitive_topic, en_question, hi_question)
    ("dv", "Is anyone at home hurting you?", "क्या घर में आपको कोई परेशानी या मार-पीट है?"),
    (
        "disability",
        "Is there any disability or long illness in the family?",
        "क्या परिवार में कोई दिव्यांगता या लंबी बीमारी है?",
    ),
    ("caste", "May I know your community for scheme matching?", "योजना मिलान के लिए जाति बताएंगे?"),
]


_ASK_PLAN: list[tuple[str, str]] = [
    (
        "Do you have a small landholding or do you do farm work?",
        "क्या आपकी छोटी जमीन है या खेती का काम है?",
    ),
    ("Do you have a bank account in your name?", "क्या आपके नाम पर बैंक खाता है?"),
    ("Do you live in a village or in a city?", "आप गांव में रहती हैं या शहर में?"),
    (
        "Are you facing any issue with wages, work conditions, or maternity leave?",
        "क्या मजदूरी, काम की स्थिति या मातृत्व अवकाश में कोई समस्या है?",
    ),
    (
        "Do you have a BPL card and an LPG connection at home?",
        "क्या आपके पास बीपीएल कार्ड है और घर में गैस कनेक्शन है?",
    ),
]


def _pick_language(observation_language: str) -> str:
    return observation_language if observation_language in {"en", "hi", "hinglish"} else "en"


def _select_question(asked: set[str], language: str) -> str:
    for en, hi in _ASK_PLAN:
        key = en
        if key in asked:
            continue
        return hi if language == "hi" else en
    return "Anything else you want to share?" if language != "hi" else "और कुछ बताना है?"


def _select_probe(probed: set[str], language: str) -> tuple[str, str] | None:
    for topic, en, hi in _PROBE_PLAN:
        if topic in probed:
            continue
        return topic, (hi if language == "hi" else en)
    return None


def _build_plan_from_facts(elicited: list[str]) -> ActionPlan:
    facts = set(elicited)

    scheme_score: dict[str, int] = {}
    for f in facts:
        for s in _FACT_TO_SCHEMES.get(f, []):
            scheme_score[s] = scheme_score.get(s, 0) + 1

    framework_score: dict[str, int] = {}
    for f in facts:
        for fw in _FACT_TO_FRAMEWORKS.get(f, []):
            framework_score[fw] = framework_score.get(fw, 0) + 1

    schemes: list[SchemeRecommendation] = []
    for sid, score in scheme_score.items():
        if score < _SCHEME_MIN_FACTS.get(sid, 1):
            continue
        rationale = sorted(_required_facts_for_scheme(sid) & facts)
        schemes.append(
            SchemeRecommendation(
                scheme_id=sid,
                rationale_facts=rationale[:3],
                required_documents=list(_DOCS_SCHEME.get(sid, ["Aadhaar"])),
                application_path=ApplicationPath(),
            )
        )

    legal_routes: list[LegalRouteRecommendation] = []
    for fid, score in framework_score.items():
        if score < _FRAMEWORK_MIN_FACTS.get(fid, 1):
            continue
        rationale = sorted(_required_facts_for_framework(fid) & facts)
        legal_routes.append(
            LegalRouteRecommendation(
                framework_id=fid,
                applicable_situation=", ".join(rationale[:3]) or "as described",
                forum=_FORUMS.get(fid, "appropriate forum"),
                procedural_steps=list(_PROCEDURE.get(fid, ["File a written complaint"])),
                free_legal_aid_contact=FreeLegalAidContact(
                    authority=_DEFAULT_DLSA_AUTHORITY,  # type: ignore[arg-type]
                    contact_id=_DEFAULT_DLSA_CONTACT_ID,
                ),
                required_documents=list(_DOCS_FRAMEWORK.get(fid, ["Identity proof"])),
            )
        )

    if not schemes and not legal_routes:
        # avoid format gate: include a no-op route to a free-legal-aid contact.
        # this keeps the plan structurally valid even when nothing was elicited.
        legal_routes.append(
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="general guidance",
                forum=_FORUMS["domestic_violence_act_2005"],
                procedural_steps=["Approach a Protection Officer for general counselling"],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority=_DEFAULT_DLSA_AUTHORITY,  # type: ignore[arg-type]
                    contact_id=_DEFAULT_DLSA_CONTACT_ID,
                ),
                required_documents=["Identity proof"],
            )
        )

    summary = (
        "we will help you apply for the schemes you qualify for and connect you to free legal aid."
    )
    return ActionPlan(
        schemes=schemes,
        legal_routes=legal_routes,
        most_important_next_step=(
            "contact your district legal services authority for free legal aid"
        ),
        plain_summary=PlainSummary(language="en", text=summary),
    )


def build_scripted_baseline(
    *,
    max_asks: int = 3,
    max_probes: int = 1,
    finalize_at: int | None = None,
) -> Callable[..., Any]:
    """produce a scripted advisor.

    max_asks: how many ASK turns before considering FINALIZE.
    max_probes: how many PROBE turns to attempt for sensitive disclosure.
    finalize_at: hard cap turn for FINALIZE; if None, finalizes once asks+probes
                 budget is exhausted.
    """

    def advisor(observation, state):
        language = _pick_language(observation.language)
        ti = state.turn_index

        if finalize_at is not None and ti >= finalize_at:
            return Finalize(plan=_build_plan_from_facts(state.elicited_facts))

        if ti < max_probes:
            choice = _select_probe(set(), language)
            if choice is not None:
                topic, q = choice
                return Probe(question=q, sensitive_topic=topic, language=language)  # type: ignore[arg-type]

        if ti < max_probes + max_asks:
            return Ask(question=_select_question(set(), language), language=language)  # type: ignore[arg-type]

        return Finalize(plan=_build_plan_from_facts(state.elicited_facts))

    return advisor


__all__ = ["build_scripted_baseline"]
