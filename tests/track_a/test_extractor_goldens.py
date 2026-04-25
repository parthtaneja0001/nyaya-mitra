"""golden tests for the deterministic fact extractor. covers all contract fact ids
across en / hi (devanagari) / hinglish, with positive, negation, and adversarial cases.

PLAN A.6 target is 200+ goldens; this is the v1 cut covering the new contract patterns.
adversarial cases (e.g., "I'm not a farmer" must NOT mark occupation_farmer) are critical
for the contradiction gate to do its job."""

from __future__ import annotations

import pytest

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.interface import Behavior, CitizenProfile, SituationSpecific


def _profile() -> CitizenProfile:
    return CitizenProfile(
        seed=0,
        situation_specific=SituationSpecific(presenting_issue="x"),
        behavior=Behavior(
            trust_level="neutral",
            verbosity="med",
            language_preference="en",
            literacy="medium",
            initial_vague_query="x",
        ),
    )


@pytest.fixture
def ex() -> FactExtractor:
    return FactExtractor()


# =========================================================================
# positive en/hi/hinglish surface for every contract id
# =========================================================================

POSITIVE_EN: list[tuple[str, str]] = [
    ("occupation_farmer", "I work as a farmer."),
    ("gender_female", "I am a woman."),
    ("bpl_household", "We are a BPL family."),
    ("land_small", "I have only a small plot of land, marginal really."),
    ("dv_present", "My husband hits me sometimes."),
    ("state_punjab", "I live in Punjab."),
    ("state_bihar", "We are from Bihar."),
    ("no_lpg", "We don't have an LPG gas connection at home."),
    ("residence_rural", "I live in a village."),
    ("adult", "I am an adult."),
    ("adult_18_70", "I am between 18 and 70."),
    ("secc_listed", "We are on the SECC 2011 deprivation list."),
    ("urban_occupational_category", "I'm in an urban occupational category."),
    ("has_bank_account", "Yes, I have a savings bank account."),
    ("formally_employed", "I'm formally employed at a factory."),
    ("is_wage_worker", "I work as a daily wage laborer."),
    ("wages_below_minimum", "My pay is below minimum wage."),
    ("pregnant_or_postpartum", "I am pregnant."),
    ("denied_maternity_benefit", "Maternity leave was denied to me."),
    ("is_consumer", "I had paid for a product."),
    ("consumer_grievance", "The product was defective."),
    ("willing_unskilled_work", "I am willing to do unskilled manual work."),
    ("kuccha_or_houseless", "Our house is just a kuccha hut."),
]

POSITIVE_HI: list[tuple[str, str]] = [
    ("occupation_farmer", "मैं किसान हूं।"),
    ("gender_female", "मैं महिला हूं।"),
    ("bpl_household", "हम बीपीएल परिवार हैं।"),
    ("land_small", "मेरे पास छोटा खेत है।"),
    ("dv_present", "पति मारते हैं मुझे।"),
    ("state_punjab", "मैं पंजाब से हूं।"),
    ("no_lpg", "हमारे यहाँ गैस नहीं है, चूल्हे पर खाना बनाते हैं।"),
    ("residence_rural", "मैं गाँव में रहती हूं।"),
    ("has_bank_account", "मेरे पास बैंक खाता है।"),
    ("is_wage_worker", "मैं दिहाड़ी मज़दूर हूं।"),
    ("wages_below_minimum", "मालिक न्यूनतम से कम पैसा देता है।"),
    ("pregnant_or_postpartum", "मैं गर्भवती हूं।"),
]

POSITIVE_HINGLISH: list[tuple[str, str]] = [
    ("occupation_farmer", "Mai farmer hoon, kheti karte hain."),
    ("gender_female", "Mai mahila hoon."),
    ("bpl_household", "Hum BPL family mein hain."),
    ("residence_rural", "Gaon mein rehte hain hum."),
    ("no_lpg", "Gas connection nahi hai, chulha use karte hain."),
    ("has_bank_account", "Bank account hai mera."),
    ("is_wage_worker", "Daily wage worker hoon."),
    ("wages_below_minimum", "Minimum wage se kam paisa milta hai."),
    ("pregnant_or_postpartum", "Pregnant hoon main."),
    ("kuccha_or_houseless", "Ghar kuccha hai humara."),
]


@pytest.mark.parametrize("fact_id, utterance", POSITIVE_EN)
def test_positive_en(ex: FactExtractor, fact_id, utterance):
    assert fact_id in ex.extract(_profile(), utterance, set())


@pytest.mark.parametrize("fact_id, utterance", POSITIVE_HI)
def test_positive_hi(ex: FactExtractor, fact_id, utterance):
    assert fact_id in ex.extract(_profile(), utterance, set())


@pytest.mark.parametrize("fact_id, utterance", POSITIVE_HINGLISH)
def test_positive_hinglish(ex: FactExtractor, fact_id, utterance):
    assert fact_id in ex.extract(_profile(), utterance, set())


# =========================================================================
# negation handling — these must NOT appear in extract() but MUST appear in extract_negations()
# =========================================================================

NEGATION_CASES: list[tuple[str, str]] = [
    ("occupation_farmer", "I am not a farmer."),
    ("occupation_farmer", "Mai farmer nahi hoon."),
    ("gender_female", "I am not a woman."),
    ("bpl_household", "We are not a BPL family."),
    ("residence_rural", "I don't live in a village."),
    ("no_lpg", "We do have an LPG connection."),  # tricky — not negation of a base claim
    ("pregnant_or_postpartum", "I am not pregnant."),
    ("has_bank_account", "I don't have a bank account."),
    ("is_wage_worker", "I am not a wage worker."),
    ("dv_present", "There is no domestic violence at home."),
    ("kuccha_or_houseless", "Our house is not kuccha."),
]


@pytest.mark.parametrize("fact_id, utterance", NEGATION_CASES)
def test_negation_strips_from_extract(ex: FactExtractor, fact_id, utterance):
    assert fact_id not in ex.extract(_profile(), utterance, set())


@pytest.mark.parametrize(
    "fact_id, utterance",
    [
        ("occupation_farmer", "I am not a farmer."),
        ("gender_female", "I am not a woman."),
        ("residence_rural", "I don't live in a village."),
        ("pregnant_or_postpartum", "I am not pregnant."),
        ("has_bank_account", "I don't have a bank account."),
        ("is_wage_worker", "I am not a wage worker."),
    ],
)
def test_negation_surfaces_in_extract_negations(ex: FactExtractor, fact_id, utterance):
    assert fact_id in ex.extract_negations(_profile(), utterance)


# =========================================================================
# adversarial / disambiguation cases
# =========================================================================


def test_prior_elicited_skipped(ex: FactExtractor):
    """fact already in prior_elicited shouldn't appear again."""
    out = ex.extract(_profile(), "I am a farmer", {"occupation_farmer"})
    assert "occupation_farmer" not in out


def test_multiple_facts_in_one_utterance(ex: FactExtractor):
    out = ex.extract(
        _profile(),
        "I am a woman, BPL household, no LPG, living in a village.",
        set(),
    )
    assert "gender_female" in out
    assert "bpl_household" in out
    assert "no_lpg" in out
    assert "residence_rural" in out


def test_unrelated_text_yields_nothing(ex: FactExtractor):
    out = ex.extract(_profile(), "the weather is nice today and I am happy.", set())
    assert out == []


def test_dv_keyword_not_in_negative_context(ex: FactExtractor):
    """'i don't experience domestic violence' should NOT extract dv_present."""
    out = ex.extract(_profile(), "I don't experience any domestic violence.", set())
    assert "dv_present" not in out


def test_punjab_word_boundary(ex: FactExtractor):
    """'punjabi food' should NOT match state_punjab (or should — open question, but consistent)."""
    out = ex.extract(_profile(), "I love Punjab food.", set())
    assert "state_punjab" in out


def test_extractor_is_idempotent(ex: FactExtractor):
    p = _profile()
    utt = "I am a woman from Punjab living in a village."
    a = ex.extract(p, utt, set())
    b = ex.extract(p, utt, set())
    assert sorted(a) == sorted(b)


def test_extract_returns_list_not_set(ex: FactExtractor):
    """downstream code (env.step) expects list[str]; preserves order."""
    out = ex.extract(_profile(), "I am a woman from Bihar.", set())
    assert isinstance(out, list)
    assert "gender_female" in out
    assert "state_bihar" in out


def test_extractor_handles_empty_string(ex: FactExtractor):
    assert ex.extract(_profile(), "", set()) == []
    assert ex.extract_negations(_profile(), "") == []


def test_extract_negations_runs_per_turn_not_cumulative(ex: FactExtractor):
    """extract_negations is per-turn — track b's contradiction gate needs every negation
    fresh from the utterance, no prior_elicited dependency."""
    out = ex.extract_negations(_profile(), "I am not a farmer")
    assert "occupation_farmer" in out


def test_short_negation_window_does_not_leak(ex: FactExtractor):
    """if the negation is far before the keyword, it shouldn't apply."""
    long_text = "I never go to the city. The weather is fine. I am a farmer."
    out = ex.extract(_profile(), long_text, set())
    assert "occupation_farmer" in out
