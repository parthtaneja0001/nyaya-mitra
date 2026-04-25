"""adversarial case generator tests: validator, diversity, generator wrapper, scoring."""

from __future__ import annotations

import json

from eval.baselines.llm_protocol import FakeChat
from nyaya_mitra.case_gen import (
    DiversityTracker,
    GeneratedCase,
    ProfileValidator,
    build_generator_advisor,
    score_generation,
)
from nyaya_mitra.case_gen.validator import ValidationResult


def _good_profile_dict(**overrides) -> dict:
    base = {
        "seed": 999,
        "demographics": {"gender": "female", "age": 30, "state": "punjab", "residence": "rural"},
        "economic": {
            "occupation": "farmer",
            "holds_cultivable_land": True,
            "monthly_income": 6000,
            "bpl_household": True,
        },
        "family": {"marital_status": "married", "children": 2},
        "situation_specific": {
            "presenting_issue": "small farm, low income, abuse at home",
            "hidden_facts": {"land_acres": 1.2},
            "sensitive_facts": {"dv_present": True, "dv_history": True},
        },
        "behavior": {
            "trust_level": "wary",
            "verbosity": "med",
            "language_preference": "hi",
            "literacy": "low",
            "initial_vague_query": "I need help",
        },
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
    return base


# ---------- validator: schema ----------


class TestValidatorSchema:
    def test_accepts_valid_profile_no_derive(self):
        v = ProfileValidator(derive_fn=None)
        result = v.validate(_good_profile_dict())
        assert result.valid is True

    def test_rejects_missing_required_field(self):
        v = ProfileValidator()
        bad = _good_profile_dict()
        del bad["situation_specific"]
        result = v.validate(bad)
        assert result.valid is False
        assert result.schema_error

    def test_rejects_bad_enum_value(self):
        v = ProfileValidator()
        bad = _good_profile_dict()
        bad["behavior"]["trust_level"] = "extremely_wary"  # not in literal
        result = v.validate(bad)
        assert result.valid is False


# ---------- validator: consistency ----------


class TestValidatorConsistency:
    def test_minor_married_inconsistency(self):
        v = ProfileValidator(derive_fn=None)
        bad = _good_profile_dict(demographics={"age": 14}, family={"marital_status": "married"})
        result = v.validate(bad)
        assert result.valid is False
        assert any("age 14" in s for s in result.inconsistencies or [])

    def test_software_engineer_low_income_inconsistency(self):
        v = ProfileValidator(derive_fn=None)
        bad = _good_profile_dict(
            economic={
                "occupation": "software engineer",
                "monthly_income": 2000,
                "bpl_household": False,
            }
        )
        result = v.validate(bad)
        assert result.valid is False
        assert any("monthly_income" in s for s in result.inconsistencies or [])

    def test_income_tax_payer_with_bpl_inconsistency(self):
        v = ProfileValidator(derive_fn=None)
        bad = _good_profile_dict(
            economic={
                "income_tax_payer": True,
                "bpl_household": True,
            }
        )
        result = v.validate(bad)
        assert result.valid is False

    def test_holds_land_urban_non_farmer_inconsistency(self):
        v = ProfileValidator(derive_fn=None)
        bad = _good_profile_dict(
            demographics={"residence": "urban"},
            economic={"holds_cultivable_land": True, "occupation": "tailor"},
        )
        result = v.validate(bad)
        assert result.valid is False


# ---------- validator: degenerate (no schemes/frameworks) ----------


class TestValidatorDegeneracy:
    def test_degenerate_when_derive_returns_empty(self):
        v = ProfileValidator(derive_fn=lambda p: ([], []))
        result = v.validate(_good_profile_dict())
        assert result.valid is False
        assert result.degenerate is True

    def test_not_degenerate_when_at_least_one_match(self):
        v = ProfileValidator(derive_fn=lambda p: (["pm_kisan"], []))
        result = v.validate(_good_profile_dict())
        assert result.valid is True


# ---------- diversity ----------


class TestDiversity:
    def test_empty_tracker_zero_similarity(self):
        t = DiversityTracker()
        assert t.max_similarity(_good_profile_dict()) == 0.0

    def test_identical_profile_max_similarity(self):
        t = DiversityTracker()
        p = _good_profile_dict()
        t.record(p)
        assert t.max_similarity(p) == 1.0

    def test_different_profile_lower_similarity(self):
        t = DiversityTracker()
        t.record(_good_profile_dict())
        new = _good_profile_dict(
            demographics={"state": "kerala", "residence": "urban"},
            economic={"occupation": "tailor", "holds_cultivable_land": False},
            situation_specific={
                "presenting_issue": "consumer dispute over electronics",
                "hidden_facts": {"defective_goods": True},
                "sensitive_facts": {},
            },
        )
        assert t.max_similarity(new) < 0.5

    def test_window_caps_history(self):
        t = DiversityTracker(window=3)
        for i in range(10):
            t.record(_good_profile_dict(seed=i))
        assert t.size == 3

    def test_penalty_is_non_positive(self):
        t = DiversityTracker(weight=0.5)
        t.record(_good_profile_dict())
        assert t.penalty(_good_profile_dict()) <= 0.0

    def test_window_validation(self):
        import pytest

        with pytest.raises(ValueError):
            DiversityTracker(window=0)


# ---------- generator wrapper ----------


class TestGenerator:
    def test_generator_emits_a_case_per_call(self):
        good = _good_profile_dict()
        chat = FakeChat([json.dumps(good)])
        validator = ProfileValidator(derive_fn=lambda p: (["pm_kisan"], []))
        tracker = DiversityTracker()
        gen = build_generator_advisor(chat, validator, tracker)

        case = gen()
        assert isinstance(case, GeneratedCase)
        assert case.parsed is not None
        assert case.validation.valid is True

    def test_generator_handles_garbage_response(self):
        chat = FakeChat(["this is not json at all sorry"])
        validator = ProfileValidator(derive_fn=lambda p: (["pm_kisan"], []))
        tracker = DiversityTracker()
        gen = build_generator_advisor(chat, validator, tracker)

        case = gen()
        assert case.parsed is None
        assert case.validation.valid is False

    def test_generator_finds_json_in_chatter(self):
        good = _good_profile_dict()
        wrapped = f"Here you go:\n```json\n{json.dumps(good)}\n```\nLet me know."
        chat = FakeChat([wrapped])
        validator = ProfileValidator()
        tracker = DiversityTracker()
        gen = build_generator_advisor(chat, validator, tracker)

        case = gen()
        assert case.parsed is not None


# ---------- scoring ----------


class TestScoring:
    def test_invalid_returns_minus_one(self):
        case = GeneratedCase(
            raw_text="x",
            parsed=None,
            parse_error="bad",
            validation=ValidationResult(valid=False, schema_error="bad"),
            similarity=0.0,
        )
        assert score_generation(case) == -1.0

    def test_valid_with_low_advisor_reward_yields_high_generator_reward(self):
        case = GeneratedCase(
            raw_text="",
            parsed={"x": 1},
            parse_error=None,
            validation=ValidationResult(valid=True),
            similarity=0.0,
            advisor_total_reward=-0.5,
        )
        # -(-0.5) - 0.5 * 0 = 0.5
        assert score_generation(case) == 0.5

    def test_valid_with_high_advisor_reward_yields_low_generator_reward(self):
        case = GeneratedCase(
            raw_text="",
            parsed={"x": 1},
            parse_error=None,
            validation=ValidationResult(valid=True),
            similarity=0.0,
            advisor_total_reward=0.9,
        )
        # -(0.9) - 0 = -0.9
        assert score_generation(case) == -0.9

    def test_diversity_penalty_applied(self):
        case = GeneratedCase(
            raw_text="",
            parsed={"x": 1},
            parse_error=None,
            validation=ValidationResult(valid=True),
            similarity=0.8,
            advisor_total_reward=0.0,
        )
        # -0 - 0.5 * 0.8 = -0.4
        assert score_generation(case, diversity_weight=0.5) == -0.4
