"""golden tests for each deterministic reward component.

each component gets at least 5 hand-crafted (profile, plan, expected_score)
golden checks per PLAN B.4. organized by component for readability.
"""

from __future__ import annotations

import pytest

from nyaya_mitra.rewards.components import (
    compute_dignity_judge,
    compute_document_accuracy,
    compute_fact_coverage,
    compute_harm_penalty,
    compute_integration_bonus,
    compute_legal_precision,
    compute_legal_recall,
    compute_procedural_correctness,
    compute_scheme_precision,
    compute_scheme_recall,
    compute_sensitivity_correctness,
    compute_turn_efficiency,
)
from nyaya_mitra.rewards.components.dignity_judge import set_judge

from .conftest import (
    Ask,
    Explain,
    FakeKB,
    Probe,
    make_ctx,
    make_legal_rec,
    make_plan,
    make_profile,
    make_scheme_rec,
    make_turn,
)

# ---------- scheme_precision ----------


class TestSchemePrecision:
    def test_empty_plan_returns_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 1.0

    def test_all_correct_returns_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan", "pmuy"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan"), make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 1.0

    def test_half_correct_returns_half(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan"), make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 0.5

    def test_all_wrong_returns_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=[]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 0.0

    def test_no_eligible_no_suggestions_returns_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=[]),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 1.0

    def test_suggesting_subset_full_precision(self, kb_basic: FakeKB):
        # only one of two eligible suggested -> precision 1.0 (recall is what suffers)
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan", "pmuy"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
        )
        assert compute_scheme_precision(ctx) == 1.0


# ---------- scheme_recall ----------


class TestSchemeRecall:
    def test_no_eligible_returns_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=[]),
            plan=make_plan(schemes=[make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        assert compute_scheme_recall(ctx) == 1.0

    def test_all_found(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan", "pmuy"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan"), make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        assert compute_scheme_recall(ctx) == 1.0

    def test_half_found(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan", "pmuy"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
        )
        assert compute_scheme_recall(ctx) == 0.5

    def test_none_found(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_scheme_recall(ctx) == 0.0

    def test_extra_irrelevant_does_not_help(self, kb_basic: FakeKB):
        # suggesting an irrelevant scheme doesn't increase recall
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan", "pmuy"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan"), make_scheme_rec("foo")]),
            kb=kb_basic,
        )
        assert compute_scheme_recall(ctx) == 0.5


# ---------- legal_precision / legal_recall ----------


class TestLegalPrecisionRecall:
    def test_precision_empty_plan_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(applicable_frameworks=["domestic_violence_act_2005"]),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_legal_precision(ctx) == 1.0

    def test_precision_all_wrong(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(applicable_frameworks=[]),
            plan=make_plan(legal_routes=[make_legal_rec("domestic_violence_act_2005")]),
            kb=kb_basic,
        )
        assert compute_legal_precision(ctx) == 0.0

    def test_recall_no_applicable_one(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_legal_recall(ctx) == 1.0

    def test_recall_partial(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(applicable_frameworks=["domestic_violence_act_2005", "rti"]),
            plan=make_plan(legal_routes=[make_legal_rec("domestic_violence_act_2005")]),
            kb=kb_basic,
        )
        assert compute_legal_recall(ctx) == 0.5

    def test_recall_full(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(applicable_frameworks=["domestic_violence_act_2005"]),
            plan=make_plan(legal_routes=[make_legal_rec("domestic_violence_act_2005")]),
            kb=kb_basic,
        )
        assert compute_legal_recall(ctx) == 1.0


# ---------- document_accuracy ----------


class TestDocumentAccuracy:
    def test_empty_plan_one(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_document_accuracy(ctx) == 1.0

    def test_perfect_match(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[
                    make_scheme_rec(
                        "pm_kisan",
                        documents=["Aadhaar", "Bank account", "Land record"],
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_document_accuracy(ctx) == 1.0

    def test_normalization_handles_punctuation(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[
                    make_scheme_rec(
                        "pm_kisan",
                        documents=["aadhaar.", "bank-account", "land record!!!"],
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_document_accuracy(ctx) == 1.0

    def test_partial_overlap_returns_jaccard(self, kb_basic: FakeKB):
        # suggested ∩ canonical / suggested ∪ canonical
        # suggested = {aadhaar, foo}, canonical = {aadhaar, bank account, land record}
        # |∩|=1, |∪|=4 -> 0.25
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan", documents=["Aadhaar", "Foo"])]),
            kb=kb_basic,
        )
        assert compute_document_accuracy(ctx) == pytest.approx(0.25)

    def test_unknown_scheme_skipped(self, kb_basic: FakeKB):
        # unknown scheme is excluded from the average; here only valid scheme remains
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[
                    make_scheme_rec(
                        "pm_kisan", documents=["Aadhaar", "Bank account", "Land record"]
                    ),
                    make_scheme_rec("does_not_exist", documents=["x"]),
                ]
            ),
            kb=kb_basic,
        )
        assert compute_document_accuracy(ctx) == 1.0

    def test_empty_required_zero_score(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[make_scheme_rec("pm_kisan", documents=[])],
            ),
            kb=kb_basic,
        )
        assert compute_document_accuracy(ctx) == 0.0


# ---------- procedural_correctness ----------


class TestProceduralCorrectness:
    def test_no_legal_routes_one(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_procedural_correctness(ctx) == 1.0

    def test_perfect_route(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec(
                        "domestic_violence_act_2005",
                        forum="Magistrate of the First Class",
                        procedural_steps=[
                            "approach protection officer",
                            "file dv-1 form",
                            "magistrate grants protection order",
                        ],
                        authority="DLSA",
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_procedural_correctness(ctx) == 1.0

    def test_wrong_forum(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec(
                        "domestic_violence_act_2005",
                        forum="District Forum",
                        procedural_steps=[
                            "approach protection officer",
                            "file dv-1 form",
                            "magistrate grants protection order",
                        ],
                        authority="DLSA",
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_procedural_correctness(ctx) < 1.0

    def test_wrong_authority(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec(
                        "domestic_violence_act_2005",
                        forum="Magistrate of the First Class",
                        procedural_steps=[
                            "approach protection officer",
                            "file dv-1 form",
                            "magistrate grants protection order",
                        ],
                        authority="NALSA",
                        contact_id="nalsa_central",
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_procedural_correctness(ctx) < 1.0

    def test_unknown_framework_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(legal_routes=[make_legal_rec("does_not_exist")]),
            kb=kb_basic,
        )
        assert compute_procedural_correctness(ctx) == 0.0

    def test_step_order_matters(self, kb_basic: FakeKB):
        good = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec(
                        "domestic_violence_act_2005",
                        forum="Magistrate of the First Class",
                        procedural_steps=[
                            "approach protection officer",
                            "file dv-1 form",
                            "magistrate grants protection order",
                        ],
                    )
                ]
            ),
            kb=kb_basic,
        )
        scrambled = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec(
                        "domestic_violence_act_2005",
                        forum="Magistrate of the First Class",
                        procedural_steps=[
                            "magistrate grants protection order",
                            "file dv-1 form",
                            "approach protection officer",
                        ],
                    )
                ]
            ),
            kb=kb_basic,
        )
        assert compute_procedural_correctness(good) > compute_procedural_correctness(scrambled)


# ---------- fact_coverage ----------


class TestFactCoverage:
    def test_no_relevant_facts_one(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_fact_coverage(ctx) == 1.0

    def test_full_coverage(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            elicited_facts=["occupation_farmer", "land_small"],
            kb=kb_basic,
        )
        assert compute_fact_coverage(ctx) == 1.0

    def test_partial_coverage(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            elicited_facts=["occupation_farmer"],
            kb=kb_basic,
        )
        assert compute_fact_coverage(ctx) == 0.5

    def test_irrelevant_facts_dont_help(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            elicited_facts=["unrelated_fact_1", "unrelated_fact_2"],
            kb=kb_basic,
        )
        assert compute_fact_coverage(ctx) == 0.0

    def test_integrated_profile_aggregates_facts(self, kb_basic: FakeKB):
        # scheme + framework together: relevant = {occupation_farmer, land_small,
        # gender_female, dv_present}. elicited 2/4 -> 0.5.
        ctx = make_ctx(
            profile=make_profile(
                eligible_schemes=["pm_kisan"],
                applicable_frameworks=["domestic_violence_act_2005"],
            ),
            plan=make_plan(),
            elicited_facts=["occupation_farmer", "gender_female"],
            kb=kb_basic,
        )
        assert compute_fact_coverage(ctx) == 0.5


# ---------- sensitivity_correctness ----------


class TestSensitivityCorrectness:
    def test_no_sensitive_facts_one(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_sensitivity_correctness(ctx) == 1.0

    def test_correct_probe_full_credit(self, kb_basic: FakeKB):
        probe = Probe(question="is there violence at home?", sensitive_topic="dv", language="en")
        ctx = make_ctx(
            profile=make_profile(sensitive_facts={"dv_present": True}),
            plan=make_plan(),
            transcript=[make_turn(0, probe)],
            elicited_facts=["dv_present"],
            kb=kb_basic,
        )
        assert compute_sensitivity_correctness(ctx) == 1.0

    def test_wrong_probe_topic_zero(self, kb_basic: FakeKB):
        probe = Probe(question="caste?", sensitive_topic="caste", language="en")
        ctx = make_ctx(
            profile=make_profile(sensitive_facts={"dv_present": True}),
            plan=make_plan(),
            transcript=[make_turn(0, probe)],
            elicited_facts=["dv_present"],
            kb=kb_basic,
        )
        assert compute_sensitivity_correctness(ctx) == 0.0

    def test_no_probe_at_all_zero(self, kb_basic: FakeKB):
        ask = Ask(question="how are you?", language="en")
        ctx = make_ctx(
            profile=make_profile(sensitive_facts={"dv_present": True}),
            plan=make_plan(),
            transcript=[make_turn(0, ask)],
            elicited_facts=["dv_present"],
            kb=kb_basic,
        )
        assert compute_sensitivity_correctness(ctx) == 0.0

    def test_two_sensitive_one_correct_half(self, kb_basic: FakeKB):
        probe_dv = Probe(question="dv?", sensitive_topic="dv", language="en")
        ctx = make_ctx(
            profile=make_profile(sensitive_facts={"dv_present": True, "caste_sc": True}),
            plan=make_plan(),
            transcript=[make_turn(0, probe_dv)],
            elicited_facts=["dv_present", "caste_sc"],
            kb=kb_basic,
        )
        assert compute_sensitivity_correctness(ctx) == 0.5


# ---------- turn_efficiency ----------


class TestTurnEfficiency:
    def test_zero_max_returns_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic, info={"max_turns": 0})
        assert compute_turn_efficiency(ctx, {"fact_coverage": 1.0}) == 0.0

    def test_below_coverage_threshold_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_turn_efficiency(ctx, {"fact_coverage": 0.4}) == 0.0

    def test_no_turns_full_efficiency(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_turn_efficiency(ctx, {"fact_coverage": 1.0}) == 1.0

    def test_half_turns(self, kb_basic: FakeKB):
        ask = Ask(question="...", language="en")
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[make_turn(i, ask) for i in range(10)],
            kb=kb_basic,
        )
        # used=10, max=20 -> 0.5
        assert compute_turn_efficiency(ctx, {"fact_coverage": 1.0}) == 0.5

    def test_overrun_clamps_to_zero(self, kb_basic: FakeKB):
        ask = Ask(question="...", language="en")
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[make_turn(i, ask) for i in range(25)],
            kb=kb_basic,
        )
        assert compute_turn_efficiency(ctx, {"fact_coverage": 1.0}) == 0.0


# ---------- integration_bonus ----------


class TestIntegrationBonus:
    def test_no_integrated_profile_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(),
            kb=kb_basic,
        )
        assert compute_integration_bonus(ctx, {}) == 0.0

    def test_integrated_full_credit(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(
                eligible_schemes=["pm_kisan"],
                applicable_frameworks=["domestic_violence_act_2005"],
            ),
            plan=make_plan(),
            kb=kb_basic,
        )
        score = compute_integration_bonus(
            ctx,
            {
                "scheme_precision": 0.9,
                "scheme_recall": 0.8,
                "legal_precision": 0.7,
                "legal_recall": 0.6,
            },
        )
        assert score == 1.0

    def test_integrated_threshold_miss_zero(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(
                eligible_schemes=["pm_kisan"],
                applicable_frameworks=["domestic_violence_act_2005"],
            ),
            plan=make_plan(),
            kb=kb_basic,
        )
        score = compute_integration_bonus(
            ctx,
            {
                "scheme_precision": 1.0,
                "scheme_recall": 1.0,
                "legal_precision": 0.4,
                "legal_recall": 1.0,
            },
        )
        assert score == 0.0


# ---------- harm_penalty ----------


class TestHarmPenalty:
    def test_no_harm(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
        )
        assert compute_harm_penalty(ctx) == 0.0

    def test_one_wrong_scheme_minus_005(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=["pm_kisan"]),
            plan=make_plan(schemes=[make_scheme_rec("pmuy")]),
            kb=kb_basic,
        )
        assert compute_harm_penalty(ctx) == pytest.approx(-0.05)

    def test_unknown_scheme_does_not_double_count(self, kb_basic: FakeKB):
        # hallucinated id is gated separately; harm_penalty skips it
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=[]),
            plan=make_plan(schemes=[make_scheme_rec("does_not_exist")]),
            kb=kb_basic,
        )
        assert compute_harm_penalty(ctx) == 0.0

    def test_caps_at_minus_020(self, kb_basic: FakeKB):
        # 6 wrong but valid schemes -> capped at 4*-0.05 = -0.20
        kb = FakeKB(
            schemes={f"s{i}": {} for i in range(10)},
        )
        ctx = make_ctx(
            profile=make_profile(eligible_schemes=[]),
            plan=make_plan(
                schemes=[make_scheme_rec(f"s{i}") for i in range(6)],
            ),
            kb=kb,
        )
        assert compute_harm_penalty(ctx) == pytest.approx(-0.20)


# ---------- dignity_judge ----------


class TestDignityJudge:
    def teardown_method(self) -> None:
        set_judge(None)

    def test_judge_callable_used_when_set(self, kb_basic: FakeKB):
        set_judge(lambda _ctx: 0.8)
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_dignity_judge(ctx) == pytest.approx(0.8)

    def test_judge_clamps_to_unit_interval(self, kb_basic: FakeKB):
        set_judge(lambda _ctx: 5.0)
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert compute_dignity_judge(ctx) == 1.0
        set_judge(lambda _ctx: -2.0)
        assert compute_dignity_judge(ctx) == 0.0

    def test_judge_exception_falls_back_to_heuristic(self, kb_basic: FakeKB):
        def bad_judge(_ctx):
            raise RuntimeError("oops")

        set_judge(bad_judge)
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        score = compute_dignity_judge(ctx)
        assert 0.0 <= score <= 1.0

    def test_heuristic_penalizes_jargon_low_literacy(self, kb_basic: FakeKB):
        explain = Explain(
            content="invoke section 125 of crpc and seek injunction",
            target_literacy="low",
            language="en",
        )
        ctx = make_ctx(
            profile=make_profile(literacy="low"),
            plan=make_plan(),
            transcript=[make_turn(0, explain)],
            kb=kb_basic,
        )
        score = compute_dignity_judge(ctx)
        assert score < 1.0

    def test_heuristic_penalizes_blame(self, kb_basic: FakeKB):
        ask = Ask(question="why didn't you save the receipts?", language="en")
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[make_turn(0, ask)],
            kb=kb_basic,
        )
        score = compute_dignity_judge(ctx)
        assert score < 1.0

    def test_heuristic_clean_full(self, kb_basic: FakeKB):
        explain = Explain(
            content="we will fill simple form together", target_literacy="low", language="en"
        )
        ctx = make_ctx(
            profile=make_profile(literacy="low"),
            plan=make_plan(),
            transcript=[make_turn(0, explain)],
            kb=kb_basic,
        )
        assert compute_dignity_judge(ctx) == 1.0
