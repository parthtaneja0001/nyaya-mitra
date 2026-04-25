"""gates: format_validity, hallucination, contradiction, sim_leak passthrough.

each gate test demonstrates: (a) it triggers on the right plan, (b) it doesn't
trigger on a valid plan, (c) the sneaky case where the plan is valid but the
gate still fires (PLAN B.4: "sneaky" tests required).
"""

from __future__ import annotations

from nyaya_mitra.rewards.gates import (
    check_contradiction,
    check_format,
    check_hallucination,
    leaked_turn_indices,
)

from .conftest import (
    Ask,
    FakeKB,
    Probe,
    make_citizen_turn,
    make_ctx,
    make_legal_rec,
    make_plan,
    make_profile,
    make_scheme_rec,
    make_turn,
)

# ---------- format_validity ----------


class TestFormatGate:
    def test_valid_plan_passes(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
        )
        assert check_format(ctx) is False

    def test_empty_plan_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert check_format(ctx) is True

    def test_blank_next_step_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")], next_step="   "),
            kb=kb_basic,
        )
        assert check_format(ctx) is True

    def test_blank_summary_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")], summary_text=""),
            kb=kb_basic,
        )
        assert check_format(ctx) is True

    def test_explicit_format_violation_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan")]),
            kb=kb_basic,
            info={"max_turns": 20, "format_violation": True},
        )
        assert check_format(ctx) is True


# ---------- hallucination ----------


class TestHallucinationGate:
    def test_known_ids_pass(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[make_scheme_rec("pm_kisan")],
                legal_routes=[make_legal_rec("domestic_violence_act_2005", contact_id="dlsa_test")],
            ),
            kb=kb_basic,
        )
        assert check_hallucination(ctx) is False

    def test_unknown_scheme_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("ghost_scheme")]),
            kb=kb_basic,
        )
        assert check_hallucination(ctx) is True

    def test_unknown_framework_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(legal_routes=[make_legal_rec("ghost_act")]),
            kb=kb_basic,
        )
        assert check_hallucination(ctx) is True

    def test_unknown_contact_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                legal_routes=[
                    make_legal_rec("domestic_violence_act_2005", contact_id="ghost_contact"),
                ]
            ),
            kb=kb_basic,
        )
        assert check_hallucination(ctx) is True


# ---------- contradiction ----------


class TestContradictionGate:
    def test_grounded_facts_pass(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[make_scheme_rec("pm_kisan", rationale_facts=["occupation_farmer"])],
            ),
            elicited_facts=["occupation_farmer"],
            kb=kb_basic,
        )
        assert check_contradiction(ctx) is False

    def test_ungrounded_fact_fires(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[make_scheme_rec("pm_kisan", rationale_facts=["occupation_farmer"])],
            ),
            elicited_facts=[],
            kb=kb_basic,
        )
        assert check_contradiction(ctx) is True

    def test_negated_fact_fires_sneaky(self, kb_basic: FakeKB):
        # plan is valid AND fact is in elicited_facts BUT it was negated by the citizen.
        # this is the "sneaky" case from PLAN B.4.
        citizen_turn = make_citizen_turn(
            0,
            revealed=[],
            info={"negated_facts": ["occupation_farmer"]},
        )
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(
                schemes=[make_scheme_rec("pm_kisan", rationale_facts=["occupation_farmer"])],
            ),
            transcript=[citizen_turn],
            elicited_facts=["occupation_farmer"],
            kb=kb_basic,
        )
        assert check_contradiction(ctx) is True

    def test_no_rationale_facts_passes(self, kb_basic: FakeKB):
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(schemes=[make_scheme_rec("pm_kisan", rationale_facts=[])]),
            kb=kb_basic,
        )
        assert check_contradiction(ctx) is False


# ---------- sim_leak passthrough ----------


class TestSimLeak:
    def test_no_turns_no_leaks(self, kb_basic: FakeKB):
        ctx = make_ctx(profile=make_profile(), plan=make_plan(), kb=kb_basic)
        assert leaked_turn_indices(ctx) == set()

    def test_single_leak_recorded(self, kb_basic: FakeKB):
        ask = Ask(question="how is family?", language="en")
        leak_turn = make_turn(0, ask, info={"sim_leak": True})
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[leak_turn],
            kb=kb_basic,
        )
        assert leaked_turn_indices(ctx) == {0}

    def test_probe_no_leak(self, kb_basic: FakeKB):
        probe = Probe(question="dv?", sensitive_topic="dv", language="en")
        turn = make_turn(0, probe, info={"sim_leak": False})
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[turn],
            kb=kb_basic,
        )
        assert leaked_turn_indices(ctx) == set()

    def test_multiple_leaks(self, kb_basic: FakeKB):
        ask = Ask(question="...", language="en")
        ctx = make_ctx(
            profile=make_profile(),
            plan=make_plan(),
            transcript=[
                make_turn(0, ask, info={"sim_leak": True}),
                make_turn(1, ask, info={"sim_leak": False}),
                make_turn(2, ask, info={"sim_leak": True}),
            ],
            kb=kb_basic,
        )
        assert leaked_turn_indices(ctx) == {0, 2}
