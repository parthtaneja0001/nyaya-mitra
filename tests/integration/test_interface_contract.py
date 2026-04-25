"""
contract test for the seam between tracks A and B.

if this passes, both tracks can integrate cleanly.
if it fails, the interface has drifted — STOP and re-sync.

most checks are pytest.skip stubs that get filled in as tracks land.
the cross-track-imports check is live from day one — it's the mechanical
guardrail that prevents silent coupling.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "nyaya_mitra"

TRACK_A_PACKAGES = ("env", "citizen", "knowledge", "profile")
TRACK_B_PACKAGES = ("rewards", "case_gen", "advisor")
TRACK_B_TOPLEVEL_DIRS = ("training", "eval")
ALLOWED_CROSS_IMPORTS = {"nyaya_mitra.env.client"}


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module)
    return out


def _is_in_packages(import_name: str, packages: tuple[str, ...]) -> bool:
    for pkg in packages:
        prefix = f"nyaya_mitra.{pkg}"
        if import_name == prefix or import_name.startswith(prefix + "."):
            return True
    return False


def _scan_dir(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_cross_track_imports():
    """
    track A code may not import track B internals; track B may not import track A internals.
    only allowed cross-import is nyaya_mitra.env.client (the public http client).
    both sides freely import nyaya_mitra.interface.
    """
    violations: list[str] = []

    for pkg in TRACK_A_PACKAGES:
        pkg_root = SRC_ROOT / pkg
        if not pkg_root.exists():
            continue
        for py in _scan_dir(pkg_root):
            for imp in _module_imports(py):
                if imp in ALLOWED_CROSS_IMPORTS:
                    continue
                if _is_in_packages(imp, TRACK_B_PACKAGES):
                    violations.append(f"[A] {py.relative_to(REPO_ROOT)} imports [B] {imp}")

    for pkg in TRACK_B_PACKAGES:
        pkg_root = SRC_ROOT / pkg
        if not pkg_root.exists():
            continue
        for py in _scan_dir(pkg_root):
            for imp in _module_imports(py):
                if imp in ALLOWED_CROSS_IMPORTS:
                    continue
                if _is_in_packages(imp, TRACK_A_PACKAGES):
                    violations.append(f"[B] {py.relative_to(REPO_ROOT)} imports [A] {imp}")

    for top in TRACK_B_TOPLEVEL_DIRS:
        top_root = REPO_ROOT / top
        if not top_root.exists():
            continue
        for py in _scan_dir(top_root):
            for imp in _module_imports(py):
                if imp in ALLOWED_CROSS_IMPORTS:
                    continue
                if _is_in_packages(imp, TRACK_A_PACKAGES):
                    violations.append(f"[B] {py.relative_to(REPO_ROOT)} imports [A] {imp}")

    assert not violations, "cross-track imports detected:\n  " + "\n  ".join(violations)


def test_interface_imports_cleanly():
    """interface is the only shared package and must import without side effects."""
    import nyaya_mitra.interface as iface

    assert hasattr(iface, "AdvisorAction")
    assert hasattr(iface, "ActionPlan")
    assert hasattr(iface, "CitizenObservation")
    assert hasattr(iface, "CitizenProfile")
    assert hasattr(iface, "ALL_KEYS")
    assert hasattr(iface, "SCHEME_SCHEMA")


def test_reward_keys_are_unique_strings():
    from nyaya_mitra.interface import ALL_KEYS

    assert len(ALL_KEYS) == len(set(ALL_KEYS))
    for k in ALL_KEYS:
        assert isinstance(k, str) and k


def test_action_plan_requires_legal_aid_contact():
    """
    structural anti-liability rule: every legal route must carry a free_legal_aid_contact.
    pydantic should reject construction without it.
    """
    from pydantic import ValidationError

    from nyaya_mitra.interface import LegalRouteRecommendation

    with pytest.raises(ValidationError):
        LegalRouteRecommendation(
            framework_id="domestic_violence_act_2005",
            applicable_situation="x",
            forum="magistrate",
            procedural_steps=["a"],
            required_documents=["b"],
        )


def test_action_plan_round_trips_through_json():
    from nyaya_mitra.interface import (
        ActionPlan,
        FreeLegalAidContact,
        LegalRouteRecommendation,
        PlainSummary,
    )

    plan = ActionPlan(
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="x",
                forum="magistrate",
                procedural_steps=["a"],
                free_legal_aid_contact=FreeLegalAidContact(authority="DLSA", contact_id="dl_001"),
                required_documents=["b"],
            )
        ],
        most_important_next_step="contact dlsa",
        plain_summary=PlainSummary(language="hi", text="..."),
    )
    raw = plan.model_dump_json()
    revived = ActionPlan.model_validate_json(raw)
    assert revived == plan


def test_full_episode_with_stub_advisor():
    """end-to-end: env reset → step(Ask) → step(Finalize) with track B's reward fn wired in.
    asserts every key in interface.ALL_KEYS is present in the terminal info breakdown."""
    from nyaya_mitra.citizen.extractor import FactExtractor
    from nyaya_mitra.citizen.simulator import CitizenSimulator
    from nyaya_mitra.env.environment import NyayaMitraEnv
    from nyaya_mitra.interface import (
        ALL_KEYS,
        ActionPlan,
        ApplicationPath,
        Ask,
        Finalize,
        FreeLegalAidContact,
        LegalRouteRecommendation,
        PlainSummary,
        SchemeRecommendation,
    )
    from nyaya_mitra.knowledge.loader import KnowledgeBase
    from nyaya_mitra.rewards import make_env_reward_fn
    from nyaya_mitra.rewards.kb_adapter import DuckTypedKB

    kb = KnowledgeBase()
    reward_fn = make_env_reward_fn(DuckTypedKB(kb))
    env = NyayaMitraEnv(kb, CitizenSimulator(), FactExtractor(), reward_fn=reward_fn)
    obs = env.reset(seed=0)
    assert obs.turn == 0

    res = env.step(Ask(question="tell me more about your situation", language="en"))
    assert not res.done

    plan = ActionPlan(
        schemes=[
            SchemeRecommendation(
                scheme_id="pm_kisan",
                rationale_facts=["occupation_farmer"],
                required_documents=["Aadhaar"],
                application_path=ApplicationPath(),
            )
        ],
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="dv at home",
                forum="magistrate",
                procedural_steps=["file dv-1"],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority="DLSA", contact_id="dlsa_ludhiana"
                ),
                required_documents=["id"],
            )
        ],
        most_important_next_step="contact dlsa",
        plain_summary=PlainSummary(language="en", text="we will help"),
    )
    res = env.step(Finalize(plan=plan))
    assert res.done
    assert res.observation is None
    assert "reward_breakdown" in res.info
    breakdown = res.info["reward_breakdown"]
    for k in ALL_KEYS:
        assert k in breakdown, f"missing reward key {k}"
    assert isinstance(breakdown["total"], float)


def test_state_locked_without_debug_env(monkeypatch):
    """env.state() must raise without NYAYA_DEBUG=1 and return a dict with it."""
    from nyaya_mitra.citizen.extractor import FactExtractor
    from nyaya_mitra.citizen.simulator import CitizenSimulator
    from nyaya_mitra.env.environment import NyayaMitraEnv
    from nyaya_mitra.knowledge.loader import KnowledgeBase

    monkeypatch.delenv("NYAYA_DEBUG", raising=False)
    env = NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="NYAYA_DEBUG"):
        env.state()

    monkeypatch.setenv("NYAYA_DEBUG", "1")
    snap = env.state()
    assert "profile" in snap
    assert "elicited_facts" in snap


def test_aggregator_emits_all_keys():
    """track B aggregator must emit every key in interface.ALL_KEYS, including TOTAL.

    uses an in-test fake kb so this stays self-contained. depends on
    nyaya_mitra.rewards.compute_reward and the public RewardContext shape.
    """
    from nyaya_mitra.interface import (
        ALL_KEYS,
        ActionPlan,
        ApplicationPath,
        Behavior,
        CitizenProfile,
        DerivedGroundTruth,
        PlainSummary,
        SchemeRecommendation,
        SituationSpecific,
    )
    from nyaya_mitra.rewards import compute_reward
    from nyaya_mitra.rewards.context import RewardContext

    class _FakeKB:
        def has_scheme(self, sid):
            return sid == "pm_kisan"

        def has_framework(self, fid):
            return False

        def has_contact(self, authority, cid):
            return False

        def documents_for_scheme(self, sid):
            return ["Aadhaar"]

        def documents_for_framework(self, fid):
            return []

        def procedural_steps_for_framework(self, fid):
            return []

        def forum_for_framework(self, fid):
            return None

        def legal_aid_authority_for_framework(self, fid):
            return None

        def relevant_facts_for_scheme(self, sid):
            return set()

        def relevant_facts_for_framework(self, fid):
            return set()

    profile = CitizenProfile(
        seed=1,
        situation_specific=SituationSpecific(presenting_issue="x"),
        behavior=Behavior(
            trust_level="neutral",
            verbosity="med",
            language_preference="en",
            literacy="medium",
            initial_vague_query="x",
        ),
        derived_ground_truth=DerivedGroundTruth(eligible_schemes=["pm_kisan"]),
    )
    plan = ActionPlan(
        schemes=[
            SchemeRecommendation(
                scheme_id="pm_kisan",
                rationale_facts=[],
                required_documents=["Aadhaar"],
                application_path=ApplicationPath(),
            )
        ],
        most_important_next_step="apply at csc",
        plain_summary=PlainSummary(language="en", text="will help apply for pm-kisan."),
    )
    ctx = RewardContext(
        profile=profile,
        plan=plan,
        transcript=[],
        elicited_facts=set(),
        kb=_FakeKB(),
        info={"max_turns": 20},
    )
    out = compute_reward(ctx)
    for k in ALL_KEYS:
        assert k in out, f"missing reward key {k}"
    assert isinstance(out["total"], float)


def test_kb_json_matches_schema():
    """every kb json file under knowledge/data/ must validate against the schemas in
    interface/kb_schemas.py."""
    from nyaya_mitra.knowledge.validators import validate_kb

    errors = validate_kb()
    assert not errors, f"kb schema violations: {errors}"
