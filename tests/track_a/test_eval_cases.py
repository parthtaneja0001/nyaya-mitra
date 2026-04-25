"""eval-case smoke tests: every case parses, derives ground truth, and matches its
cohort discipline (welfare-only / legal-only / integrated)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyaya_mitra.interface import CitizenProfile
from nyaya_mitra.knowledge.loader import KnowledgeBase
from nyaya_mitra.profile.derivation import derive_ground_truth

EVAL_ROOT = Path(__file__).resolve().parent.parent.parent / "eval" / "eval_cases"
COHORTS = ("welfare_only", "legal_only", "integrated")


def _all_cases():
    cases: list[tuple[str, Path]] = []
    for cohort in COHORTS:
        for p in sorted((EVAL_ROOT / cohort).glob("*.json")):
            cases.append((cohort, p))
    return cases


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase()


def test_thirty_eval_cases_present():
    cases = _all_cases()
    assert len(cases) == 30, f"expected 30 eval cases, got {len(cases)}"


def test_each_cohort_has_ten_cases():
    for cohort in COHORTS:
        cases = list((EVAL_ROOT / cohort).glob("*.json"))
        assert len(cases) == 10, f"{cohort} has {len(cases)} cases (expected 10)"


@pytest.mark.parametrize(
    "cohort, path", _all_cases(), ids=lambda x: x.stem if hasattr(x, "stem") else str(x)
)
def test_eval_case_parses_and_derives(cohort: str, path: Path, kb: KnowledgeBase):
    raw = json.loads(path.read_text(encoding="utf-8"))
    profile = CitizenProfile.model_validate(raw)
    truth = derive_ground_truth(profile, kb)
    schemes = set(truth.eligible_schemes)
    frameworks = set(truth.applicable_frameworks)

    if cohort == "welfare_only":
        assert len(schemes) >= 1, f"{path.name}: welfare-only needs >=1 scheme"
        assert not frameworks, f"{path.name}: welfare-only must have 0 frameworks, got {frameworks}"
    elif cohort == "legal_only":
        assert len(frameworks) >= 1, f"{path.name}: legal-only needs >=1 framework"
        scheme_set = schemes - {"pmsby"}
        assert not scheme_set, (
            f"{path.name}: legal-only allows pmsby only, got extra schemes {scheme_set}"
        )
    elif cohort == "integrated":
        assert len(schemes) >= 1, f"{path.name}: integrated needs >=1 scheme"
        assert len(frameworks) >= 1, f"{path.name}: integrated needs >=1 framework"


def test_eval_seeds_are_held_out_from_training_seeds():
    """eval seeds use seed_id 100+; training seeds use 1-14. no overlap."""
    eval_seeds: set[int] = set()
    for _, p in _all_cases():
        raw = json.loads(p.read_text(encoding="utf-8"))
        eval_seeds.add(raw["seed"])

    train_root = (
        Path(__file__).resolve().parent.parent.parent / "src" / "nyaya_mitra" / "profile" / "seeds"
    )
    train_seeds: set[int] = set()
    for p in train_root.rglob("*.json"):
        raw = json.loads(p.read_text(encoding="utf-8"))
        train_seeds.add(raw["seed"])

    overlap = eval_seeds & train_seeds
    assert not overlap, f"eval seeds overlap with training seeds: {sorted(overlap)}"


def test_eval_cases_stratified_by_language():
    langs: set[str] = set()
    for _, p in _all_cases():
        raw = json.loads(p.read_text(encoding="utf-8"))
        langs.add(raw["behavior"]["language_preference"])
    assert {"en", "hi", "hinglish"}.issubset(langs), f"missing language coverage: {langs}"


def test_eval_cases_stratified_by_state():
    states: set[str] = set()
    for _, p in _all_cases():
        raw = json.loads(p.read_text(encoding="utf-8"))
        states.add(raw["demographics"]["state"])
    assert len(states) >= 8, f"only {len(states)} distinct states represented"
