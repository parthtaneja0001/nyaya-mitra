"""regression test: every seed profile must derive at least one ground-truth match.
a profile that matches nothing is dead weight for training and likely a checker bug or
a bad profile design."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyaya_mitra.interface import CitizenProfile
from nyaya_mitra.knowledge.loader import KnowledgeBase
from nyaya_mitra.profile.derivation import derive_ground_truth

SEEDS_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "src" / "nyaya_mitra" / "profile" / "seeds"
)

EXPECTED: dict[str, dict[str, set[str]]] = {
    "seed_001": {
        "schemes": {"pm_kisan", "pmuy", "mgnrega"},
        "frameworks": {"domestic_violence_act_2005"},
    },
    "seed_002": {
        "schemes": {"pmuy", "mgnrega", "pmsby"},
        "frameworks": set(),
    },
    "seed_003": {
        "schemes": {"pmsby"},
        "frameworks": {"consumer_protection_act_2019"},
    },
    "seed_004": {
        "schemes": {"pm_kisan", "pmuy", "mgnrega", "pmsby"},
        "frameworks": set(),
    },
    "seed_005": {
        "schemes": {"pmsby"},
        "frameworks": {"maternity_benefit_act_1961"},
    },
    "seed_006": {
        "schemes": {"pmuy", "pmsby"},
        "frameworks": {"maternity_benefit_act_1961", "domestic_violence_act_2005"},
    },
}


def _all_seed_paths() -> list[Path]:
    return sorted(SEEDS_ROOT.rglob("*.json"))


@pytest.mark.parametrize("seed_path", _all_seed_paths(), ids=lambda p: p.stem)
def test_every_seed_matches_at_least_one_ground_truth_item(seed_path: Path):
    raw = json.loads(seed_path.read_text(encoding="utf-8"))
    profile = CitizenProfile.model_validate(raw)
    kb = KnowledgeBase()
    truth = derive_ground_truth(profile, kb)
    total = len(truth.eligible_schemes) + len(truth.applicable_frameworks)
    assert total >= 1, (
        f"{seed_path.name} matches zero schemes/frameworks; either profile is "
        f"under-specified or a checker is wrong"
    )


@pytest.mark.parametrize(
    "seed_path",
    _all_seed_paths(),
    ids=lambda p: p.stem,
)
def test_seed_ground_truth_matches_expected(seed_path: Path):
    name = seed_path.stem
    if name not in EXPECTED:
        pytest.skip(f"no expected truth for {name} yet")
    raw = json.loads(seed_path.read_text(encoding="utf-8"))
    profile = CitizenProfile.model_validate(raw)
    kb = KnowledgeBase()
    truth = derive_ground_truth(profile, kb)
    exp = EXPECTED[name]
    assert set(truth.eligible_schemes) == exp["schemes"], (
        f"{name} eligible schemes drift: got {sorted(truth.eligible_schemes)}, "
        f"expected {sorted(exp['schemes'])}"
    )
    assert set(truth.applicable_frameworks) == exp["frameworks"], (
        f"{name} applicable frameworks drift: got {sorted(truth.applicable_frameworks)}, "
        f"expected {sorted(exp['frameworks'])}"
    )


def test_seed_difficulty_split_balanced():
    """sanity: profiles distributed across easy/medium/hard."""
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    for p in _all_seed_paths():
        diff = p.parent.name
        if diff in by_diff:
            by_diff[diff] += 1
    for diff in ("easy", "medium", "hard"):
        assert by_diff[diff] >= 1, f"no profiles in {diff} bucket"
