"""dlsa directory tests: schema validity, contact_id uniqueness, every seed-profile state covered."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyaya_mitra.knowledge.loader import KnowledgeBase

DLSA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "nyaya_mitra"
    / "knowledge"
    / "data"
    / "dlsa_directory.json"
)
SEEDS_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "src" / "nyaya_mitra" / "profile" / "seeds"
)


@pytest.fixture(scope="module")
def dlsa() -> dict:
    return json.loads(DLSA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase()


def test_directory_has_nalsa_slsa_dlsa(dlsa):
    assert "NALSA" in dlsa
    assert "SLSAs" in dlsa
    assert "DLSAs" in dlsa
    assert "qualifying_categories" in dlsa


def test_nalsa_has_contact_id(dlsa):
    assert dlsa["NALSA"].get("contact_id")


def test_every_slsa_has_contact_id(dlsa):
    for state, slsa in dlsa["SLSAs"].items():
        assert slsa.get("contact_id"), f"SLSA {state} missing contact_id"


def test_every_dlsa_has_contact_id(dlsa):
    for key, d in dlsa["DLSAs"].items():
        assert d.get("contact_id"), f"DLSA {key} missing contact_id"


def test_contact_ids_are_unique(kb):
    ids = list(kb.all_contact_ids())
    assert len(ids) == len(set(ids)), f"duplicate contact_ids: {ids}"


def test_dlsa_keys_match_state_dot_district_pattern(dlsa):
    """dlsa keys must follow state.district format so the profile state field can lookup."""
    for key in dlsa["DLSAs"].keys():
        assert "." in key, f"DLSA key {key} missing dot separator"
        state, district = key.split(".", 1)
        assert state and district, f"DLSA key {key} has empty state or district"


def test_every_slsa_state_appears_in_at_least_one_dlsa(dlsa):
    """each SLSA state should have at least one DLSA so plans can route to a district contact."""
    slsa_states = set(dlsa["SLSAs"].keys())
    dlsa_states = {key.split(".", 1)[0] for key in dlsa["DLSAs"].keys()}
    missing = slsa_states - dlsa_states
    assert not missing, f"SLSAs without any DLSA: {sorted(missing)}"


def test_every_seed_profile_state_has_dlsa_coverage(dlsa):
    """if a seed profile lives in a state, that state must have at least one DLSA so the
    advisor can route plans to a real free_legal_aid_contact for that profile."""
    dlsa_states = {key.split(".", 1)[0] for key in dlsa["DLSAs"].keys()}
    profile_states: set[str] = set()
    for seed in SEEDS_ROOT.rglob("*.json"):
        raw = json.loads(seed.read_text(encoding="utf-8"))
        st = raw.get("demographics", {}).get("state")
        if st:
            profile_states.add(st)
    missing = profile_states - dlsa_states
    assert not missing, f"profile states without DLSA coverage: {sorted(missing)}"


def test_qualifying_categories_non_empty(dlsa):
    assert len(dlsa["qualifying_categories"]) >= 5
