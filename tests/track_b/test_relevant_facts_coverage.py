"""regression test: every KB scheme/framework must have a relevant_facts entry.

without this check, track A growing the KB silently zeroes fact_coverage and
integration_bonus for the new entries. this test fails loudly when that
happens, forcing the map to grow alongside the KB.

uses the real KB if available; if not, skips gracefully so this stays
useful in CI configurations that don't install track-a deps.
"""

from __future__ import annotations

import importlib

import pytest

from nyaya_mitra.rewards.kb_adapter import _DEFAULT_RELEVANT_FACTS


def _real_kb_or_skip():
    try:
        loader = importlib.import_module("nyaya_mitra.knowledge.loader")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"track-a kb not importable: {exc}")
    try:
        return loader.KnowledgeBase()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"kb construction failed: {exc}")


def test_every_scheme_has_relevant_facts_entry():
    kb = _real_kb_or_skip()
    missing = [sid for sid in kb.scheme_ids() if sid not in _DEFAULT_RELEVANT_FACTS]
    assert not missing, (
        "schemes without relevant_facts entry — fact_coverage is silently zero "
        f"for these: {missing}"
    )


def test_every_framework_has_relevant_facts_entry():
    kb = _real_kb_or_skip()
    missing = [fid for fid in kb.framework_ids() if fid not in _DEFAULT_RELEVANT_FACTS]
    assert not missing, (
        "frameworks without relevant_facts entry — fact_coverage is silently zero "
        f"for these: {missing}"
    )


def test_relevant_facts_have_non_empty_sets():
    """an entry with an empty set is just as broken as a missing entry."""
    empty = [k for k, v in _DEFAULT_RELEVANT_FACTS.items() if not v]
    assert not empty, f"relevant_facts entries with empty set: {empty}"


def test_fact_ids_are_well_formed():
    """fact ids should be snake_case lowercase. catches typos."""
    bad: list[str] = []
    for kb_id, facts in _DEFAULT_RELEVANT_FACTS.items():
        for f in facts:
            if not f or not all(c.islower() or c.isdigit() or c == "_" for c in f):
                bad.append(f"{kb_id}: {f!r}")
    assert not bad, f"malformed fact ids: {bad}"


def test_kb_adapter_agrees_with_track_a_relevant_facts():
    """canonical source is profile/relevant_facts._RELEVANT_BY_ID. mine must match.

    drift between the two maps means fact_coverage and the env-side relevant_facts()
    disagree on what counts as relevant — agent learns one signal, env reports
    another. fail loudly here so the next divergence gets caught immediately.

    long-term fix is to move the mapping into KB JSON (interface task on board).
    """
    try:
        track_a = importlib.import_module("nyaya_mitra.profile.relevant_facts")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"track-a profile module not importable: {exc}")
    canonical = getattr(track_a, "_RELEVANT_BY_ID", None)
    if canonical is None:  # pragma: no cover
        pytest.skip("track-a _RELEVANT_BY_ID not exposed")

    diffs: list[str] = []
    for kb_id, facts in canonical.items():
        mine = _DEFAULT_RELEVANT_FACTS.get(kb_id)
        if mine is None:
            diffs.append(f"{kb_id}: track-a has {sorted(facts)}, kb_adapter missing")
            continue
        if mine != facts:
            only_a = sorted(facts - mine)
            only_b = sorted(mine - facts)
            diffs.append(f"{kb_id}: track-a only={only_a} kb_adapter only={only_b}")
    extra_in_b = sorted(set(_DEFAULT_RELEVANT_FACTS) - set(canonical))
    if extra_in_b:
        diffs.append(f"kb_adapter has entries track-a does not: {extra_in_b}")
    assert not diffs, (
        "fact-id drift between kb_adapter and profile/relevant_facts:\n  " + "\n  ".join(diffs)
    )
