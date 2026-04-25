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
