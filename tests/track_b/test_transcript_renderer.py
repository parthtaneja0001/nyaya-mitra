"""transcript renderer tests. uses the scripted baseline so we get real
EpisodeResult shapes without an LLM."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from demo.transcript_renderer import (
    render_demo_set,
    render_side_by_side,
    render_transcript,
)
from eval.baselines.scripted_baseline import build_scripted_baseline
from training.rollout import run_episode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _wire():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module("scripts.wire_rewards")
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


def _run() -> object:
    wire = _wire()
    env = wire.build_env(max_turns=6)
    return run_episode(env, build_scripted_baseline(), seed=1)


def test_render_transcript_includes_summary_block():
    r = _run()
    md = render_transcript(r, title="case_001")
    assert "case_001" in md
    assert "seed" in md and "total reward" in md
    assert "Action plan" in md
    assert "Reward breakdown" in md


def test_render_transcript_lists_turns():
    r = _run()
    md = render_transcript(r)
    # at least one ASK or PROBE before FINALIZE
    assert "FINALIZE" in md


def test_render_side_by_side_shows_both_labels():
    left = _run()
    right = _run()
    md = render_side_by_side(left, right, left_label="vanilla", right_label="trained")
    assert "vanilla" in md
    assert "trained" in md
    assert "## vanilla" in md
    assert "## trained" in md


def test_render_side_by_side_header_table_present():
    left = _run()
    right = _run()
    md = render_side_by_side(left, right)
    assert "| | total | finalized | turns | gates passed |" in md


def test_render_demo_set_writes_one_file_per_case(tmp_path: Path):
    rs = [_run(), _run(), _run()]
    paths = render_demo_set(
        tmp_path,
        {"baseline": rs, "trained": rs},
        cases=["wel_001", "leg_001", "int_001"],
    )
    assert set(paths.keys()) == {"wel_001", "leg_001", "int_001"}
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 0


def test_render_demo_set_handles_single_label(tmp_path: Path):
    rs = [_run()]
    paths = render_demo_set(tmp_path, {"only": rs}, cases=["solo"])
    assert paths["solo"].exists()


def test_render_demo_set_handles_empty(tmp_path: Path):
    paths = render_demo_set(tmp_path, {})
    assert paths == {}
