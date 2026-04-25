"""eval harness end-to-end tests using the scripted baseline (no LLM needed)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.baselines.scripted_baseline import build_scripted_baseline
from eval.eval_harness import (
    COHORTS,
    _bridge_eval_cases,
    _teardown_bridge,
    render_report,
    run_eval,
    write_report,
)
from eval.metrics import (
    cohort_metrics,
    overall_from_episodes,
)
from training.rollout import EpisodeResult


def test_bridge_creates_per_cohort_directories():
    try:
        bridge = _bridge_eval_cases()
        assert set(bridge.keys()) == set(COHORTS)
        for cohort, paths in bridge.items():
            assert paths, f"no eval files bridged for cohort {cohort}"
            for p in paths[:1]:
                data = json.loads(p.read_text(encoding="utf-8"))
                assert "seed" in data
    finally:
        _teardown_bridge()


def test_run_eval_with_scripted_baseline_returns_30_episodes():
    advisor = build_scripted_baseline(max_asks=2, max_probes=1)
    report = run_eval(advisor=advisor, model_label="scripted")
    total = report["overall"].n
    # 10 cases per cohort * 3 cohorts = 30
    assert total == 30
    for cohort in COHORTS:
        m = report["per_cohort"][cohort]
        assert m.n == 10, f"{cohort} has n={m.n}, expected 10"


def test_run_eval_each_cohort_produces_breakdown():
    advisor = build_scripted_baseline()
    report = run_eval(advisor=advisor, model_label="scripted")
    for cohort in COHORTS:
        m = report["per_cohort"][cohort]
        # gates ran (count is integer >= 0)
        assert all(v >= 0 for v in m.gate_trigger_counts.values())
        assert 0.0 <= m.pct_finalized <= 100.0
        assert 0.0 <= m.pct_all_gates_passed <= 100.0


def test_render_report_includes_headline_and_per_cohort():
    advisor = build_scripted_baseline()
    report = run_eval(advisor=advisor, model_label="scripted-test")
    md = render_report(report)
    assert "scripted-test" in md
    assert "Headline" in md
    assert "Per-cohort" in md
    assert "welfare_only" in md
    assert "legal_only" in md
    assert "integrated" in md
    assert "Reward components" in md
    assert "Gate triggers" in md


def test_write_report_creates_file(tmp_path: Path):
    advisor = build_scripted_baseline()
    report = run_eval(advisor=advisor, model_label="scripted-write")
    out = tmp_path / "report.md"
    write_report(report, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "scripted-write" in text


def test_metrics_handle_empty_results():
    m = cohort_metrics("empty", [])
    assert m.n == 0
    assert m.mean_total_reward == 0.0
    assert m.pct_all_gates_passed == 0.0


def test_metrics_compute_basic_numbers():
    # synthetic episodes with known breakdowns
    eps = [
        EpisodeResult(
            seed=1,
            difficulty=None,
            turns=[],
            final_breakdown={
                "scheme_precision": 1.0,
                "scheme_recall": 1.0,
                "legal_precision": 1.0,
                "legal_recall": 1.0,
                "integration_bonus": 1.0,
                "sensitivity_correctness": 1.0,
                "turn_efficiency": 0.5,
                "gate_format_violation": 0.0,
                "gate_hallucination": 0.0,
                "gate_contradiction": 0.0,
                "gate_sim_leak": 0.0,
                "total": 0.95,
            },
            total_reward=0.95,
            finalized=True,
            truncated_by_env=False,
            elicited_facts=[],
            sim_leak_count=0,
            wall_seconds=0.1,
        ),
        EpisodeResult(
            seed=2,
            difficulty=None,
            turns=[],
            final_breakdown={
                "scheme_precision": 0.0,
                "scheme_recall": 0.0,
                "legal_precision": 0.0,
                "legal_recall": 0.0,
                "integration_bonus": 0.0,
                "sensitivity_correctness": 0.0,
                "turn_efficiency": 0.0,
                "gate_format_violation": 1.0,
                "gate_hallucination": 0.0,
                "gate_contradiction": 0.0,
                "gate_sim_leak": 0.0,
                "total": -1.0,
            },
            total_reward=-1.0,
            finalized=False,
            truncated_by_env=True,
            elicited_facts=[],
            sim_leak_count=0,
            wall_seconds=0.1,
        ),
    ]
    m = cohort_metrics("test", eps)
    assert m.n == 2
    assert abs(m.mean_total_reward - (-0.025)) < 1e-9
    assert m.pct_all_gates_passed == 50.0
    assert m.pct_integrated_solved == 50.0  # one episode hit it
    assert m.gate_trigger_counts["gate_format_violation"] == 1
    assert m.gate_trigger_counts["gate_hallucination"] == 0


def test_overall_aggregates_across_cohorts():
    eps_a = [_make_ep(1, 0.6), _make_ep(2, 0.4)]
    eps_b = [_make_ep(3, 0.2)]
    overall = overall_from_episodes(eps_a + eps_b)
    assert overall.n == 3
    assert abs(overall.mean_total_reward - (0.6 + 0.4 + 0.2) / 3) < 1e-9


def _make_ep(seed: int, total: float) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        difficulty=None,
        turns=[],
        final_breakdown={"total": total},
        total_reward=total,
        finalized=True,
        truncated_by_env=False,
        elicited_facts=[],
        sim_leak_count=0,
        wall_seconds=0.1,
    )
