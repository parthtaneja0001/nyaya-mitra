"""plot smoke tests. uses synthetic training-step data and synthetic eval results
to confirm each renderer writes a non-empty PNG without raising."""

from __future__ import annotations

from pathlib import Path

from eval.metrics import CohortMetrics
from eval.plot import (
    TrainingStep,
    baseline_vs_trained_bars,
    gate_trigger_frequency,
    integration_solve_rate,
    load_training_jsonl,
    render_all,
    reward_components_stacked,
    sim_leak_over_training,
    total_reward_curve,
)


def _synth_steps(n: int = 50) -> list[TrainingStep]:
    return [
        TrainingStep(
            step=i,
            total_reward=0.1 + 0.005 * i,
            components={
                "scheme_precision": 0.05,
                "scheme_recall": 0.04,
                "legal_precision": 0.05,
                "legal_recall": 0.05,
                "fact_coverage": 0.05,
                "integration_bonus": 0.03,
            },
            gate_counts={
                "gate_format_violation": max(0, 5 - i // 10),
                "gate_hallucination": max(0, 3 - i // 15),
                "gate_contradiction": 0,
                "gate_sim_leak": max(0, 2 - i // 20),
            },
            sim_leak_count=max(0, 4 - i // 12),
        )
        for i in range(n)
    ]


def _synth_eval() -> dict[str, dict[str, CohortMetrics]]:
    def m(label: str, mean: float, integrated_pct: float) -> dict[str, CohortMetrics]:
        return {
            "welfare_only": CohortMetrics(
                cohort="welfare_only",
                n=10,
                mean_total_reward=mean,
                median_total_reward=mean,
                p25_total_reward=mean - 0.1,
                p75_total_reward=mean + 0.1,
                pct_finalized=100.0,
                pct_truncated=0.0,
                mean_turns=5.0,
                pct_all_gates_passed=100.0,
                pct_integrated_solved=0.0,
                mean_sensitivity_correctness=0.8,
                mean_scheme_precision=0.8,
                mean_scheme_recall=0.5,
                mean_legal_precision=1.0,
                mean_legal_recall=1.0,
                mean_turn_efficiency=0.4,
                mean_sim_leak_count=0.0,
            ),
            "legal_only": CohortMetrics(
                cohort="legal_only",
                n=10,
                mean_total_reward=mean - 0.1,
                median_total_reward=mean - 0.1,
                p25_total_reward=mean - 0.2,
                p75_total_reward=mean,
                pct_finalized=100.0,
                pct_truncated=0.0,
                mean_turns=5.0,
                pct_all_gates_passed=100.0,
                pct_integrated_solved=0.0,
                mean_sensitivity_correctness=0.8,
                mean_scheme_precision=0.0,
                mean_scheme_recall=1.0,
                mean_legal_precision=1.0,
                mean_legal_recall=0.5,
                mean_turn_efficiency=0.4,
                mean_sim_leak_count=0.0,
            ),
            "integrated": CohortMetrics(
                cohort="integrated",
                n=10,
                mean_total_reward=mean,
                median_total_reward=mean,
                p25_total_reward=mean - 0.1,
                p75_total_reward=mean + 0.1,
                pct_finalized=100.0,
                pct_truncated=0.0,
                mean_turns=6.0,
                pct_all_gates_passed=100.0,
                pct_integrated_solved=integrated_pct,
                mean_sensitivity_correctness=0.4,
                mean_scheme_precision=1.0,
                mean_scheme_recall=0.5,
                mean_legal_precision=1.0,
                mean_legal_recall=0.5,
                mean_turn_efficiency=0.0,
                mean_sim_leak_count=0.0,
            ),
        }

    return {
        "vanilla": m("vanilla", 0.30, 5.0),
        "prompted": m("prompted", 0.45, 30.0),
        "trained": m("trained", 0.65, 70.0),
    }


def test_total_reward_curve_writes_png(tmp_path: Path):
    out = tmp_path / "total.png"
    total_reward_curve(_synth_steps(), out)
    assert out.exists() and out.stat().st_size > 1000


def test_components_stacked_writes_png(tmp_path: Path):
    out = tmp_path / "stacked.png"
    reward_components_stacked(_synth_steps(), out)
    assert out.exists() and out.stat().st_size > 1000


def test_gate_trigger_frequency_writes_png(tmp_path: Path):
    out = tmp_path / "gates.png"
    gate_trigger_frequency(_synth_steps(), out)
    assert out.exists() and out.stat().st_size > 1000


def test_sim_leak_over_training_writes_png(tmp_path: Path):
    out = tmp_path / "leaks.png"
    sim_leak_over_training(_synth_steps(), out)
    assert out.exists() and out.stat().st_size > 1000


def test_baseline_vs_trained_bars_writes_png(tmp_path: Path):
    out = tmp_path / "bars.png"
    baseline_vs_trained_bars(_synth_eval(), out=out)
    assert out.exists() and out.stat().st_size > 1000


def test_integration_solve_rate_writes_png(tmp_path: Path):
    out = tmp_path / "headline.png"
    integration_solve_rate(_synth_eval(), out=out)
    assert out.exists() and out.stat().st_size > 1000


def test_render_all_creates_six_pngs(tmp_path: Path):
    paths = render_all(
        training_steps=_synth_steps(),
        eval_results=_synth_eval(),
        out_dir=tmp_path,
    )
    assert len(paths) == 6
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 1000


def test_render_all_with_no_data_emits_placeholders(tmp_path: Path):
    """no training run yet, no eval — still writes 6 placeholder PNGs."""
    paths = render_all(training_steps=None, eval_results=None, out_dir=tmp_path)
    assert len(paths) == 6
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 500


def test_load_training_jsonl_round_trips(tmp_path: Path):
    src = tmp_path / "metrics.jsonl"
    src.write_text(
        "\n".join(
            [
                '{"step":0,"total_reward":0.1,"components":{"a":1.0},"gate_counts":{},"sim_leak_count":0}',
                '{"step":1,"total_reward":0.2,"components":{"a":1.0},"gate_counts":{},"sim_leak_count":1}',
            ]
        ),
        encoding="utf-8",
    )
    steps = load_training_jsonl(src)
    assert len(steps) == 2
    assert steps[0].total_reward == 0.1
    assert steps[1].sim_leak_count == 1


def test_load_training_jsonl_handles_missing_file(tmp_path: Path):
    assert load_training_jsonl(tmp_path / "nope.jsonl") == []
