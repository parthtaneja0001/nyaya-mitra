"""eval harness: runs an Advisor against the 30 held-out cases, groups results
by cohort, computes metrics, and emits a markdown report.

usage from CLI:
    python -m eval.eval_harness --advisor scripted --out eval/report.md

usage from python:
    from eval.eval_harness import run_eval
    report = run_eval(advisor=my_advisor, model_label="trained-7b")

cohort assignment: filename prefix (wel_/leg_/int_) -> welfare_only/legal_only/
integrated. seeds are derived from the eval-case files' "seed" field so the
env can reset to the right profile (TrackA's load_profile reads seeds dir;
eval cases live in eval/eval_cases/, so we pass them in via difficulty='eval'
mapped through the env factory).

since the env's load_profile only knows about src/nyaya_mitra/profile/seeds/,
the harness writes a one-time symlink/copy bridge so eval_cases are visible to
the env loader. the bridge is created on demand and idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from eval.metrics import CohortMetrics, cohort_metrics, overall_from_episodes
from training.rollout import EpisodeResult, run_episode

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CASES_ROOT = REPO_ROOT / "eval" / "eval_cases"
SEEDS_ROOT = REPO_ROOT / "src" / "nyaya_mitra" / "profile" / "seeds"
EVAL_BRIDGE_DIR = SEEDS_ROOT / "_eval"


COHORTS = ("welfare_only", "legal_only", "integrated")
_PREFIX_TO_COHORT = {"wel": "welfare_only", "leg": "legal_only", "int": "integrated"}


def _bridge_eval_cases() -> dict[str, list[Path]]:
    """ensure each eval case is reachable by the env's load_profile.

    load_profile reads from `seeds/<difficulty>/*.json`. we mirror the eval
    cases under `seeds/_eval/<cohort>/<file>.json` so difficulty='_eval/<cohort>'
    selects them. the bridge is recreated on every call (idempotent) and is
    gitignored.
    """
    out: dict[str, list[Path]] = {c: [] for c in COHORTS}
    if EVAL_BRIDGE_DIR.exists():
        shutil.rmtree(EVAL_BRIDGE_DIR)
    EVAL_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        src_dir = EVAL_CASES_ROOT / cohort
        if not src_dir.exists():
            continue
        bridge = EVAL_BRIDGE_DIR / cohort
        bridge.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.json")):
            dst = bridge / src.name
            shutil.copy2(src, dst)
            out[cohort].append(dst)
    return out


def _teardown_bridge() -> None:
    """remove the runtime bridge dir. callers do this after run_eval finishes
    so the dir doesn't leak into other tests / git status."""
    if EVAL_BRIDGE_DIR.exists():
        shutil.rmtree(EVAL_BRIDGE_DIR)


def _seeds_for_cohort(cohort_paths: list[Path]) -> list[int]:
    """read the 'seed' int from each eval-case file. these are passed to env.reset
    so derive_ground_truth runs against the correct profile."""
    seeds: list[int] = []
    for p in cohort_paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            seeds.append(int(data.get("seed", 0)))
        except Exception:
            seeds.append(0)
    return seeds


def run_eval(
    *,
    advisor: Callable,
    env_factory: Callable | None = None,
    model_label: str = "advisor",
    cohorts: Iterable[str] = COHORTS,
    on_episode: Callable[[str, EpisodeResult], None] | None = None,
) -> dict:
    """runs the advisor against the held-out cases and returns a report dict.

    returns: {
        "model_label": str,
        "per_cohort": {cohort: CohortMetrics},
        "overall": CohortMetrics,
        "episodes": {cohort: list[EpisodeResult]},
    }
    """
    if env_factory is None:
        # default factory: scripts.wire_rewards.build_env
        from scripts.wire_rewards import build_env  # local import — avoids GPU side-effects

        env_factory = build_env

    bridge = _bridge_eval_cases()

    results_per_cohort: dict[str, list[EpisodeResult]] = {}
    metrics_per_cohort: dict[str, CohortMetrics] = {}
    all_results: list[EpisodeResult] = []

    for cohort in cohorts:
        if cohort not in COHORTS:
            continue
        paths = bridge.get(cohort, [])
        seeds = _seeds_for_cohort(paths)
        cohort_results: list[EpisodeResult] = []
        difficulty = f"_eval/{cohort}"
        for seed in seeds:
            env = env_factory()
            try:
                r = run_episode(env, advisor, seed=seed, difficulty=difficulty)
            finally:
                env.close()
            cohort_results.append(r)
            all_results.append(r)
            if on_episode is not None:
                try:
                    on_episode(cohort, r)
                except Exception:
                    pass
        results_per_cohort[cohort] = cohort_results
        metrics_per_cohort[cohort] = cohort_metrics(cohort, cohort_results)

    report = {
        "model_label": model_label,
        "per_cohort": metrics_per_cohort,
        "overall": overall_from_episodes(all_results),
        "episodes": results_per_cohort,
    }
    _teardown_bridge()
    return report


def render_report(report: dict, *, include_episode_summary: bool = True) -> str:
    """produces a markdown report with headline numbers and per-cohort breakdown."""
    lines: list[str] = []
    label = report["model_label"]
    overall: CohortMetrics = report["overall"]
    per_cohort: dict[str, CohortMetrics] = report["per_cohort"]

    lines.append(f"# Eval report — {label}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Cases: {overall.n}")
    lines.append(f"- Mean total reward: **{overall.mean_total_reward:.3f}**")
    lines.append(f"- Median: {overall.median_total_reward:.3f}")
    lines.append(f"- P25 / P75: {overall.p25_total_reward:.3f} / {overall.p75_total_reward:.3f}")
    lines.append(f"- All gates passed: **{overall.pct_all_gates_passed:.1f}%**")
    if "integrated" in per_cohort:
        lines.append(
            f"- Integrated cases solved: **{per_cohort['integrated'].pct_integrated_solved:.1f}%**"
        )
    lines.append("")
    lines.append("## Per-cohort")
    lines.append("")
    lines.append(
        "| Cohort | n | mean reward | gates passed | finalized | mean turns | sensitivity F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for cohort in COHORTS:
        m = per_cohort.get(cohort)
        if m is None:
            continue
        lines.append(
            f"| {cohort} | {m.n} | {m.mean_total_reward:.3f} | "
            f"{m.pct_all_gates_passed:.1f}% | {m.pct_finalized:.1f}% | "
            f"{m.mean_turns:.1f} | {m.mean_sensitivity_correctness:.2f} |"
        )
    lines.append("")
    lines.append("## Reward components (means)")
    lines.append("")
    lines.append("| Cohort | scheme P | scheme R | legal P | legal R | turn eff |")
    lines.append("|---|---|---|---|---|---|")
    for cohort in COHORTS:
        m = per_cohort.get(cohort)
        if m is None:
            continue
        lines.append(
            f"| {cohort} | {m.mean_scheme_precision:.2f} | {m.mean_scheme_recall:.2f} | "
            f"{m.mean_legal_precision:.2f} | {m.mean_legal_recall:.2f} | "
            f"{m.mean_turn_efficiency:.2f} |"
        )
    lines.append("")
    lines.append("## Gate triggers (count)")
    lines.append("")
    lines.append("| Cohort | format | hallucination | contradiction | sim leak |")
    lines.append("|---|---|---|---|---|")
    for cohort in COHORTS:
        m = per_cohort.get(cohort)
        if m is None:
            continue
        gc = m.gate_trigger_counts
        lines.append(
            f"| {cohort} | {gc.get('gate_format_violation', 0)} | "
            f"{gc.get('gate_hallucination', 0)} | {gc.get('gate_contradiction', 0)} | "
            f"{gc.get('gate_sim_leak', 0)} |"
        )
    lines.append("")

    if include_episode_summary:
        lines.append("## Episode summary")
        lines.append("")
        for cohort in COHORTS:
            results = report.get("episodes", {}).get(cohort, [])
            if not results:
                continue
            lines.append(f"### {cohort}")
            lines.append("")
            lines.append("| seed | reward | finalized | turns | sim_leak |")
            lines.append("|---|---|---|---|---|")
            for r in results:
                lines.append(
                    f"| {r.seed} | {r.total_reward:.3f} | "
                    f"{'✓' if r.finalized else '✗'} | {len(r.turns)} | {r.sim_leak_count} |"
                )
            lines.append("")

    return "\n".join(lines)


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(report), encoding="utf-8")


def _select_advisor_by_name(name: str) -> Callable:
    if name == "scripted":
        from eval.baselines.scripted_baseline import build_scripted_baseline

        return build_scripted_baseline()
    raise SystemExit(
        f"unknown advisor: {name!r}. only 'scripted' is wired in CLI mode "
        "(others need an LLM backend; call run_eval() directly)."
    )


def _main():
    p = argparse.ArgumentParser(description="run the nyaya-mitra eval harness")
    p.add_argument("--advisor", default="scripted", help="advisor name (scripted)")
    p.add_argument("--out", default="eval/report.md", help="output report path")
    p.add_argument("--label", default=None, help="model label (defaults to advisor name)")
    args = p.parse_args()

    advisor = _select_advisor_by_name(args.advisor)
    report = run_eval(advisor=advisor, model_label=args.label or args.advisor)
    write_report(report, REPO_ROOT / args.out)
    overall: CohortMetrics = report["overall"]
    print(
        f"wrote {args.out} :: n={overall.n} mean_reward={overall.mean_total_reward:.3f} "
        f"gates_passed={overall.pct_all_gates_passed:.1f}%"
    )


if __name__ == "__main__":
    _main()


__all__ = ["COHORTS", "render_report", "run_eval", "write_report"]
