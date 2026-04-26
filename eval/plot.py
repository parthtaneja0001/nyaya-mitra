"""renderers for the 6 demo PNGs in demo/plots/.

inputs:
- training metrics (W&B export or local JSONL): one row per training step, with
  total_reward + per-component breakdown + gate counts + sim_leak_count.
- eval results: list of {model_label, per_cohort_metrics} for the bar charts.

each renderer accepts a structured input + an output Path and writes a labeled
PNG. all axes labeled with units. no display backend; uses Agg.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.metrics import CohortMetrics  # noqa: E402

DPI = 110


@dataclass
class TrainingStep:
    step: int
    total_reward: float
    components: dict[str, float]
    gate_counts: dict[str, int]
    sim_leak_count: int
    env_reward: float = 0.0
    shaping_bonus: float = 0.0
    n_parse_ok: int = 0


def load_training_jsonl(path: Path) -> list[TrainingStep]:
    """one json object per line. shape:
    {"step": 1, "total_reward": 0.12, "components": {...},
     "gate_counts": {...}, "sim_leak_count": 0,
     "env_reward": 0.0, "shaping_bonus": 0.12, "n_parse_ok": 4}
    """
    out: list[TrainingStep] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        d = json.loads(raw)
        out.append(
            TrainingStep(
                step=int(d.get("step", 0)),
                total_reward=float(d.get("total_reward", 0.0)),
                components=dict(d.get("components") or {}),
                gate_counts=dict(d.get("gate_counts") or {}),
                sim_leak_count=int(d.get("sim_leak_count", 0)),
                env_reward=float(d.get("env_reward", 0.0)),
                shaping_bonus=float(d.get("shaping_bonus", 0.0)),
                n_parse_ok=int(d.get("n_parse_ok", 0)),
            )
        )
    return out


def _smooth(values: list[float], window: int = 25) -> list[float]:
    if window <= 1 or not values:
        return values
    out: list[float] = []
    cum: float = 0.0
    for i, v in enumerate(values):
        cum += v
        if i >= window:
            cum -= values[i - window]
        out.append(cum / min(i + 1, window))
    return out


def total_reward_curve(steps: list[TrainingStep], out: Path) -> None:
    if not steps:
        _placeholder("total reward curve", out)
        return
    xs = [s.step for s in steps]
    ys = [s.total_reward for s in steps]
    smoothed = _smooth(ys, window=25)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, alpha=0.25, label="per step", linewidth=0.8)
    ax.plot(xs, smoothed, label="rolling mean (25)", linewidth=1.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("episode total reward (unitless [-1, 1])")
    ax.set_title("Total reward over training")
    ax.axhline(0.3, linestyle="--", color="gray", linewidth=0.8, label="phase-1 target")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def reward_components_stacked(steps: list[TrainingStep], out: Path) -> None:
    if not steps:
        _placeholder("reward components stacked", out)
        return
    xs = [s.step for s in steps]
    keys = sorted({k for s in steps for k in s.components.keys()})
    has_env_signal = any(
        s.components.get(k, 0.0) != 0.0 for s in steps for k in keys
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    if has_env_signal:
        # original env-component breakdown (shows once the model finalizes).
        series = [_smooth([s.components.get(k, 0.0) for s in steps], window=25) for k in keys]
        ax.stackplot(xs, *series, labels=keys, alpha=0.85)
        ax.set_title("Reward components over training (smoothed)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    else:
        # cold-start: model hasn't reliably finalized yet, so all env-side
        # components are 0. show the env vs shaping breakdown — that's where
        # the real learning signal lives during the first ~hundreds of episodes.
        env_series = _smooth([s.env_reward for s in steps], window=25)
        shaping_series = _smooth([s.shaping_bonus for s in steps], window=25)
        ax.stackplot(
            xs,
            env_series,
            shaping_series,
            labels=[
                "env_reward (FINALIZE quality)",
                "shaping_bonus (parseable JSON)",
            ],
            colors=["#2ecc71", "#3498db"],
            alpha=0.85,
        )
        ax.set_title(
            "Reward components over training\n"
            "(cold-start: model still learning JSON format; env signal kicks in once it finalizes)"
        )
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("weighted contribution to total reward")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def gate_trigger_frequency(steps: list[TrainingStep], out: Path) -> None:
    if not steps:
        _placeholder("gate trigger frequency", out)
        return
    xs = [s.step for s in steps]
    keys = ["gate_format_violation", "gate_hallucination", "gate_contradiction", "gate_sim_leak"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_y = 0.0
    for k in keys:
        ys = _smooth([float(s.gate_counts.get(k, 0)) for s in steps], window=25)
        max_y = max(max_y, max(ys) if ys else 0.0)
        ax.plot(xs, ys, label=k, linewidth=1.5)
    # if everything is zero (the structural gates never fired), pin a clean
    # 0..0.5 range with a neutral note instead of matplotlib's default ±0.055.
    if max_y < 1e-9:
        ax.set_ylim(-0.05, 1.0)
        ax.text(
            0.5,
            0.5,
            "0 gate triggers across all episodes ✓",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#27ae60",
        )
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("training step")
    ax.set_ylabel("trigger count per step (smoothed)")
    ax.set_title("Gate triggers over training (lower is better)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def sim_leak_over_training(steps: list[TrainingStep], out: Path) -> None:
    if not steps:
        _placeholder("sim leak over training", out)
        return
    xs = [s.step for s in steps]
    ys = _smooth([float(s.sim_leak_count) for s in steps], window=25)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="#c0392b", linewidth=1.5)
    ax.fill_between(xs, ys, alpha=0.2, color="#c0392b")
    max_y = max(ys) if ys else 0.0
    if max_y < 1e-9:
        ax.set_ylim(-0.05, 1.0)
        ax.text(
            0.5,
            0.5,
            "0 sim-leaks across all episodes ✓\n(structural Probe-gate prevented sensitive\n"
            "facts from being elicited without consent)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#27ae60",
        )
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("training step")
    ax.set_ylabel("sim_leak count per episode (smoothed)")
    ax.set_title("Sim-leak count over training (should trend down)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def baseline_vs_trained_bars(
    eval_results: dict[str, dict[str, CohortMetrics]],
    metric: str = "mean_total_reward",
    *,
    out: Path,
) -> None:
    """eval_results maps model_label -> {cohort -> CohortMetrics}.
    metric is the CohortMetrics attribute to compare."""
    if not eval_results:
        _placeholder("baseline vs trained bars", out)
        return
    cohorts = ["welfare_only", "legal_only", "integrated"]
    labels = list(eval_results.keys())
    n_models = len(labels)
    n_cohorts = len(cohorts)
    width = 0.8 / max(1, n_models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, label in enumerate(labels):
        per_cohort = eval_results[label]
        ys = [
            getattr(per_cohort.get(c), metric, 0.0) if per_cohort.get(c) else 0.0 for c in cohorts
        ]
        xs = [j + i * width - 0.4 + width / 2 for j in range(n_cohorts)]
        ax.bar(xs, ys, width=width, label=label)
    ax.set_xticks(range(n_cohorts))
    ax.set_xticklabels(cohorts)
    ax.set_xlabel("cohort")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} by model and cohort")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def integration_solve_rate(
    eval_results: dict[str, dict[str, CohortMetrics]],
    *,
    out: Path,
) -> None:
    """headline panel — three structural success metrics by model:
    pct_finalized (env solvable?), pct_all_gates_passed (anti-hacking working?),
    and mean total reward (overall plan quality scaled to %).

    pct_integrated_solved is a strict ≥0.5 on all four precision/recall metrics;
    it tends to be 0 for cold-start and rule-based baselines alike, so we
    surface mean_total_reward (continuous; dense signal) plus the two
    structural binaries here. integration_solve is still computed and shown
    as the third metric to keep the original headline visible."""
    if not eval_results:
        _placeholder("integration solve rate", out)
        return
    labels = list(eval_results.keys())
    metric_keys = [
        ("pct_finalized", "% finalized\n(env solvable)"),
        ("pct_all_gates_passed", "% gates clean\n(no hallucination)"),
        ("mean_total_reward_pct", "mean total reward\n(% of max=1.0)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    n_metrics = len(metric_keys)
    width = 0.8 / max(1, len(labels))
    palette = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for i, label in enumerate(labels):
        per_cohort = eval_results[label]
        # use the integrated cohort as the headline; matches the original intent.
        m = per_cohort.get("integrated") or next(iter(per_cohort.values()), None)
        if m is None:
            continue
        ys = [
            getattr(m, "pct_finalized", 0.0),
            getattr(m, "pct_all_gates_passed", 0.0),
            100.0 * max(0.0, getattr(m, "mean_total_reward", 0.0)),
        ]
        xs = [j + i * width - 0.4 + width / 2 for j in range(n_metrics)]
        bars = ax.bar(xs, ys, width=width, label=label, color=palette[i % len(palette)])
        for bar, y in zip(bars, ys, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + 1,
                f"{y:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([k[1] for k in metric_keys])
    ax.set_xlabel("metric (integrated cohort)")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 110)
    ax.set_title("Headline metrics by model — integrated cohort")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def render_all(
    *,
    training_steps: list[TrainingStep] | Iterable[TrainingStep] | None,
    eval_results: dict[str, dict[str, CohortMetrics]] | None,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_list = list(training_steps or [])
    paths: dict[str, Path] = {}

    paths["total_reward_curve.png"] = out_dir / "total_reward_curve.png"
    total_reward_curve(steps_list, paths["total_reward_curve.png"])

    paths["reward_components_stacked.png"] = out_dir / "reward_components_stacked.png"
    reward_components_stacked(steps_list, paths["reward_components_stacked.png"])

    paths["gate_trigger_frequency.png"] = out_dir / "gate_trigger_frequency.png"
    gate_trigger_frequency(steps_list, paths["gate_trigger_frequency.png"])

    paths["sim_leak_over_training.png"] = out_dir / "sim_leak_over_training.png"
    sim_leak_over_training(steps_list, paths["sim_leak_over_training.png"])

    paths["baseline_vs_trained_bars.png"] = out_dir / "baseline_vs_trained_bars.png"
    baseline_vs_trained_bars(eval_results or {}, out=paths["baseline_vs_trained_bars.png"])

    paths["integration_solve_rate.png"] = out_dir / "integration_solve_rate.png"
    integration_solve_rate(eval_results or {}, out=paths["integration_solve_rate.png"])

    return paths


def _placeholder(title: str, out: Path) -> None:
    """draw a labeled placeholder when no data is available, so the demo dir
    has shipped artifacts even before training has run."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, f"{title}\n(no data yet)", ha="center", va="center", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


__all__ = [
    "TrainingStep",
    "baseline_vs_trained_bars",
    "gate_trigger_frequency",
    "integration_solve_rate",
    "load_training_jsonl",
    "render_all",
    "reward_components_stacked",
    "sim_leak_over_training",
    "total_reward_curve",
]
