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


def load_training_jsonl(path: Path) -> list[TrainingStep]:
    """one json object per line. shape:
    {"step": 1, "total_reward": 0.12, "components": {...},
     "gate_counts": {...}, "sim_leak_count": 0}
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
    keys = sorted({k for s in steps for k in s.components.keys()})
    xs = [s.step for s in steps]
    series = [_smooth([s.components.get(k, 0.0) for s in steps], window=25) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(xs, *series, labels=keys, alpha=0.85)
    ax.set_xlabel("training step")
    ax.set_ylabel("weighted contribution to total reward")
    ax.set_title("Reward components over training (smoothed)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
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
    for k in keys:
        ys = _smooth([float(s.gate_counts.get(k, 0)) for s in steps], window=25)
        ax.plot(xs, ys, label=k, linewidth=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("trigger count per step (smoothed)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Gate triggers over training (lower is better)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5, which="both")
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
    """headline number — % of integrated cases solved (both schemes AND legal
    correctly identified at the bonus threshold)."""
    if not eval_results:
        _placeholder("integration solve rate", out)
        return
    labels = list(eval_results.keys())
    ys = []
    for label in labels:
        m = eval_results[label].get("integrated")
        ys.append(getattr(m, "pct_integrated_solved", 0.0) if m else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, ys, color="#2980b9")
    for bar, y in zip(bars, ys, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 1,
            f"{y:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("integration solve rate (%)")
    ax.set_xlabel("model")
    ax.set_ylim(0, max(100.0, max(ys) * 1.15) if ys else 100.0)
    ax.set_title("Integrated cases: % solved (headline metric)")
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
