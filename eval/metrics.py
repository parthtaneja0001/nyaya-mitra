"""eval metrics. computed from a list of EpisodeResult, grouped by cohort.

cohort = welfare_only | legal_only | integrated. determined by the eval-case
filename prefix (wel_/leg_/int_) or by inspecting derived_ground_truth on the
profile that produced each episode.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from nyaya_mitra.interface.reward_keys import (
    GATE_CONTRADICTION,
    GATE_FORMAT_VIOLATION,
    GATE_HALLUCINATION,
    GATE_SIM_LEAK,
    INTEGRATION_BONUS,
    LEGAL_PRECISION,
    LEGAL_RECALL,
    SCHEME_PRECISION,
    SCHEME_RECALL,
    SENSITIVITY_CORRECTNESS,
    TURN_EFFICIENCY,
)
from training.rollout import EpisodeResult


@dataclass
class CohortMetrics:
    cohort: str
    n: int
    mean_total_reward: float
    median_total_reward: float
    p25_total_reward: float
    p75_total_reward: float
    pct_finalized: float
    pct_truncated: float
    mean_turns: float
    pct_all_gates_passed: float
    pct_integrated_solved: float  # only meaningful for integrated cohort
    mean_sensitivity_correctness: float
    mean_scheme_precision: float
    mean_scheme_recall: float
    mean_legal_precision: float
    mean_legal_recall: float
    mean_turn_efficiency: float
    mean_sim_leak_count: float
    gate_trigger_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class HarnessReport:
    model_label: str
    per_cohort: dict[str, CohortMetrics]
    overall: CohortMetrics


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = k - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _gate_passed(b: dict[str, float]) -> bool:
    """all hard gates clean."""
    return (
        b.get(GATE_FORMAT_VIOLATION, 0.0) == 0.0
        and b.get(GATE_HALLUCINATION, 0.0) == 0.0
        and b.get(GATE_CONTRADICTION, 0.0) == 0.0
    )


def _integrated_solved(b: dict[str, float]) -> bool:
    """integration bonus is binary {0,1}; non-zero means both sides cleared 0.5."""
    return b.get(INTEGRATION_BONUS, 0.0) > 0.5


def cohort_metrics(cohort: str, results: Iterable[EpisodeResult]) -> CohortMetrics:
    rs = [r for r in results if r.error is None]
    if not rs:
        return CohortMetrics(
            cohort=cohort,
            n=0,
            mean_total_reward=0.0,
            median_total_reward=0.0,
            p25_total_reward=0.0,
            p75_total_reward=0.0,
            pct_finalized=0.0,
            pct_truncated=0.0,
            mean_turns=0.0,
            pct_all_gates_passed=0.0,
            pct_integrated_solved=0.0,
            mean_sensitivity_correctness=0.0,
            mean_scheme_precision=0.0,
            mean_scheme_recall=0.0,
            mean_legal_precision=0.0,
            mean_legal_recall=0.0,
            mean_turn_efficiency=0.0,
            mean_sim_leak_count=0.0,
            gate_trigger_counts={
                GATE_FORMAT_VIOLATION: 0,
                GATE_HALLUCINATION: 0,
                GATE_CONTRADICTION: 0,
                GATE_SIM_LEAK: 0,
            },
        )

    totals = [r.total_reward for r in rs]
    finalized = sum(1 for r in rs if r.finalized)
    truncated = sum(1 for r in rs if r.truncated_by_env)
    breakdowns = [r.final_breakdown for r in rs]

    gate_counts = {
        GATE_FORMAT_VIOLATION: sum(
            1 for b in breakdowns if b.get(GATE_FORMAT_VIOLATION, 0.0) > 0.0
        ),
        GATE_HALLUCINATION: sum(1 for b in breakdowns if b.get(GATE_HALLUCINATION, 0.0) > 0.0),
        GATE_CONTRADICTION: sum(1 for b in breakdowns if b.get(GATE_CONTRADICTION, 0.0) > 0.0),
        GATE_SIM_LEAK: sum(1 for b in breakdowns if b.get(GATE_SIM_LEAK, 0.0) > 0.0),
    }

    return CohortMetrics(
        cohort=cohort,
        n=len(rs),
        mean_total_reward=_safe_mean(totals),
        median_total_reward=_percentile(totals, 0.5),
        p25_total_reward=_percentile(totals, 0.25),
        p75_total_reward=_percentile(totals, 0.75),
        pct_finalized=100.0 * finalized / len(rs),
        pct_truncated=100.0 * truncated / len(rs),
        mean_turns=_safe_mean([float(len(r.turns)) for r in rs]),
        pct_all_gates_passed=100.0 * sum(1 for b in breakdowns if _gate_passed(b)) / len(rs),
        pct_integrated_solved=100.0 * sum(1 for b in breakdowns if _integrated_solved(b)) / len(rs),
        mean_sensitivity_correctness=_safe_mean(
            [b.get(SENSITIVITY_CORRECTNESS, 0.0) for b in breakdowns]
        ),
        mean_scheme_precision=_safe_mean([b.get(SCHEME_PRECISION, 0.0) for b in breakdowns]),
        mean_scheme_recall=_safe_mean([b.get(SCHEME_RECALL, 0.0) for b in breakdowns]),
        mean_legal_precision=_safe_mean([b.get(LEGAL_PRECISION, 0.0) for b in breakdowns]),
        mean_legal_recall=_safe_mean([b.get(LEGAL_RECALL, 0.0) for b in breakdowns]),
        mean_turn_efficiency=_safe_mean([b.get(TURN_EFFICIENCY, 0.0) for b in breakdowns]),
        mean_sim_leak_count=_safe_mean([float(r.sim_leak_count) for r in rs]),
        gate_trigger_counts=gate_counts,
    )


def overall_from_episodes(results: Iterable[EpisodeResult]) -> CohortMetrics:
    """flatten across cohorts by recomputing on the union of episodes. avoids
    weighted-mean reconstruction (which loses median/percentile fidelity)."""
    return cohort_metrics("overall", results)


__all__ = [
    "CohortMetrics",
    "HarnessReport",
    "cohort_metrics",
    "overall_from_episodes",
]
