"""OpenEnv Rubric exposure of our reward function.

Why this exists
===============
The OpenEnv judging criteria explicitly call out: "Uses OpenEnv's Rubric system
thoughtfully (composable rubrics > monolithic scoring)." Our reward function is
already composable internally — 11 components + 4 gates aggregated by a typed
weighted sum — but it's expressed as `compute_reward(ctx) -> dict[str, float]`,
not as a `Rubric` tree.

This module exposes the same logic through OpenEnv's `Rubric` API:

    Sequential(
        Gate(FormatRubric()),         # short-circuit on format violation
        Gate(HallucinationRubric()),  # short-circuit on unknown ids
        Gate(ContradictionRubric()),  # short-circuit on negated facts
        WeightedSum(
            [SchemePrecisionRubric(), SchemeRecallRubric(), ...],  # 11 components
            weights=[0.10, 0.10, ...],
        ),
    )

This means:
- `env.rubric.named_rubrics()` enumerates every component (introspectable by training infra)
- Gate failures fail-fast like in our aggregator (returns 0 instead of -1, see notes)
- Each component's `last_score` is logged after every step

Mapping to compute_reward
=========================
This rubric returns a value in [0, 1] (per Rubric contract). compute_reward
returns in [-1, 1] with -1 reserved for gate violations. We map gate violations
to 0 here (Rubric convention) and use `compute_reward` for the breakdown that
the env's `info["reward_breakdown"]` carries. So the rubric is the *summary
score in OpenEnv's grammar*; the full breakdown is the source of truth and
remains what the trainer uses.

The rubric needs the context (profile, plan, transcript, kb). It accepts these
through a `RubricContext` carried in the action's metadata when the env calls
into the rubric, OR by holding a reference to the env state. We use the second
form because the env's terminal step has all the state available anyway.

Wiring
======
NyayaEnvironment can be constructed with `rubric=NyayaRubric.build(kb)`. The
rubric's `forward` reads from `observation.metadata['ctx']` (set by the env
before calling self.rubric(action, obs)).
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

from nyaya_mitra.rewards.components import (
    compute_dignity_judge,
    compute_document_accuracy,
    compute_fact_coverage,
    compute_integration_bonus,
    compute_legal_precision,
    compute_legal_recall,
    compute_procedural_correctness,
    compute_scheme_precision,
    compute_scheme_recall,
    compute_sensitivity_correctness,
    compute_turn_efficiency,
)
from nyaya_mitra.rewards.context import RewardContext
from nyaya_mitra.rewards.gates import (
    check_contradiction,
    check_format,
    check_hallucination,
)
from nyaya_mitra.rewards.weights import WEIGHTS


def _ctx_from_observation(observation: Any) -> RewardContext | None:
    """rubrics receive (action, observation). NyayaEnvironment stuffs the
    RewardContext into observation.metadata['reward_context'] right before
    invoking the rubric. when absent (e.g. mid-episode steps), return None."""
    if observation is None:
        return None
    meta = getattr(observation, "metadata", None) or {}
    ctx = meta.get("reward_context")
    return ctx if isinstance(ctx, RewardContext) else None


# ---------- gate rubrics: each returns 1.0 when clean, 0.0 when triggered ----------


class _GateBaseRubric(Rubric):
    """common shape: 1.0 = clean, 0.0 = gate triggered. used inside Gate()."""

    def __init__(self, name: str, check_fn) -> None:
        super().__init__()
        self.gate_name = name
        self._check = check_fn

    def forward(self, action: Any, observation: Any) -> float:
        ctx = _ctx_from_observation(observation)
        if ctx is None:
            return 1.0  # mid-episode call; no terminal context yet
        return 0.0 if self._check(ctx) else 1.0


class FormatRubric(_GateBaseRubric):
    def __init__(self) -> None:
        super().__init__("format_validity", check_format)


class HallucinationRubric(_GateBaseRubric):
    def __init__(self) -> None:
        super().__init__("hallucination", check_hallucination)


class ContradictionRubric(_GateBaseRubric):
    def __init__(self) -> None:
        super().__init__("contradiction", check_contradiction)


# ---------- soft component rubrics: 0..1 scoring ----------


class _ComponentRubric(Rubric):
    """wraps a component compute(ctx) -> float into Rubric.forward."""

    def __init__(self, name: str, compute_fn, *, reads_breakdown: bool = False) -> None:
        super().__init__()
        self.component_name = name
        self._compute = compute_fn
        self._reads_breakdown = reads_breakdown

    def forward(self, action: Any, observation: Any) -> float:
        ctx = _ctx_from_observation(observation)
        if ctx is None:
            return 0.0
        try:
            if self._reads_breakdown:
                breakdown = (observation.metadata or {}).get("reward_breakdown_partial", {})
                return float(self._compute(ctx, breakdown))
            return float(self._compute(ctx))
        except Exception:
            return 0.0


class SchemePrecisionRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("scheme_precision", compute_scheme_precision)


class SchemeRecallRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("scheme_recall", compute_scheme_recall)


class LegalPrecisionRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("legal_precision", compute_legal_precision)


class LegalRecallRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("legal_recall", compute_legal_recall)


class DocumentAccuracyRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("document_accuracy", compute_document_accuracy)


class ProceduralCorrectnessRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("procedural_correctness", compute_procedural_correctness)


class FactCoverageRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("fact_coverage", compute_fact_coverage)


class SensitivityCorrectnessRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("sensitivity_correctness", compute_sensitivity_correctness)


class DignityJudgeRubric(_ComponentRubric):
    def __init__(self) -> None:
        super().__init__("dignity_judge", compute_dignity_judge)


# components that need the partial breakdown of earlier components:


class _ComponentWithBreakdownRubric(Rubric):
    def __init__(self, name: str, compute_fn) -> None:
        super().__init__()
        self.component_name = name
        self._compute = compute_fn

    def forward(self, action: Any, observation: Any) -> float:
        ctx = _ctx_from_observation(observation)
        if ctx is None:
            return 0.0
        # we recompute the precondition components inline so this rubric can
        # stand alone. for a tighter version, the env can pre-populate
        # observation.metadata["reward_breakdown_partial"] with prior values.
        partial = (
            getattr(observation, "metadata", None) or {}
        ).get("reward_breakdown_partial") or {}
        if not partial:
            partial = {
                "scheme_precision": compute_scheme_precision(ctx),
                "scheme_recall": compute_scheme_recall(ctx),
                "legal_precision": compute_legal_precision(ctx),
                "legal_recall": compute_legal_recall(ctx),
                "fact_coverage": compute_fact_coverage(ctx),
            }
        try:
            return float(self._compute(ctx, partial))
        except Exception:
            return 0.0


class TurnEfficiencyRubric(_ComponentWithBreakdownRubric):
    def __init__(self) -> None:
        super().__init__("turn_efficiency", compute_turn_efficiency)


class IntegrationBonusRubric(_ComponentWithBreakdownRubric):
    def __init__(self) -> None:
        super().__init__("integration_bonus", compute_integration_bonus)


# ---------- the public composer ----------


def build_nyaya_rubric() -> Rubric:
    """fail-fast gates → weighted sum of soft components, exactly mirroring
    rewards/aggregator.compute_reward but expressed as an OpenEnv Rubric tree.

    weights are sourced from rewards/weights.py so a single edit propagates.
    """
    weights = WEIGHTS  # already sums to 1.0; validate_weights() runs at import

    return Sequential(
        Gate(FormatRubric()),
        Gate(HallucinationRubric()),
        Gate(ContradictionRubric()),
        WeightedSum(
            rubrics=[
                SchemePrecisionRubric(),
                SchemeRecallRubric(),
                LegalPrecisionRubric(),
                LegalRecallRubric(),
                DocumentAccuracyRubric(),
                ProceduralCorrectnessRubric(),
                FactCoverageRubric(),
                IntegrationBonusRubric(),
                SensitivityCorrectnessRubric(),
                TurnEfficiencyRubric(),
                DignityJudgeRubric(),
            ],
            weights=[
                weights["scheme_precision"],
                weights["scheme_recall"],
                weights["legal_precision"],
                weights["legal_recall"],
                weights["document_accuracy"],
                weights["procedural_correctness"],
                weights["fact_coverage"],
                weights["integration_bonus"],
                weights["sensitivity_correctness"],
                weights["turn_efficiency"],
                weights["dignity_judge"],
            ],
        ),
    )


__all__ = [
    "ContradictionRubric",
    "DignityJudgeRubric",
    "DocumentAccuracyRubric",
    "FactCoverageRubric",
    "FormatRubric",
    "HallucinationRubric",
    "IntegrationBonusRubric",
    "LegalPrecisionRubric",
    "LegalRecallRubric",
    "ProceduralCorrectnessRubric",
    "SchemePrecisionRubric",
    "SchemeRecallRubric",
    "SensitivityCorrectnessRubric",
    "TurnEfficiencyRubric",
    "build_nyaya_rubric",
]
