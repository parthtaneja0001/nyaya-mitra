"""deterministic reward components + the capped llm-judged dignity component."""

from __future__ import annotations

from nyaya_mitra.rewards.components.dignity_judge import compute as compute_dignity_judge
from nyaya_mitra.rewards.components.document_accuracy import compute as compute_document_accuracy
from nyaya_mitra.rewards.components.fact_coverage import compute as compute_fact_coverage
from nyaya_mitra.rewards.components.harm_penalty import compute as compute_harm_penalty
from nyaya_mitra.rewards.components.integration_bonus import compute as compute_integration_bonus
from nyaya_mitra.rewards.components.legal_precision import compute as compute_legal_precision
from nyaya_mitra.rewards.components.legal_recall import compute as compute_legal_recall
from nyaya_mitra.rewards.components.procedural_correctness import (
    compute as compute_procedural_correctness,
)
from nyaya_mitra.rewards.components.scheme_precision import compute as compute_scheme_precision
from nyaya_mitra.rewards.components.scheme_recall import compute as compute_scheme_recall
from nyaya_mitra.rewards.components.sensitivity_correctness import (
    compute as compute_sensitivity_correctness,
)
from nyaya_mitra.rewards.components.turn_efficiency import compute as compute_turn_efficiency

__all__ = [
    "compute_dignity_judge",
    "compute_document_accuracy",
    "compute_fact_coverage",
    "compute_harm_penalty",
    "compute_integration_bonus",
    "compute_legal_precision",
    "compute_legal_recall",
    "compute_procedural_correctness",
    "compute_scheme_precision",
    "compute_scheme_recall",
    "compute_sensitivity_correctness",
    "compute_turn_efficiency",
]
