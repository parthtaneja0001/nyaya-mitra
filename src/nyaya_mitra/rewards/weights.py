"""canonical weights for the soft component budget.

PLAN B.2 #3 (corrected decomposition). soft components sum to 1.0; HARM_PENALTY
is a separate negative additive (not in this dict). gates short-circuit total
to -1 when triggered.

DIGNITY_JUDGE is capped at 0.05 — see PLAN B.2 #1: no LLM-judged component
above 5%. all other components are deterministic and capped at 15%.
"""

from __future__ import annotations

from nyaya_mitra.interface.reward_keys import (
    DIGNITY_JUDGE,
    DOCUMENT_ACCURACY,
    FACT_COVERAGE,
    INTEGRATION_BONUS,
    LEGAL_PRECISION,
    LEGAL_RECALL,
    PROCEDURAL_CORRECTNESS,
    SCHEME_PRECISION,
    SCHEME_RECALL,
    SENSITIVITY_CORRECTNESS,
    TURN_EFFICIENCY,
)

WEIGHTS: dict[str, float] = {
    SCHEME_PRECISION: 0.10,
    SCHEME_RECALL: 0.10,
    LEGAL_PRECISION: 0.10,
    LEGAL_RECALL: 0.10,
    DOCUMENT_ACCURACY: 0.10,
    PROCEDURAL_CORRECTNESS: 0.10,
    FACT_COVERAGE: 0.12,
    INTEGRATION_BONUS: 0.15,
    SENSITIVITY_CORRECTNESS: 0.05,
    TURN_EFFICIENCY: 0.03,
    DIGNITY_JUDGE: 0.05,
}

GATE_FAIL_TOTAL = -1.0
DETERMINISTIC_CAP = 0.15
LLM_JUDGE_CAP = 0.05

LLM_JUDGE_KEYS: frozenset[str] = frozenset({DIGNITY_JUDGE})


def validate_weights() -> None:
    """invariants the aggregator and tests rely on. raises ValueError on violation."""
    total = sum(WEIGHTS.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"soft weights must sum to 1.0, got {total!r}")
    for k, w in WEIGHTS.items():
        cap = LLM_JUDGE_CAP if k in LLM_JUDGE_KEYS else DETERMINISTIC_CAP
        if w > cap + 1e-9:
            raise ValueError(f"{k} weight {w} exceeds cap {cap}")
        if w < 0:
            raise ValueError(f"{k} weight is negative")


validate_weights()
