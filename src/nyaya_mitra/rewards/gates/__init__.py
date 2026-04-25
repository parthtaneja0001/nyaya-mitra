"""hard gates short-circuit total to -1 when triggered. sim_leak is a soft passthrough
that adjusts shaping but never gates the total.
"""

from __future__ import annotations

from nyaya_mitra.rewards.gates.contradiction import check as check_contradiction
from nyaya_mitra.rewards.gates.format_validity import check as check_format
from nyaya_mitra.rewards.gates.hallucination import check as check_hallucination
from nyaya_mitra.rewards.gates.sim_leak_passthrough import (
    leaked_turn_indices as leaked_turn_indices,
)

__all__ = [
    "check_contradiction",
    "check_format",
    "check_hallucination",
    "leaked_turn_indices",
]
