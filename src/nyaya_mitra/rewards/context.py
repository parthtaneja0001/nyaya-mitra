"""reward context — bundles everything components need.

decoupled from track A's env types so the reward function stays pure and
easily testable. track A constructs a RewardContext at terminal step and
passes it to compute_reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nyaya_mitra.interface import (
    ActionPlan,
    AdvisorAction,
    CitizenObservation,
    CitizenProfile,
)
from nyaya_mitra.rewards.kb_protocol import KnowledgeBase


@dataclass(frozen=True)
class Turn:
    """one transcript entry. action is set on advisor turns; citizen_observation
    on citizen turns. shaping/sensitivity components iterate and filter by type.
    info carries per-turn signals from the env (sim_leak, etc.)."""

    index: int
    action: AdvisorAction | None
    citizen_observation: CitizenObservation | None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardContext:
    """terminal-step bundle. components and gates read from this; never mutate."""

    profile: CitizenProfile
    plan: ActionPlan
    transcript: list[Turn]
    elicited_facts: set[str]
    kb: KnowledgeBase
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def max_turns(self) -> int:
        return int(self.info.get("max_turns", 20))

    @property
    def turns_used(self) -> int:
        """count of advisor turns in the transcript."""
        return sum(1 for t in self.transcript if t.action is not None)

    @property
    def format_violation(self) -> bool:
        return bool(self.info.get("format_violation", False))

    @property
    def truncated_by_env(self) -> bool:
        return bool(self.info.get("truncated_by_env", False))
