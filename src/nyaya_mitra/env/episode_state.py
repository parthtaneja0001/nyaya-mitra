"""mutable per-episode state. lives only inside the env process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nyaya_mitra.interface import CitizenProfile


@dataclass
class TurnRecord:
    actor: str
    payload: dict[str, Any]
    revealed: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)


@dataclass
class EpisodeState:
    profile: CitizenProfile
    max_turns: int = 20
    turn: int = 0
    elicited_facts: set[str] = field(default_factory=set)
    negated_facts: set[str] = field(default_factory=set)
    transcript: list[TurnRecord] = field(default_factory=list)
    shaping_running: dict[str, float] = field(default_factory=dict)
    done: bool = False
    sim_leak_count: int = 0
