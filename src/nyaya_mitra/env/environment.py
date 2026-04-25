"""nyaya mitra env. gym-style reset/step/state/close."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.episode_state import EpisodeState, TurnRecord
from nyaya_mitra.interface import (
    ALL_KEYS,
    TOTAL,
    ActionPlan,
    AdvisorAction,
    CitizenObservation,
    CitizenProfile,
    Finalize,
    Probe,
)
from nyaya_mitra.knowledge.loader import KnowledgeBase
from nyaya_mitra.profile.derivation import load_profile

RewardFn = Callable[
    [CitizenProfile, ActionPlan, list[TurnRecord], set[str]],
    dict[str, float],
]

ShapingFn = Callable[
    [int, AdvisorAction, list[str], bool, str],
    dict[str, float],
]

_SENSITIVE_FACT_PREFIXES = (
    "caste_",
    "dv_",
    "disability_",
    "immigration_",
    "hiv_",
    "orientation_",
    "mental_",
)


@dataclass
class StepResult:
    observation: CitizenObservation | None
    reward: float
    done: bool
    info: dict[str, Any]


class NyayaMitraEnv:
    def __init__(
        self,
        kb: KnowledgeBase,
        sim: CitizenSimulator,
        extractor: FactExtractor,
        reward_fn: RewardFn | None = None,
        shaping_fn: ShapingFn | None = None,
        max_turns: int = 20,
    ) -> None:
        self.kb = kb
        self.sim = sim
        self.extractor = extractor
        self.reward_fn = reward_fn
        self.shaping_fn = shaping_fn
        self.max_turns = max_turns
        self._state: EpisodeState | None = None

    def reset(self, seed: int = 0, difficulty: str | None = None) -> CitizenObservation:
        profile = load_profile(seed=seed, difficulty=difficulty, kb=self.kb)
        self._state = EpisodeState(profile=profile, max_turns=self.max_turns)
        utterance = self.sim.initial_utterance(profile)
        revealed = self.extractor.extract(profile, utterance, self._state.elicited_facts)
        negated = self.extractor.extract_negations(profile, utterance)
        self._state.elicited_facts.update(revealed)
        self._state.negated_facts.update(negated)
        self._state.transcript.append(
            TurnRecord(
                actor="citizen",
                payload={"utterance": utterance},
                revealed=revealed,
                negated=negated,
            )
        )
        return self._observation(revealed)

    def step(self, action: AdvisorAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._state.done:
            raise RuntimeError("episode already done; call reset()")

        self._state.transcript.append(TurnRecord(actor="advisor", payload=action.model_dump()))
        self._state.turn += 1

        if isinstance(action, Finalize):
            return self._terminal(action.plan, truncated=False)

        if self._state.turn >= self.max_turns:
            return self._terminal(plan=None, truncated=True)

        utterance = self.sim.respond(
            self._state.profile,
            prior_transcript=self._state.transcript,
            advisor_action=action,
        )
        revealed = self.extractor.extract(
            self._state.profile, utterance, self._state.elicited_facts
        )
        negated = self.extractor.extract_negations(self._state.profile, utterance)
        sim_leak = self._detect_sim_leak(action, revealed)
        if sim_leak:
            self._state.sim_leak_count += 1

        self._state.elicited_facts.update(revealed)
        self._state.negated_facts.update(negated)

        if self.shaping_fn is not None:
            delta = self.shaping_fn(
                self._state.turn,
                action,
                revealed,
                sim_leak,
                self._state.profile.behavior.literacy,
            )
            for k, v in delta.items():
                self._state.shaping_running[k] = self._state.shaping_running.get(k, 0.0) + v

        self._state.transcript.append(
            TurnRecord(
                actor="citizen",
                payload={"utterance": utterance, "sim_leak": sim_leak},
                revealed=revealed,
                negated=negated,
            )
        )

        return StepResult(
            observation=self._observation(revealed),
            reward=0.0,
            done=False,
            info={
                "elicited_facts": sorted(self._state.elicited_facts),
                "negated_facts": sorted(negated),
                "turn": self._state.turn,
                "max_turns": self.max_turns,
                "phase": "ongoing",
                "sim_leak": sim_leak,
            },
        )

    def state(self) -> dict[str, Any]:
        if not os.environ.get("NYAYA_DEBUG"):
            raise RuntimeError("state() requires NYAYA_DEBUG=1")
        if self._state is None:
            return {}
        return {
            "profile": self._state.profile.model_dump(),
            "turn": self._state.turn,
            "max_turns": self._state.max_turns,
            "done": self._state.done,
            "elicited_facts": sorted(self._state.elicited_facts),
            "negated_facts": sorted(self._state.negated_facts),
            "shaping_running": dict(self._state.shaping_running),
            "sim_leak_count": self._state.sim_leak_count,
            "transcript": [
                {
                    "actor": t.actor,
                    "payload": t.payload,
                    "revealed": t.revealed,
                    "negated": t.negated,
                }
                for t in self._state.transcript
            ],
        }

    def close(self) -> None:
        self._state = None

    def _observation(self, revealed_this_turn: list[str]) -> CitizenObservation:
        s = self._state
        assert s is not None
        last_citizen = next((t for t in reversed(s.transcript) if t.actor == "citizen"), None)
        utt = last_citizen.payload.get("utterance", "") if last_citizen else ""
        return CitizenObservation(
            citizen_utterance=utt,
            language=s.profile.behavior.language_preference,
            turn=s.turn,
            max_turns=s.max_turns,
            elicited_facts=sorted(s.elicited_facts),
            facts_revealed_this_turn=revealed_this_turn,
        )

    def _terminal(self, plan: ActionPlan | None, truncated: bool) -> StepResult:
        s = self._state
        assert s is not None
        s.done = True
        if plan is None or self.reward_fn is None:
            breakdown: dict[str, float] = {k: 0.0 for k in ALL_KEYS}
        else:
            breakdown = dict(self.reward_fn(s.profile, plan, s.transcript, s.elicited_facts))
            for k in ALL_KEYS:
                breakdown.setdefault(k, 0.0)
        total = float(breakdown.get(TOTAL, 0.0))
        return StepResult(
            observation=None,
            reward=total,
            done=True,
            info={
                "elicited_facts": sorted(s.elicited_facts),
                "negated_facts": sorted(s.negated_facts),
                "turn": s.turn,
                "max_turns": self.max_turns,
                "phase": "terminal",
                "truncated_by_env": truncated,
                "format_violation": False,
                "shaping_running": dict(s.shaping_running),
                "reward_breakdown": breakdown,
                "sim_leak_count": s.sim_leak_count,
            },
        )

    def _detect_sim_leak(self, action: AdvisorAction, revealed: list[str]) -> bool:
        sensitive = [f for f in revealed if any(f.startswith(p) for p in _SENSITIVE_FACT_PREFIXES)]
        if not sensitive:
            return False
        return not isinstance(action, Probe)
