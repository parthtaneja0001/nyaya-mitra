"""OpenEnv-compliant wrapper around NyayaMitraEnv.

Subclasses openenv.core.env_server.interfaces.Environment[NyayaAction, NyayaObservation, NyayaState]
so it integrates with OpenEnv's tooling (rubric introspection, gym-style API
verification, training infra discovery).

The internal NyayaMitraEnv keeps its richer interface (StepResult with info dict,
reward_breakdown). Other code in this repo — rollout, eval, training — talks to
that. This wrapper exists for OpenEnv conformance: it's what the server in
env/server.py serves and what judges' tooling will introspect.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import TypeAdapter

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.interface import (
    AdvisorAction,
)
from nyaya_mitra.knowledge.loader import KnowledgeBase

_ADVISOR_ADAPTER: TypeAdapter[AdvisorAction] = TypeAdapter(AdvisorAction)


class NyayaAction(Action):
    """OpenEnv Action envelope around our domain AdvisorAction.

    We carry the domain action as a nested dict to keep wire-format flexibility;
    the wrapper validates and discriminates on action.type internally.
    """

    advisor: dict[str, Any]


class NyayaObservation(Observation):
    """OpenEnv Observation envelope.

    `reward` (inherited) is set on terminal step from compute_reward total.
    `done` (inherited) flips True on terminal step.
    Citizen-side data lives under nyaya_mitra-prefixed fields so judges can
    inspect them without learning our Pydantic schema.
    """

    citizen_utterance: str = ""
    language: str = "en"
    turn: int = 0
    max_turns: int = 20
    elicited_facts: list[str] = []
    reward_breakdown: dict[str, float] | None = None


class NyayaState(State):
    """OpenEnv State envelope. inherits episode_id + step_count.

    Adds nothing beyond the parent; the rich state lives in NyayaMitraEnv's
    EpisodeState. We only expose minimal episode metadata here so accidental
    leaks of profile/ground-truth through state() never happen on the
    public surface.
    """

    sim_leak_count: int = 0
    elicited_facts: list[str] = []


class NyayaEnvironment(Environment[NyayaAction, NyayaObservation, NyayaState]):
    """OpenEnv Environment subclass. composes the existing NyayaMitraEnv internally.

    Construct with no args for the default config (real KB + smart citizen sim
    + fact extractor + no reward fn). Pass kwargs through to NyayaMitraEnv to
    override.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        *,
        kb: KnowledgeBase | None = None,
        sim: CitizenSimulator | None = None,
        extractor: FactExtractor | None = None,
        reward_fn: Any = None,
        shaping_fn: Any = None,
        max_turns: int = 20,
        rubric: Any = None,
        attach_default_rubric: bool = True,
    ) -> None:
        # default to the composable Rubric tree so judges' tooling sees it via env.rubric
        if rubric is None and attach_default_rubric:
            from nyaya_mitra.rewards.openenv_rubric import build_nyaya_rubric

            rubric = build_nyaya_rubric()
        super().__init__(rubric=rubric)
        self._inner = NyayaMitraEnv(
            kb=kb or KnowledgeBase(),
            sim=sim or CitizenSimulator(),
            extractor=extractor or FactExtractor(),
            reward_fn=reward_fn,
            shaping_fn=shaping_fn,
            max_turns=max_turns,
        )
        self._episode_id: str | None = None
        self._step_count: int = 0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> NyayaObservation:
        difficulty = kwargs.get("difficulty")
        seed = 0 if seed is None else int(seed)
        obs = self._inner.reset(seed=seed, difficulty=difficulty)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        return NyayaObservation(
            citizen_utterance=obs.citizen_utterance,
            language=obs.language,
            turn=obs.turn,
            max_turns=obs.max_turns,
            elicited_facts=list(obs.elicited_facts),
            done=False,
            reward=None,
            metadata={"episode_id": self._episode_id, "step_count": self._step_count},
        )

    def step(
        self,
        action: NyayaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> NyayaObservation:
        self._step_count += 1
        adv = _ADVISOR_ADAPTER.validate_python(action.advisor)
        result = self._inner.step(adv)

        if result.done:
            terminal_obs = NyayaObservation(
                citizen_utterance="",
                language="en",
                turn=int(result.info.get("turn", 0)),
                max_turns=int(result.info.get("max_turns", 20)),
                elicited_facts=list(result.info.get("elicited_facts") or []),
                reward_breakdown=dict(result.info.get("reward_breakdown") or {}),
                done=True,
                reward=float(result.reward),
                metadata={
                    "episode_id": self._episode_id,
                    "step_count": self._step_count,
                    "phase": result.info.get("phase"),
                    "truncated_by_env": bool(result.info.get("truncated_by_env", False)),
                    "sim_leak_count": int(result.info.get("sim_leak_count", 0)),
                },
            )
            # populate rubric last_score for OpenEnv introspection. we wire the
            # RewardContext into observation.metadata['reward_context'] so the
            # rubric tree's component leaves can read profile/plan/transcript.
            self._populate_rubric_last_scores(action, terminal_obs)
            return terminal_obs

        obs = result.observation
        assert obs is not None
        return NyayaObservation(
            citizen_utterance=obs.citizen_utterance,
            language=obs.language,
            turn=obs.turn,
            max_turns=obs.max_turns,
            elicited_facts=list(obs.elicited_facts),
            done=False,
            reward=None,
            metadata={
                "episode_id": self._episode_id,
                "step_count": self._step_count,
                "sim_leak": bool(result.info.get("sim_leak", False)),
                "negated_facts": list(result.info.get("negated_facts") or []),
            },
        )

    @property
    def state(self) -> NyayaState:
        """OpenEnv-compliant state: only safe-to-expose metadata.

        the rich internal state (profile, transcript, ground-truth) lives on
        NyayaMitraEnv and is locked behind NYAYA_DEBUG. this property is the
        public surface; we keep it minimal so episode_id + step_count + a
        couple of derived counts are available without ever leaking the
        ground-truth that the rubric is scoring against.
        """
        return NyayaState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            sim_leak_count=getattr(self._inner._state, "sim_leak_count", 0)
            if self._inner._state is not None
            else 0,
            elicited_facts=sorted(self._inner._state.elicited_facts)
            if self._inner._state is not None
            else [],
        )

    def close(self) -> None:
        self._inner.close()

    def _populate_rubric_last_scores(
        self, action: NyayaAction, terminal_obs: NyayaObservation
    ) -> None:
        """build a RewardContext from the inner env state and call the rubric so
        each component leaf records its last_score. introspection-only — does
        not change the reward written into observation.reward (which already
        came from compute_reward via reward_fn)."""
        if self.rubric is None:
            return
        inner = self._inner._state
        if inner is None or inner.profile is None:
            return
        # we don't have the plan as a typed object here without reparsing the
        # action; the inner env already validated it via the FINALIZE path,
        # and the reward_breakdown is the source of truth for downstream code.
        # the rubric introspection is best-effort: skip on any inner mismatch.
        try:
            from nyaya_mitra.interface import Finalize
            from nyaya_mitra.rewards.context import RewardContext
            from nyaya_mitra.rewards.kb_adapter import DuckTypedKB

            adv = _ADVISOR_ADAPTER.validate_python(action.advisor)
            if not isinstance(adv, Finalize):
                return
            ctx = RewardContext(
                profile=inner.profile,
                plan=adv.plan,
                transcript=[],  # rubric components that need transcript are best-effort
                elicited_facts=set(inner.elicited_facts),
                kb=DuckTypedKB(self._inner.kb),
                info={"max_turns": self._inner.max_turns},
            )
            scratch = NyayaObservation(
                **{**terminal_obs.model_dump(), "metadata": {"reward_context": ctx}}
            )
            self.rubric(action, scratch)
        except Exception:
            # introspection is best-effort; never let it break the episode
            pass


__all__ = [
    "NyayaAction",
    "NyayaEnvironment",
    "NyayaObservation",
    "NyayaState",
]
