"""episode runner. drives a callable advisor through one full episode against
the wired env, collects the transcript, and returns an EpisodeResult.

design notes
============
- Advisor protocol is just `Callable[[CitizenObservation, RolloutState], AdvisorAction]`.
  this decouples rollout from the model surface — the same runner is used for
  scripted advisors (tests, baselines) and trained policies (training script).

- RolloutState carries everything the advisor might want for prompt construction:
  cumulative observation history, accumulated elicited_facts, turn count,
  per-turn info dicts. it is constructed by rollout, not by the advisor.

- the runner never inspects the action — env.step does the validation. malformed
  actions result in done=True/reward=-1 from the env (gate fires server-side).

- failures are surfaced as `EpisodeResult.error`; they don't raise. the trainer
  decides whether to skip or fail the run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from nyaya_mitra.interface import (
    AdvisorAction,
    CitizenObservation,
    Finalize,
)


class _StepResult(Protocol):
    observation: CitizenObservation | None
    reward: float
    done: bool
    info: dict[str, Any]


class _Env(Protocol):
    """duck-typed env surface. matches track-a's NyayaMitraEnv without importing it."""

    def reset(self, seed: int = 0, difficulty: str | None = None) -> CitizenObservation: ...

    def step(self, action: AdvisorAction) -> _StepResult: ...

    def close(self) -> None: ...


@dataclass
class TurnLog:
    """one advisor exchange. the citizen response that follows is recorded as the
    next observation in the *next* TurnLog (or terminal observation=None)."""

    turn_index: int
    observation_in: CitizenObservation
    action: AdvisorAction
    info: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    error: str | None = None


@dataclass
class EpisodeResult:
    seed: int
    difficulty: str | None
    turns: list[TurnLog]
    final_breakdown: dict[str, float]
    total_reward: float
    finalized: bool
    truncated_by_env: bool
    elicited_facts: list[str]
    sim_leak_count: int
    wall_seconds: float
    error: str | None = None


@dataclass
class RolloutState:
    """visible to the advisor on each call. immutable from the advisor's perspective."""

    seed: int
    turn_index: int
    max_turns: int
    elicited_facts: list[str]
    history: list[CitizenObservation]
    last_info: dict[str, Any]


Advisor = Callable[[CitizenObservation, RolloutState], AdvisorAction]


def run_episode(
    env: _Env,
    advisor: Advisor,
    *,
    seed: int = 0,
    difficulty: str | None = None,
) -> EpisodeResult:
    """run one episode end-to-end. always returns; never raises on advisor errors."""
    started = time.perf_counter()
    turns: list[TurnLog] = []
    history: list[CitizenObservation] = []
    final_breakdown: dict[str, float] = {}
    total = 0.0
    finalized = False
    truncated = False
    sim_leak = 0
    elicited: list[str] = []
    error: str | None = None

    try:
        obs = env.reset(seed=seed, difficulty=difficulty)
    except Exception as exc:
        return EpisodeResult(
            seed=seed,
            difficulty=difficulty,
            turns=[],
            final_breakdown={},
            total_reward=0.0,
            finalized=False,
            truncated_by_env=False,
            elicited_facts=[],
            sim_leak_count=0,
            wall_seconds=time.perf_counter() - started,
            error=f"reset failed: {exc!r}",
        )
    history.append(obs)
    last_info: dict[str, Any] = {}

    while True:
        state = RolloutState(
            seed=seed,
            turn_index=obs.turn,
            max_turns=obs.max_turns,
            elicited_facts=list(obs.elicited_facts),
            history=list(history),
            last_info=dict(last_info),
        )
        try:
            action = advisor(obs, state)
        except Exception as exc:
            error = f"advisor raised on turn {obs.turn}: {exc!r}"
            break

        try:
            res = env.step(action)
        except Exception as exc:
            error = f"env.step raised on turn {obs.turn}: {exc!r}"
            break

        turn_log = TurnLog(
            turn_index=obs.turn,
            observation_in=obs,
            action=action,
            info=dict(res.info or {}),
            reward=float(res.reward),
            done=bool(res.done),
        )
        turns.append(turn_log)
        last_info = dict(res.info or {})

        if res.done:
            total = float(res.reward)
            final_breakdown = dict(last_info.get("reward_breakdown") or {})
            finalized = isinstance(action, Finalize) and not last_info.get(
                "truncated_by_env", False
            )
            truncated = bool(last_info.get("truncated_by_env", False))
            sim_leak = int(last_info.get("sim_leak_count", 0))
            elicited = list(last_info.get("elicited_facts") or [])
            break

        if res.observation is None:
            error = f"env returned no observation on continuing turn {obs.turn}"
            break

        obs = res.observation
        history.append(obs)

    return EpisodeResult(
        seed=seed,
        difficulty=difficulty,
        turns=turns,
        final_breakdown=final_breakdown,
        total_reward=total,
        finalized=finalized,
        truncated_by_env=truncated,
        elicited_facts=elicited,
        sim_leak_count=sim_leak,
        wall_seconds=time.perf_counter() - started,
        error=error,
    )


def run_episodes(
    env_factory: Callable[[], _Env],
    advisor: Advisor,
    *,
    seeds: list[int],
    difficulty: str | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    """run a batch. each episode gets a fresh env from env_factory so state from
    the previous episode never leaks (matters once we add LLM-side caching).
    on_episode fires after each episode; useful for streaming logs to W&B."""
    out: list[EpisodeResult] = []
    for seed in seeds:
        env = env_factory()
        try:
            result = run_episode(env, advisor, seed=seed, difficulty=difficulty)
        finally:
            env.close()
        out.append(result)
        if on_episode is not None:
            try:
                on_episode(result)
            except Exception:
                pass
    return out


__all__ = [
    "Advisor",
    "EpisodeResult",
    "RolloutState",
    "TurnLog",
    "run_episode",
    "run_episodes",
]
