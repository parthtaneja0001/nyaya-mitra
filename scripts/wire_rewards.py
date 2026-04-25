"""bootstrap helper: build a NyayaMitraEnv wired to track-b's reward + shaping fns.

usage:
    from scripts.wire_rewards import build_env

    env = build_env(seed=0)
    obs = env.reset()
    ...

cross-track import note: scripts/ is exempt from the contract-test cross-track
scan, so this file is the only place that legitimately imports from both
nyaya_mitra.env and nyaya_mitra.rewards. neither side imports the other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.knowledge.loader import KnowledgeBase
from nyaya_mitra.rewards import compute_shaping, make_env_reward_fn
from nyaya_mitra.rewards.kb_adapter import DuckTypedKB


def build_env(
    *,
    max_turns: int = 20,
    relevant_facts: dict[str, set[str]] | None = None,
    extra_info: Callable[[Any], dict[str, Any]] | None = None,
) -> NyayaMitraEnv:
    """construct a NyayaMitraEnv with reward_fn and shaping_fn already wired.

    relevant_facts: optional override for the kb_adapter's default mapping.
    extra_info: optional hook the reward fn calls at terminal step (see
                rewards.aggregator.make_env_reward_fn for the contract).
    """
    kb = KnowledgeBase()
    adapter = DuckTypedKB(kb, relevant_facts=relevant_facts)
    reward_fn = make_env_reward_fn(adapter, extra_info=extra_info, max_turns=max_turns)

    return NyayaMitraEnv(
        kb=kb,
        sim=CitizenSimulator(),
        extractor=FactExtractor(),
        reward_fn=reward_fn,
        shaping_fn=compute_shaping,
        max_turns=max_turns,
    )


__all__ = ["build_env"]
