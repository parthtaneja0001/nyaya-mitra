"""diversity tracker. penalizes the generator for emitting profiles too similar
to recent emissions.

similarity = jaccard between key-token sets of (presenting_issue + situation_specific
hidden+sensitive fact keys + occupation + state). cheap, no embeddings dep.

state is per-instance and bounded (default last 50). callers mutate via
record(profile_dict) and read penalty() on the candidate before recording.
"""

from __future__ import annotations

import re
from collections import deque

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(s.lower()))


def _signature_tokens(profile: dict) -> set[str]:
    out: set[str] = set()
    sit = profile.get("situation_specific") or {}
    out |= _tokens(str(sit.get("presenting_issue") or ""))
    out |= {"hf:" + k.lower() for k in (sit.get("hidden_facts") or {})}
    out |= {"sf:" + k.lower() for k in (sit.get("sensitive_facts") or {})}
    econ = profile.get("economic") or {}
    out |= _tokens(str(econ.get("occupation") or ""))
    demo = profile.get("demographics") or {}
    out |= _tokens(str(demo.get("state") or ""))
    out |= _tokens(str(demo.get("residence") or ""))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DiversityTracker:
    def __init__(self, *, window: int = 50, weight: float = 0.5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._weight = weight
        self._recent: deque[set[str]] = deque(maxlen=window)

    @property
    def window(self) -> int:
        return self._window

    @property
    def size(self) -> int:
        return len(self._recent)

    def max_similarity(self, profile: dict) -> float:
        sig = _signature_tokens(profile)
        if not self._recent:
            return 0.0
        return max(_jaccard(sig, prev) for prev in self._recent)

    def penalty(self, profile: dict) -> float:
        """non-positive scalar to add to the generator reward."""
        return -self._weight * self.max_similarity(profile)

    def record(self, profile: dict) -> None:
        self._recent.append(_signature_tokens(profile))


__all__ = ["DiversityTracker"]
