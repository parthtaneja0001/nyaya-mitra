"""minimal llm chat protocol shared by vanilla + prompted baselines.

we don't import transformers/torch here. the protocol is a pure callable:

    LLMChat = Callable[[list[ChatMessage], dict | None], str]

implementations live in eval.baselines.runtime (gpu, optional). tests use a
fake implementation that returns deterministic strings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


LLMChat = Callable[[list[ChatMessage], dict | None], str]


class FakeChat:
    """deterministic stand-in for an LLM. round-robins through scripted replies.

    used in tests so the baseline modules can be exercised without a model. each
    call advances the cursor; reset() rewinds.
    """

    def __init__(self, replies: list[str]) -> None:
        if not replies:
            raise ValueError("FakeChat needs at least one reply")
        self._replies = list(replies)
        self._cursor = 0
        self.calls: list[list[ChatMessage]] = []

    def __call__(self, messages: list[ChatMessage], options: dict | None = None) -> str:
        self.calls.append(list(messages))
        out = self._replies[self._cursor % len(self._replies)]
        self._cursor += 1
        return out

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()


__all__ = ["ChatMessage", "FakeChat", "LLMChat", "Role"]
