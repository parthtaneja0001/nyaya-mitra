"""vanilla baseline: minimal system prompt, no KB context.

the honest "what does the base model do without help?" lower bound. uses the
LLMChat protocol so it works with any backend (HF transformers, OpenAI-compat,
fakes). the system prompt only describes the action schema; it does not list
schemes, frameworks, or the citizen's profile.
"""

from __future__ import annotations

from collections.abc import Callable

from eval.baselines.action_parser import parse_action
from eval.baselines.llm_protocol import ChatMessage, LLMChat

VANILLA_SYSTEM = """You are an Indian welfare paralegal. Talk to the citizen and \
help them. After collecting facts, output a final ActionPlan with applicable schemes \
and legal routes plus free legal aid contacts.

On every turn output a single JSON object, nothing else:
- {"type":"ASK","question":"...","language":"en|hi|hinglish"}
- {"type":"PROBE","question":"...","sensitive_topic":"caste|dv|disability|immigration|hiv_status|sexual_orientation|mental_health","language":"..."}
- {"type":"EXPLAIN","content":"...","target_literacy":"low|medium|high","language":"..."}
- {"type":"FINALIZE","plan":{"schemes":[...],"legal_routes":[...],"most_important_next_step":"...","plain_summary":{"language":"...","text":"..."}}}
"""


def _format_observation(observation, state) -> str:
    parts: list[str] = []
    parts.append(f"Turn {observation.turn} of {observation.max_turns}.")
    parts.append(f"Citizen ({observation.language}): {observation.citizen_utterance!r}")
    if observation.elicited_facts:
        parts.append("Facts elicited so far: " + ", ".join(observation.elicited_facts))
    return "\n".join(parts)


def build_vanilla_baseline(chat: LLMChat) -> Callable:
    """returns an Advisor backed by the LLM with a minimal system prompt."""

    def advisor(observation, state):
        messages: list[ChatMessage] = [
            {"role": "system", "content": VANILLA_SYSTEM},
            {"role": "user", "content": _format_observation(observation, state)},
        ]
        text = chat(messages, None)
        action, _err = parse_action(text, fallback_language=observation.language)
        return action

    return advisor


__all__ = ["VANILLA_SYSTEM", "build_vanilla_baseline"]
