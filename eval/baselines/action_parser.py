"""parses an LLM response into an AdvisorAction.

the model is instructed to emit a single JSON object matching one of:
  {"type":"ASK","question":"...","language":"en"}
  {"type":"PROBE","question":"...","sensitive_topic":"dv","language":"en"}
  {"type":"EXPLAIN","content":"...","target_literacy":"low","language":"en"}
  {"type":"FINALIZE","plan":{...}}

the parser is forgiving: it strips code fences, finds the first balanced
JSON object, and uses pydantic for validation. on failure it returns a safe
default Ask so the episode continues; the trainer can use the parse_error
flag in info to penalize malformed outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from nyaya_mitra.interface import (
    AdvisorAction,
    Ask,
    Explain,
    Finalize,
    Probe,
)

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_blob(text: str) -> str | None:
    """find a balanced top-level JSON object in text. handles fenced blocks."""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    # scan for first '{' and walk to its matching '}'
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def parse_action(text: str, *, fallback_language: str = "en") -> tuple[AdvisorAction, str | None]:
    """returns (action, parse_error_or_None).

    on any parse/validation failure produces a safe Ask("could you tell me more?")
    so the episode continues; the parse_error string is meant to be logged into
    info["parse_error"] by the caller for trainer-side penalization.
    """
    if not isinstance(text, str) or not text.strip():
        return _safe_ask(fallback_language), "empty response"

    blob = _extract_json_blob(text)
    if not blob:
        return _safe_ask(fallback_language), "no json object found"

    try:
        payload: Any = json.loads(blob)
    except json.JSONDecodeError as exc:
        return _safe_ask(fallback_language), f"json decode: {exc.msg}"

    if not isinstance(payload, dict):
        return _safe_ask(fallback_language), "json is not an object"

    t = payload.get("type")
    try:
        if t == "ASK":
            return Ask.model_validate(payload), None
        if t == "PROBE":
            return Probe.model_validate(payload), None
        if t == "EXPLAIN":
            return Explain.model_validate(payload), None
        if t == "FINALIZE":
            return Finalize.model_validate(payload), None
    except ValidationError as exc:
        return _safe_ask(fallback_language), f"validation: {exc.error_count()} errors"

    return _safe_ask(fallback_language), f"unknown type: {t!r}"


def _safe_ask(language: str) -> Ask:
    lang = language if language in {"en", "hi", "hinglish"} else "en"
    return Ask(question="Could you tell me a little more about your situation?", language=lang)  # type: ignore[arg-type]


__all__ = ["parse_action"]
