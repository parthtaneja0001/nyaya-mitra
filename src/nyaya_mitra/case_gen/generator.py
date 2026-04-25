"""generator wrapper: LLM produces a profile json, validator+diversity score it,
trainer optimizes against -advisor_reward + diversity_penalty + invalid_penalty.

GenerationResult composes the four signals so the trainer's reward is a single
scalar even though there are three separate penalties.

reward formula (per PLAN B.6):
    R_gen = - advisor_total_reward
            - 0.5 * max_similarity   # diversity
            - 1.0 if invalid else 0  # validity gate

build_generator_advisor wraps an LLMChat into a Callable[[None], dict] — given
no input, ask the model to emit a profile json. used by training/train_grpo.py
in phase 2.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from eval.baselines.llm_protocol import ChatMessage, LLMChat
from nyaya_mitra.case_gen.diversity import DiversityTracker
from nyaya_mitra.case_gen.validator import ProfileValidator, ValidationResult

GENERATOR_SYSTEM = """You generate Indian citizen profiles in JSON for an RL training \
environment. Each profile must:

1. Match the schema below (CitizenProfile from interface).
2. Be internally consistent (don't make a 14-year-old married, don't make a software \
   engineer with monthly_income 4000, etc).
3. Match at least one welfare scheme OR one legal framework via the eligibility \
   checkers — no degenerate profiles.
4. Be diverse from recent generations: vary state, occupation, presenting_issue, \
   sensitive_facts.

Output ONE JSON object per call, no commentary. Schema:

{
  "seed": int,
  "demographics": {"gender": "...", "age": int, "state": "...", "district": "...", "residence": "rural|urban"},
  "economic": {"occupation": "...", "holds_cultivable_land": bool, "monthly_income": int, "bpl_household": bool, ...},
  "family": {"marital_status": "...", "children": int},
  "situation_specific": {
    "presenting_issue": "...",
    "hidden_facts": {...},
    "sensitive_facts": {"dv_present": bool, ...}
  },
  "behavior": {"trust_level": "wary|neutral|open", "verbosity": "low|med|high",
               "language_preference": "en|hi|hinglish", "literacy": "low|medium|high",
               "initial_vague_query": "..."}
}
"""


@dataclass
class GeneratedCase:
    raw_text: str
    parsed: dict | None
    parse_error: str | None
    validation: ValidationResult
    similarity: float
    advisor_total_reward: float | None = None
    reward: float = 0.0


def _extract_json(text: str) -> tuple[dict | None, str | None]:
    """forgiving json extraction. shared with eval.baselines.action_parser shape
    but kept local so case_gen has no dependency on eval/."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    s = text.strip()
    # strip code fences
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl >= 0:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    # find first balanced object
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(s):
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
                blob = s[start : i + 1]
                try:
                    return json.loads(blob), None
                except json.JSONDecodeError as exc:
                    return None, f"json decode: {exc.msg}"
    return None, "no json object found"


def score_generation(
    candidate: GeneratedCase,
    *,
    diversity_weight: float = 0.5,
    invalid_weight: float = 1.0,
) -> float:
    """compute generator reward from a populated GeneratedCase. assumes
    candidate.advisor_total_reward is set if the candidate was rolled out.
    """
    if not candidate.validation.valid:
        return -invalid_weight
    advisor = candidate.advisor_total_reward if candidate.advisor_total_reward is not None else 0.0
    return -advisor - diversity_weight * candidate.similarity


def build_generator_advisor(
    chat: LLMChat,
    validator: ProfileValidator,
    tracker: DiversityTracker,
    *,
    user_prompt: str = "Generate a new training profile.",
) -> Callable[[], GeneratedCase]:
    """returns a zero-arg callable that produces a fresh GeneratedCase per call.

    the caller is responsible for: rolling the candidate through the env to get
    an advisor_total_reward, calling score_generation, then tracker.record() if
    they decide to keep this generation in the pool.
    """

    def generate() -> GeneratedCase:
        messages: list[ChatMessage] = [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        text = chat(messages, None)
        parsed, parse_err = _extract_json(text)
        if parsed is None:
            return GeneratedCase(
                raw_text=text,
                parsed=None,
                parse_error=parse_err,
                validation=ValidationResult(valid=False, schema_error=parse_err),
                similarity=0.0,
            )
        validation = validator.validate(parsed)
        similarity = tracker.max_similarity(parsed)
        return GeneratedCase(
            raw_text=text,
            parsed=parsed,
            parse_error=None,
            validation=validation,
            similarity=similarity,
        )

    return generate


__all__ = [
    "GENERATOR_SYSTEM",
    "GeneratedCase",
    "build_generator_advisor",
    "score_generation",
]
