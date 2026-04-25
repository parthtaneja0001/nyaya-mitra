"""prompted baseline: same LLM as vanilla, but with KB excerpts and elicitation
guidance in the system prompt.

this is the *honest* comparison — what a non-RL approach can extract from the
same base model when given full KB context. RL has to beat this, not vanilla,
to be a meaningful win.

the KB excerpt is built once at advisor-construction time from the loaded KB,
trimmed to the fields the model needs (id, brief eligibility/applicability,
required_documents, forum, legal_aid_authority).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from eval.baselines.action_parser import parse_action
from eval.baselines.llm_protocol import ChatMessage, LLMChat

PROMPTED_SYSTEM_HEADER = """You are an Indian welfare paralegal helping vulnerable \
citizens access schemes and legal protections. You speak the citizen's language \
and literacy level. You ASK targeted questions, PROBE sensitively for protected \
attributes, EXPLAIN when needed, and FINALIZE with an evidence-grounded ActionPlan.

Critical rules:
- Never invent scheme_id or framework_id. Only use ids from the KB below.
- Every legal_routes entry MUST include free_legal_aid_contact.authority and contact_id.
- rationale_facts must reference fact ids the citizen actually revealed.
- Use PROBE (not ASK) for caste, dv, disability, immigration, hiv_status, sexual_orientation, mental_health.
- Match target_literacy to the citizen's literacy level. Avoid legal jargon for low literacy.

Action schema (output ONE JSON object per turn, nothing else):
- {"type":"ASK","question":"...","language":"en|hi|hinglish"}
- {"type":"PROBE","question":"...","sensitive_topic":"...","language":"..."}
- {"type":"EXPLAIN","content":"...","target_literacy":"low|medium|high","language":"..."}
- {"type":"FINALIZE","plan":{"schemes":[{"scheme_id":"...","rationale_facts":[...],"required_documents":[...],"application_path":{"online_url":null,"offline_office":null,"offline_steps":[]}}],"legal_routes":[{"framework_id":"...","applicable_situation":"...","forum":"...","procedural_steps":[...],"free_legal_aid_contact":{"authority":"NALSA|SLSA|DLSA","contact_id":"..."},"required_documents":[...],"limitation_period_note":null}],"most_important_next_step":"...","plain_summary":{"language":"...","text":"..."}}}
"""


def _format_scheme(s: dict) -> str:
    return (
        f"- {s['scheme_id']} ({s['category']}): "
        f"{s['eligibility_rules_human'][:160]} | "
        f"docs: {', '.join(s.get('required_documents', [])[:5])}"
    )


def _format_framework(f: dict) -> str:
    return (
        f"- {f['framework_id']} ({f['category']}): "
        f"applies when: {'; '.join(f.get('applicable_situations', [])[:2])} | "
        f"forum: {f.get('forum', '?')} | "
        f"legal aid: {f.get('legal_aid_authority', 'DLSA')}"
    )


def _format_dlsa(dlsa: dict) -> str:
    nalsa = dlsa.get("NALSA") or {}
    nalsa_id = nalsa.get("contact_id", "nalsa_central")
    slsas = list((dlsa.get("SLSAs") or {}).keys())[:5]
    dlsas = list((dlsa.get("DLSAs") or {}).keys())[:8]
    return (
        f"NALSA contact: {nalsa_id}\n"
        f"Sample SLSA states: {', '.join(slsas) or '(none)'}\n"
        f"Sample DLSA districts: {', '.join(dlsas) or '(none)'}"
    )


def _build_kb_excerpt(schemes_dir: Path, frameworks_dir: Path, dlsa_path: Path) -> str:
    schemes: list[dict] = []
    if schemes_dir.exists():
        for p in sorted(schemes_dir.glob("*.json")):
            schemes.append(json.loads(p.read_text(encoding="utf-8")))
    frameworks: list[dict] = []
    if frameworks_dir.exists():
        for p in sorted(frameworks_dir.glob("*.json")):
            frameworks.append(json.loads(p.read_text(encoding="utf-8")))
    dlsa: dict = {}
    if dlsa_path.exists():
        dlsa = json.loads(dlsa_path.read_text(encoding="utf-8"))

    parts: list[str] = []
    parts.append("# Welfare schemes")
    parts.extend(_format_scheme(s) for s in schemes)
    parts.append("\n# Legal frameworks")
    parts.extend(_format_framework(f) for f in frameworks)
    parts.append("\n# Free legal aid")
    parts.append(_format_dlsa(dlsa))
    return "\n".join(parts)


def _format_observation(observation, state) -> str:
    parts: list[str] = [f"Turn {observation.turn} of {observation.max_turns}."]
    parts.append(f'Citizen ({observation.language}): "{observation.citizen_utterance}"')
    if observation.elicited_facts:
        parts.append("Facts elicited so far: " + ", ".join(observation.elicited_facts))
    if observation.facts_revealed_this_turn:
        parts.append("New this turn: " + ", ".join(observation.facts_revealed_this_turn))
    return "\n".join(parts)


def build_prompted_baseline(
    chat: LLMChat,
    *,
    schemes_dir: Path | None = None,
    frameworks_dir: Path | None = None,
    dlsa_path: Path | None = None,
) -> Callable:
    """returns an Advisor backed by the LLM with KB context loaded into the
    system prompt. paths default to the repo's nyaya_mitra.knowledge.data tree."""
    repo = Path(__file__).resolve().parent.parent.parent
    data = repo / "src" / "nyaya_mitra" / "knowledge" / "data"
    schemes_dir = schemes_dir or (data / "schemes")
    frameworks_dir = frameworks_dir or (data / "frameworks")
    dlsa_path = dlsa_path or (data / "dlsa_directory.json")

    kb_excerpt = _build_kb_excerpt(schemes_dir, frameworks_dir, dlsa_path)
    system_full = PROMPTED_SYSTEM_HEADER + "\n\n" + kb_excerpt

    def advisor(observation, state):
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_full},
            {"role": "user", "content": _format_observation(observation, state)},
        ]
        text = chat(messages, None)
        action, _err = parse_action(text, fallback_language=observation.language)
        return action

    return advisor


__all__ = ["PROMPTED_SYSTEM_HEADER", "build_prompted_baseline"]
