"""tests for the three baselines + the action parser they share."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from eval.baselines import (
    build_prompted_baseline,
    build_scripted_baseline,
    build_vanilla_baseline,
)
from eval.baselines.action_parser import parse_action
from eval.baselines.llm_protocol import FakeChat
from nyaya_mitra.interface import (
    Ask,
    Explain,
    Finalize,
    Probe,
)
from training.rollout import run_episode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _wire():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module("scripts.wire_rewards")
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


# ---------- action parser ----------


class TestActionParser:
    def test_parses_ask(self):
        text = '{"type":"ASK","question":"hi","language":"en"}'
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Ask)
        assert action.question == "hi"

    def test_parses_probe(self):
        text = '{"type":"PROBE","question":"q","sensitive_topic":"dv","language":"en"}'
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Probe)

    def test_parses_explain(self):
        text = '{"type":"EXPLAIN","content":"c","target_literacy":"low","language":"hi"}'
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Explain)

    def test_parses_finalize_with_plan(self):
        text = json.dumps(
            {
                "type": "FINALIZE",
                "plan": {
                    "schemes": [],
                    "legal_routes": [
                        {
                            "framework_id": "domestic_violence_act_2005",
                            "applicable_situation": "x",
                            "forum": "magistrate",
                            "procedural_steps": ["a"],
                            "free_legal_aid_contact": {
                                "authority": "DLSA",
                                "contact_id": "dlsa_x",
                            },
                            "required_documents": ["b"],
                        }
                    ],
                    "most_important_next_step": "y",
                    "plain_summary": {"language": "en", "text": "z"},
                },
            }
        )
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Finalize)

    def test_strips_code_fences(self):
        text = '```json\n{"type":"ASK","question":"x","language":"en"}\n```'
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Ask)

    def test_finds_json_amid_chatter(self):
        text = 'Here is my response. {"type":"ASK","question":"x","language":"en"} hope it helps.'
        action, err = parse_action(text)
        assert err is None
        assert isinstance(action, Ask)

    def test_empty_response_fallback(self):
        action, err = parse_action("")
        assert err
        assert isinstance(action, Ask)

    def test_invalid_json_fallback(self):
        action, err = parse_action("not json at all")
        assert err
        assert isinstance(action, Ask)

    def test_unknown_type_fallback(self):
        action, err = parse_action('{"type":"WAVE","question":"x"}')
        assert err
        assert isinstance(action, Ask)

    def test_validation_error_fallback(self):
        # missing required field
        action, err = parse_action('{"type":"ASK","language":"en"}')
        assert err
        assert isinstance(action, Ask)


# ---------- scripted baseline ----------


class TestScriptedBaseline:
    def test_runs_full_episode(self):
        wire = _wire()
        env = wire.build_env(max_turns=8)
        advisor = build_scripted_baseline(max_asks=2, max_probes=1)
        result = run_episode(env, advisor, seed=1)
        assert result.error is None
        assert result.finalized is True
        assert "scheme_precision" in result.final_breakdown

    def test_finalize_at_overrides_budget(self):
        wire = _wire()
        env = wire.build_env(max_turns=10)
        advisor = build_scripted_baseline(max_asks=10, max_probes=10, finalize_at=2)
        result = run_episode(env, advisor, seed=1)
        assert result.finalized is True

    def test_plan_is_format_valid(self):
        wire = _wire()
        env = wire.build_env(max_turns=8)
        advisor = build_scripted_baseline()
        result = run_episode(env, advisor, seed=1)
        # gate didn't fire
        assert result.final_breakdown.get("gate_format_violation", 0) == 0


# ---------- vanilla baseline ----------


class TestVanillaBaseline:
    def test_one_turn_finalize(self):
        wire = _wire()
        env = wire.build_env(max_turns=4)

        finalize_blob = json.dumps(
            {
                "type": "FINALIZE",
                "plan": {
                    "schemes": [],
                    "legal_routes": [
                        {
                            "framework_id": "domestic_violence_act_2005",
                            "applicable_situation": "x",
                            "forum": "magistrate",
                            "procedural_steps": ["a"],
                            "free_legal_aid_contact": {
                                "authority": "DLSA",
                                "contact_id": "dlsa_ludhiana",
                            },
                            "required_documents": ["b"],
                        }
                    ],
                    "most_important_next_step": "y",
                    "plain_summary": {"language": "en", "text": "z"},
                },
            }
        )
        chat = FakeChat([finalize_blob])
        advisor = build_vanilla_baseline(chat)
        result = run_episode(env, advisor, seed=1)
        assert result.error is None
        assert result.finalized is True
        # the chat was called at least once
        assert len(chat.calls) >= 1
        # system prompt shows up first
        assert chat.calls[0][0]["role"] == "system"

    def test_garbage_response_falls_back_to_ask(self):
        wire = _wire()
        env = wire.build_env(max_turns=4)
        chat = FakeChat(["lol I dunno", "still nope", "give up", "fine"])
        advisor = build_vanilla_baseline(chat)
        result = run_episode(env, advisor, seed=1)
        # truncated; no finalize ever happens
        assert result.error is None
        assert result.finalized is False
        assert result.truncated_by_env is True


# ---------- prompted baseline ----------


class TestPromptedBaseline:
    def test_kb_excerpt_loads(self):
        chat = FakeChat(['{"type":"ASK","question":"hi","language":"en"}'])
        advisor = build_prompted_baseline(chat)
        # exercise once to confirm the system prompt was constructed without error
        wire = _wire()
        env = wire.build_env(max_turns=2)
        run_episode(env, advisor, seed=1)
        # first call's system message is the prompted header + KB excerpt
        sys_msg = chat.calls[0][0]
        assert sys_msg["role"] == "system"
        assert "Welfare schemes" in sys_msg["content"]
        assert "pm_kisan" in sys_msg["content"]
        assert "domestic_violence_act_2005" in sys_msg["content"]

    def test_finalize_round_trip(self):
        finalize_blob = json.dumps(
            {
                "type": "FINALIZE",
                "plan": {
                    "schemes": [
                        {
                            "scheme_id": "pm_kisan",
                            "rationale_facts": ["occupation_farmer"],
                            "required_documents": ["Aadhaar"],
                            "application_path": {
                                "online_url": None,
                                "offline_office": None,
                                "offline_steps": [],
                            },
                        }
                    ],
                    "legal_routes": [],
                    "most_important_next_step": "apply",
                    "plain_summary": {"language": "en", "text": "ok"},
                },
            }
        )
        chat = FakeChat([finalize_blob])
        advisor = build_prompted_baseline(chat)
        wire = _wire()
        env = wire.build_env(max_turns=2)
        result = run_episode(env, advisor, seed=1)
        assert result.error is None
        assert result.finalized is True


# ---------- LLM protocol ----------


def test_fakechat_round_robins_replies():
    chat = FakeChat(["a", "b"])
    assert chat([{"role": "user", "content": "x"}], None) == "a"
    assert chat([{"role": "user", "content": "y"}], None) == "b"
    assert chat([{"role": "user", "content": "z"}], None) == "a"


def test_fakechat_rejects_empty_replies():
    with pytest.raises(ValueError):
        FakeChat([])
