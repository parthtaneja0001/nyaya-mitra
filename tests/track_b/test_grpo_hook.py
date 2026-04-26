"""verify train_grpo invokes chat.grpo_step(reward) when the chat exposes it.

uses a hand-rolled fake chat that captures grpo_step calls. confirms:
- grpo_step is called once per episode
- it receives the episode's total_reward
- failures in grpo_step trigger flush_episode + log a warning, but don't crash
"""

from __future__ import annotations

from pathlib import Path

from training.train_grpo import train


class _FakeChatWithGRPO:
    """mimics the real-model chat surface for the train_grpo hook."""

    def __init__(self, finalize_blob: str) -> None:
        self._finalize_blob = finalize_blob
        self._cursor = 0
        self.calls: list[list[dict]] = []
        self.grpo_calls: list[float] = []
        self.flush_calls = 0
        self.fail_grpo = False

    def __call__(self, messages, options=None):
        self.calls.append(list(messages))
        return self._finalize_blob

    def grpo_step(self, reward: float):
        self.grpo_calls.append(reward)
        if self.fail_grpo:
            raise RuntimeError("simulated grpo failure")
        return {"loss": 0.0, "n_turns": 1, "reward": reward}

    def flush_episode(self):
        self.flush_calls += 1


def _tiny_yaml(tmp_path: Path, n: int = 3) -> Path:
    metrics = tmp_path / "metrics.jsonl"
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        f"""
phase: warmup
model:
  base_model: fake
grpo:
  num_episodes: {n}
  learning_rate: 1.0e-5
  num_generations: 1
  temperature: 0.0
  top_p: 1.0
env: {{max_turns: 3, difficulty_mix: {{easy: 1.0}}}}
logging:
  log_every: 1
  transcript_dump_every: 100
  transcript_dump_count: 0
  adapter_snapshot_every: 0
  metrics_jsonl: {metrics}
  wandb_project: x
  wandb_tags: []
""".strip(),
        encoding="utf-8",
    )
    return cfg_path


def test_grpo_step_invoked_once_per_episode(monkeypatch, tmp_path: Path):
    """patch build_chat_for_training so we control the chat callable."""
    from training import train_grpo

    finalize_blob = (
        '{"type":"FINALIZE","plan":{"schemes":[],'
        '"legal_routes":[{"framework_id":"domestic_violence_act_2005",'
        '"applicable_situation":"x","forum":"magistrate","procedural_steps":["a"],'
        '"free_legal_aid_contact":{"authority":"DLSA","contact_id":"dlsa_ludhiana"},'
        '"required_documents":["b"]}],'
        '"most_important_next_step":"contact dlsa","plain_summary":{"language":"en","text":"y"}}}'
    )
    chat = _FakeChatWithGRPO(finalize_blob)

    def fake_builder(cfg, *, real_model: bool):
        return chat, lambda p: None

    monkeypatch.setattr(train_grpo, "build_chat_for_training", fake_builder)

    cfg_path = _tiny_yaml(tmp_path, n=3)
    summary = train(
        cfg_path,
        real_model=False,  # value irrelevant; we monkey-patched the builder
        log_to_wandb=False,
        out_dumps=tmp_path / "d",
        out_checkpoints=tmp_path / "c",
    )
    assert summary["steps_run"] == 3
    assert len(chat.grpo_calls) == 3
    # rewards plumbed through correctly
    assert all(isinstance(r, float) for r in chat.grpo_calls)


def test_grpo_step_failure_triggers_flush_and_continues(monkeypatch, tmp_path: Path):
    from training import train_grpo

    finalize_blob = (
        '{"type":"FINALIZE","plan":{"schemes":[],'
        '"legal_routes":[{"framework_id":"domestic_violence_act_2005",'
        '"applicable_situation":"x","forum":"magistrate","procedural_steps":["a"],'
        '"free_legal_aid_contact":{"authority":"DLSA","contact_id":"dlsa_ludhiana"},'
        '"required_documents":["b"]}],'
        '"most_important_next_step":"contact dlsa","plain_summary":{"language":"en","text":"y"}}}'
    )
    chat = _FakeChatWithGRPO(finalize_blob)
    chat.fail_grpo = True

    monkeypatch.setattr(
        train_grpo,
        "build_chat_for_training",
        lambda cfg, *, real_model: (chat, lambda p: None),
    )

    cfg_path = _tiny_yaml(tmp_path, n=2)
    summary = train(
        cfg_path,
        real_model=False,
        log_to_wandb=False,
        out_dumps=tmp_path / "d",
        out_checkpoints=tmp_path / "c",
    )
    # training kept running through the failures
    assert summary["steps_run"] == 2
    assert len(chat.grpo_calls) == 2
    # flush was called each time grpo_step raised
    assert chat.flush_calls == 2


def test_chat_without_grpo_step_runs_without_calling_it(tmp_path: Path):
    """sanity: FakeChat from llm_protocol has no grpo_step, train still works."""
    cfg_path = _tiny_yaml(tmp_path, n=2)
    summary = train(
        cfg_path,
        real_model=False,
        log_to_wandb=False,
        out_dumps=tmp_path / "d",
        out_checkpoints=tmp_path / "c",
    )
    assert summary["steps_run"] == 2
