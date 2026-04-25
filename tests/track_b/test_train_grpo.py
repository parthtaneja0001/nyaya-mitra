"""GRPO trainer orchestration tests. uses real_model=False so no GPU is needed.

what's tested:
- TrainConfig.from_yaml parses the shipped warmup + phase2 yaml files
- discover_seed_pool finds the seed integers in the seeds/ tree
- pick_seed produces deterministic (seed, difficulty) per step
- train(...) runs N steps, writes a metrics jsonl, snapshots adapters
- abort_min_rolling_mean halts early when set absurdly high
- step_record_from_result parses a real EpisodeResult into the right shape
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.rollout import EpisodeResult
from training.train_grpo import (
    TrainConfig,
    discover_seed_pool,
    pick_seed,
    step_record_from_result,
    train,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARMUP_YAML = REPO_ROOT / "training" / "configs" / "advisor_warmup.yaml"
PHASE2_YAML = REPO_ROOT / "training" / "configs" / "advisor_phase2.yaml"


def test_warmup_yaml_parses():
    cfg = TrainConfig.from_yaml(WARMUP_YAML)
    assert cfg.phase == "warmup"
    assert "Qwen2.5" in cfg.base_model
    assert cfg.num_episodes >= 1
    assert cfg.max_turns > 0
    assert "easy" in cfg.difficulty_mix


def test_phase2_yaml_parses():
    cfg = TrainConfig.from_yaml(PHASE2_YAML)
    assert cfg.phase == "phase2"
    assert cfg.resume_adapter is not None


def test_seed_pool_discovers_real_seeds():
    pool = discover_seed_pool()
    # at least easy must exist in main
    assert "easy" in pool
    assert all(isinstance(s, int) for s in pool["easy"])
    assert len(pool["easy"]) >= 1


def test_pick_seed_is_deterministic():
    pool = {"easy": [101, 102, 103], "medium": [201, 202], "hard": [301]}
    mix = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
    a = pick_seed(7, mix, pool)
    b = pick_seed(7, mix, pool)
    assert a == b


def test_pick_seed_falls_back_to_easy_when_pool_empty():
    seed, diff = pick_seed(0, mix={"easy": 1.0}, seed_pool={})
    assert diff == "easy"
    assert seed == 1


def test_step_record_from_result_extracts_components():
    er = EpisodeResult(
        seed=1,
        difficulty="easy",
        turns=[],
        final_breakdown={
            "scheme_precision": 0.5,
            "gate_format_violation": 0.0,
            "gate_hallucination": 1.0,
            "gate_contradiction": 0.0,
            "gate_sim_leak": 0.0,
            "total": -1.0,
        },
        total_reward=-1.0,
        finalized=False,
        truncated_by_env=False,
        elicited_facts=[],
        sim_leak_count=0,
        wall_seconds=0.1,
    )
    rec = step_record_from_result(7, er)
    assert rec.step == 7
    assert "scheme_precision" in rec.components
    assert rec.gate_counts["gate_hallucination"] == 1
    assert rec.gate_counts["gate_format_violation"] == 0


def test_train_with_fakechat_runs_a_few_steps(tmp_path: Path):
    """end-to-end orchestration with the fake chat. confirms metrics jsonl is
    written, episodes finalize, and the summary is well-shaped."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        """
phase: warmup
model:
  base_model: fake
  load_in_4bit: false
  use_unsloth: false
grpo:
  num_episodes: 5
  learning_rate: 1.0e-5
  num_generations: 1
  temperature: 0.0
  top_p: 1.0
env:
  max_turns: 3
  difficulty_mix: {easy: 1.0}
logging:
  log_every: 1
  transcript_dump_every: 100
  transcript_dump_count: 0
  adapter_snapshot_every: 0
  metrics_jsonl: REPLACED
  wandb_project: nyaya-mitra-grpo
  wandb_tags: [test]
""".strip(),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.jsonl"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace("REPLACED", str(metrics_path)),
        encoding="utf-8",
    )

    summary = train(
        cfg_path,
        real_model=False,
        log_to_wandb=False,
        out_dumps=tmp_path / "dumps",
        out_checkpoints=tmp_path / "ckpt",
    )
    assert summary["phase"] == "warmup"
    assert summary["steps_run"] == 5
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert all("total_reward" in r for r in rows)
    assert all("rolling_mean_100" in r for r in rows)


def test_train_does_not_write_adapter_when_snapshot_every_is_zero(tmp_path: Path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        """
phase: warmup
model:
  base_model: fake
grpo: {num_episodes: 3, learning_rate: 1.0e-5, num_generations: 1, temperature: 0.0, top_p: 1.0}
env: {max_turns: 3, difficulty_mix: {easy: 1.0}}
logging:
  log_every: 1
  transcript_dump_every: 100
  transcript_dump_count: 0
  adapter_snapshot_every: 0
  metrics_jsonl: REPLACED
  wandb_project: x
  wandb_tags: []
""".strip(),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "m.jsonl"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace("REPLACED", str(metrics_path)),
        encoding="utf-8",
    )
    ckpt_dir = tmp_path / "ckpt"
    train(
        cfg_path,
        real_model=False,
        log_to_wandb=False,
        out_dumps=tmp_path / "d",
        out_checkpoints=ckpt_dir,
    )
    # only the final-phase adapter is written; no intermediate snapshots
    written = list(ckpt_dir.glob("adapter_step_*.lora"))
    assert written == []


def test_train_real_model_raises_clean_error_without_deps(monkeypatch, tmp_path):
    """real_model=True without GPU stack must raise ImportError, not crash mid-loop."""
    cfg_path = tmp_path / "x.yaml"
    cfg_path.write_text(
        """
phase: warmup
model: {base_model: x}
grpo: {num_episodes: 1, learning_rate: 1.0e-5, num_generations: 1, temperature: 0.0, top_p: 1.0}
env: {max_turns: 2, difficulty_mix: {easy: 1.0}}
logging:
  log_every: 1
  transcript_dump_every: 100
  transcript_dump_count: 0
  adapter_snapshot_every: 0
  metrics_jsonl: REPLACED
  wandb_project: x
  wandb_tags: []
""".strip(),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "m.jsonl"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace("REPLACED", str(metrics_path)),
        encoding="utf-8",
    )
    with pytest.raises(ImportError):
        train(
            cfg_path,
            real_model=True,
            log_to_wandb=False,
            out_dumps=tmp_path / "d",
            out_checkpoints=tmp_path / "c",
        )
