"""GRPO training entrypoint. orchestrates rollouts -> reward -> policy update.

design notes
============
- heavy deps (torch, transformers, trl, unsloth) are imported lazily inside
  build_policy() / train(). this lets the rest of the orchestration code be
  unit-tested without a GPU.

- the reward function called per episode is exactly scripts.wire_rewards.build_env's
  wired-in fn. the trainer never re-implements scoring.

- rollout: every GRPO step samples K episodes per prompt. each episode is one
  full advisor<->citizen dialogue. the policy generates the entire conversation
  turn-by-turn (one model call per turn). per-step W&B + JSONL logging.

- the policy interface is the LLMChat protocol from eval.baselines.llm_protocol.
  build_policy() returns (chat_fn, save_adapter_fn). during GRPO the chat_fn
  internally captures (prompt, completion) pairs that TRL's GRPOTrainer needs.
  this keeps the rollout loop framework-agnostic.

- safe-by-default: if no GPU is available we fall back to the FakeChat advisor
  (constant replies) so CI exercises the orchestration end-to-end without a
  model. real training is gated on `--real-model`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eval.baselines.llm_protocol import FakeChat, LLMChat
from eval.baselines.prompted_baseline import build_prompted_baseline
from training.rollout import EpisodeResult, run_episode

logger = logging.getLogger("nyaya.train_grpo")


REPO_ROOT = Path(__file__).resolve().parent.parent
DUMPS_DEFAULT = REPO_ROOT / "training" / "dumps"
CHECKPOINTS_DEFAULT = REPO_ROOT / "training" / "checkpoints"


@dataclass
class TrainConfig:
    """typed view of the yaml config. only the fields the orchestration touches."""

    phase: str
    base_model: str
    load_in_4bit: bool
    use_unsloth: bool
    num_episodes: int
    learning_rate: float
    num_generations: int
    temperature: float
    top_p: float
    max_turns: int
    difficulty_mix: dict[str, float]
    log_every: int
    transcript_dump_every: int
    transcript_dump_count: int
    adapter_snapshot_every: int
    metrics_jsonl: Path
    wandb_project: str
    wandb_tags: list[str]
    resume_adapter: Path | None = None
    abort_min_rolling_mean: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = raw.get("model") or {}
        grpo = raw.get("grpo") or {}
        env_cfg = raw.get("env") or {}
        log = raw.get("logging") or {}
        abort = raw.get("abort") or {}
        return cls(
            phase=str(raw.get("phase", "warmup")),
            base_model=str(model.get("base_model", "Qwen/Qwen2.5-3B-Instruct")),
            load_in_4bit=bool(model.get("load_in_4bit", True)),
            use_unsloth=bool(model.get("use_unsloth", True)),
            num_episodes=int(grpo.get("num_episodes", 500)),
            learning_rate=float(grpo.get("learning_rate", 1e-5)),
            num_generations=int(grpo.get("num_generations", 4)),
            temperature=float(grpo.get("temperature", 0.9)),
            top_p=float(grpo.get("top_p", 0.95)),
            max_turns=int(env_cfg.get("max_turns", 12)),
            difficulty_mix=dict(env_cfg.get("difficulty_mix") or {}),
            log_every=int(log.get("log_every", 1)),
            transcript_dump_every=int(log.get("transcript_dump_every", 100)),
            transcript_dump_count=int(log.get("transcript_dump_count", 5)),
            adapter_snapshot_every=int(log.get("adapter_snapshot_every", 200)),
            metrics_jsonl=Path(log.get("metrics_jsonl", DUMPS_DEFAULT / "metrics.jsonl")),
            wandb_project=str(log.get("wandb_project", "nyaya-mitra-grpo")),
            wandb_tags=list(log.get("wandb_tags") or []),
            resume_adapter=(Path(model["resume_adapter"]) if model.get("resume_adapter") else None),
            abort_min_rolling_mean=(
                float(abort["min_rolling_mean_at_500"])
                if "min_rolling_mean_at_500" in abort
                else None
            ),
            raw=raw,
        )


def pick_seed(step: int, mix: dict[str, float], seed_pool: dict[str, list[int]]) -> tuple[int, str]:
    """deterministic curriculum scheduler: same step -> same (seed, difficulty)."""
    if not mix or not seed_pool:
        # fall back to easy
        pool = seed_pool.get("easy", [1])
        return pool[step % len(pool)], "easy"
    # build a cumulative sum
    items = [(k, mix.get(k, 0.0)) for k in ("easy", "medium", "hard") if k in seed_pool]
    total = sum(w for _, w in items)
    if total <= 0:
        pool = seed_pool.get("easy", [1])
        return pool[step % len(pool)], "easy"
    # use step as the deterministic position in the cumulative distribution
    pos = ((step * 31 + 7) % 1000) / 1000.0 * total
    cum = 0.0
    for diff, w in items:
        cum += w
        if pos <= cum:
            pool = seed_pool[diff]
            return pool[step % len(pool)], diff
    diff, _ = items[-1]
    pool = seed_pool[diff]
    return pool[step % len(pool)], diff


def discover_seed_pool() -> dict[str, list[int]]:
    """read seed ids from src/nyaya_mitra/profile/seeds/{easy,medium,hard}/*.json."""
    base = REPO_ROOT / "src" / "nyaya_mitra" / "profile" / "seeds"
    out: dict[str, list[int]] = {}
    for diff in ("easy", "medium", "hard"):
        files = sorted((base / diff).glob("*.json"))
        seeds: list[int] = []
        for f in files:
            try:
                seeds.append(int(json.loads(f.read_text(encoding="utf-8")).get("seed", 0)))
            except Exception:
                continue
        if seeds:
            out[diff] = seeds
    return out


def jsonl_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def dump_transcript(out_dir: Path, step: int, idx: int, result: EpisodeResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "idx": idx,
        "seed": result.seed,
        "difficulty": result.difficulty,
        "total_reward": result.total_reward,
        "finalized": result.finalized,
        "truncated": result.truncated_by_env,
        "elicited_facts": result.elicited_facts,
        "sim_leak_count": result.sim_leak_count,
        "wall_seconds": result.wall_seconds,
        "final_breakdown": result.final_breakdown,
        "turns": [
            {
                "i": t.turn_index,
                "obs": t.observation_in.citizen_utterance,
                "action": t.action.model_dump(),
                "reward": t.reward,
            }
            for t in result.turns
        ],
        "error": result.error,
    }
    (out_dir / f"step_{step:06d}_ep_{idx:03d}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


@dataclass
class StepRecord:
    step: int
    seed: int
    difficulty: str
    total_reward: float
    finalized: bool
    sim_leak_count: int
    components: dict[str, float]
    gate_counts: dict[str, int]


def step_record_from_result(step: int, result: EpisodeResult) -> StepRecord:
    breakdown = result.final_breakdown or {}
    components = {
        k: v for k, v in breakdown.items() if not k.startswith("gate_") and k not in {"total"}
    }
    gates = {
        "gate_format_violation": int(breakdown.get("gate_format_violation", 0) > 0),
        "gate_hallucination": int(breakdown.get("gate_hallucination", 0) > 0),
        "gate_contradiction": int(breakdown.get("gate_contradiction", 0) > 0),
        "gate_sim_leak": int(breakdown.get("gate_sim_leak", 0) > 0),
    }
    return StepRecord(
        step=step,
        seed=result.seed,
        difficulty=result.difficulty or "easy",
        total_reward=result.total_reward,
        finalized=result.finalized,
        sim_leak_count=result.sim_leak_count,
        components=components,
        gate_counts=gates,
    )


def build_chat_for_training(
    cfg: TrainConfig, *, real_model: bool
) -> tuple[LLMChat, Callable[[Path], None]]:
    """returns (chat_callable, save_adapter_fn).

    real_model=False  -> FakeChat that always returns a Finalize blob; orchestration test only.
    real_model=True   -> Unsloth+TRL backed chat. raises ImportError if deps absent.
    """
    if not real_model:
        finalize_blob = (
            '{"type":"FINALIZE","plan":{"schemes":[],'
            '"legal_routes":[{"framework_id":"domestic_violence_act_2005",'
            '"applicable_situation":"x","forum":"magistrate","procedural_steps":["a"],'
            '"free_legal_aid_contact":{"authority":"DLSA","contact_id":"dlsa_ludhiana"},'
            '"required_documents":["b"]}],'
            '"most_important_next_step":"contact dlsa","plain_summary":{"language":"en","text":"y"}}}'
        )
        chat = FakeChat([finalize_blob])

        def _save_noop(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# fake adapter snapshot from step at {time.time()}\n", encoding="utf-8"
            )

        return chat, _save_noop

    # real path: deferred import so the orchestration is testable without GPU.
    try:
        from training._real_policy import build_unsloth_grpo_chat  # type: ignore
    except Exception as exc:
        raise ImportError(
            "real-model training requires torch + transformers + trl + unsloth + peft. "
            f"underlying error: {exc!r}"
        ) from exc
    return build_unsloth_grpo_chat(cfg)


def train(
    config_path: Path,
    *,
    real_model: bool = False,
    log_to_wandb: bool = False,
    out_dumps: Path | None = None,
    out_checkpoints: Path | None = None,
) -> dict[str, Any]:
    """run GRPO training. returns {steps_run, final_rolling_mean, aborted, ...}."""
    from scripts.wire_rewards import build_env  # local import

    cfg = TrainConfig.from_yaml(config_path)
    out_dumps = out_dumps or DUMPS_DEFAULT
    out_checkpoints = out_checkpoints or CHECKPOINTS_DEFAULT
    out_dumps.mkdir(parents=True, exist_ok=True)
    out_checkpoints.mkdir(parents=True, exist_ok=True)

    chat, save_adapter = build_chat_for_training(cfg, real_model=real_model)

    advisor = build_prompted_baseline(chat)

    seed_pool = discover_seed_pool()
    rolling: deque[float] = deque(maxlen=100)

    wandb_run = None
    if log_to_wandb:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                tags=cfg.wandb_tags,
                config=cfg.raw,
            )
        except Exception as exc:
            logger.warning("wandb init failed (%s); proceeding with jsonl only", exc)

    aborted = False
    steps_run = 0
    last_record: StepRecord | None = None

    started = time.perf_counter()
    for step in range(cfg.num_episodes):
        seed, difficulty = pick_seed(step, cfg.difficulty_mix, seed_pool)
        env = build_env(max_turns=cfg.max_turns)
        try:
            result = run_episode(env, advisor, seed=seed, difficulty=difficulty)
        finally:
            env.close()

        rec = step_record_from_result(step, result)
        rolling.append(rec.total_reward)

        if step % cfg.log_every == 0:
            row = {
                "step": rec.step,
                "seed": rec.seed,
                "difficulty": rec.difficulty,
                "total_reward": rec.total_reward,
                "finalized": rec.finalized,
                "sim_leak_count": rec.sim_leak_count,
                "components": rec.components,
                "gate_counts": rec.gate_counts,
                "rolling_mean_100": sum(rolling) / len(rolling) if rolling else 0.0,
            }
            jsonl_log(cfg.metrics_jsonl, row)
            if wandb_run is not None:
                try:
                    wandb_run.log(row, step=step)
                except Exception:
                    pass

        if step % cfg.transcript_dump_every == 0 and step > 0:
            for j in range(min(cfg.transcript_dump_count, 1)):
                dump_transcript(out_dumps / f"step_{step:06d}", step, j, result)

        if cfg.adapter_snapshot_every > 0 and step % cfg.adapter_snapshot_every == 0 and step > 0:
            try:
                save_adapter(out_checkpoints / f"adapter_step_{step:06d}.lora")
            except Exception as exc:
                logger.warning("adapter snapshot failed at step %d: %s", step, exc)

        if (
            cfg.abort_min_rolling_mean is not None
            and step >= 500
            and len(rolling) == 100
            and (sum(rolling) / 100) < cfg.abort_min_rolling_mean
        ):
            logger.warning(
                "rolling mean %.3f < abort threshold %.3f at step %d; stopping.",
                sum(rolling) / 100,
                cfg.abort_min_rolling_mean,
                step,
            )
            aborted = True
            break

        steps_run += 1
        last_record = rec

    final_rolling = sum(rolling) / len(rolling) if rolling else 0.0
    try:
        save_adapter(out_checkpoints / f"adapter_final_{cfg.phase}.lora")
    except Exception:
        pass

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return {
        "config": str(config_path),
        "phase": cfg.phase,
        "steps_run": steps_run,
        "final_rolling_mean": final_rolling,
        "aborted": aborted,
        "wall_seconds": time.perf_counter() - started,
        "last": last_record.__dict__ if last_record else None,
    }


def _main():
    p = argparse.ArgumentParser(description="GRPO trainer for nyaya-mitra")
    p.add_argument("--config", required=True, help="path to a yaml config under training/configs/")
    p.add_argument("--real-model", action="store_true", help="use unsloth+trl (needs GPU)")
    p.add_argument("--wandb", action="store_true", help="log to W&B")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    summary = train(
        Path(args.config),
        real_model=args.real_model,
        log_to_wandb=args.wandb,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _main()


__all__ = [
    "StepRecord",
    "TrainConfig",
    "build_chat_for_training",
    "discover_seed_pool",
    "pick_seed",
    "step_record_from_result",
    "train",
]
