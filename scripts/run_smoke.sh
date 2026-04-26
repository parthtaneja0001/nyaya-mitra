#!/usr/bin/env bash
# 1-hour smoke run on A100. uses Qwen 2.5 0.5B + 50 episodes to confirm the
# full pipeline (model load, rollout, reward, grpo update, save adapter, log)
# works end-to-end before burning real credits on phase 1.
#
# pre-flight:
#   - run on a host with CUDA + ~24GB GPU
#   - pip install -e ".[track_a,track_b,train]"
#   - export HF_TOKEN=hf_xxx (for any gated models; Qwen 2.5 0.5B is open)
#   - export WANDB_API_KEY=... (optional)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="$ROOT/training/configs/smoke.yaml"
mkdir -p "$ROOT/training/dumps" "$ROOT/training/checkpoints"

cat > "$CFG" <<'YAML'
phase: smoke
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  load_in_4bit: true
  use_unsloth: true

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

grpo:
  num_generations: 1
  learning_rate: 1.0e-5
  beta: 0.04
  max_prompt_length: 4096
  max_completion_length: 384
  temperature: 0.9
  top_p: 0.95
  num_episodes: 50
  episode_batch_size: 1

env:
  max_turns: 8
  difficulty_mix:
    easy: 0.8
    medium: 0.2
    hard: 0.0

logging:
  wandb_project: nyaya-mitra-grpo
  wandb_tags: [smoke, qwen-0.5b]
  log_every: 1
  transcript_dump_every: 10
  transcript_dump_count: 1
  adapter_snapshot_every: 25
  metrics_jsonl: training/dumps/smoke_metrics.jsonl
YAML

echo "=== smoke run config ==="
cat "$CFG"
echo "==="
echo "starting smoke run at $(date)..."

WANDB_FLAG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_FLAG="--wandb"
fi

python -m training.train_grpo --config "$CFG" --real-model $WANDB_FLAG | tee training/dumps/smoke.log
echo "smoke run complete at $(date)"

# quick eval against the trained adapter (the harness picks the latest under
# training/checkpoints by default once we wire that path through; for smoke
# we just confirm metrics jsonl is non-empty).
python - <<'PY'
from pathlib import Path
import json
p = Path("training/dumps/smoke_metrics.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"smoke metrics rows: {len(rows)}")
if rows:
    last = rows[-1]
    print(f"last step: {last.get('step')} reward: {last.get('total_reward'):.3f}")
    print(f"rolling mean 100: {last.get('rolling_mean_100'):.3f}")
PY
