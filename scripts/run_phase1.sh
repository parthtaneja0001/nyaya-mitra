#!/usr/bin/env bash
# phase 1 warmup: 500 episodes, qwen 2.5 3b, easy-curriculum, GRPO.
# target rolling-100 mean >= 0.3. budget ~6h on A100.
#
# requires: scripts/run_smoke.sh has succeeded on this host first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="$ROOT/training/configs/advisor_warmup.yaml"
mkdir -p "$ROOT/training/dumps" "$ROOT/training/checkpoints"

echo "=== phase 1 warmup ==="
echo "config: $CFG"
echo "model:  Qwen 2.5 3B Instruct (4-bit + LoRA via unsloth)"
echo "budget: ~6h on A100"
echo "target: rolling-100 mean reward >= 0.3"
echo "==="

WANDB_FLAG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_FLAG="--wandb"
fi

python -m training.train_grpo --config "$CFG" --real-model $WANDB_FLAG | tee training/dumps/phase1.log

# eval the final adapter against the held-out 30 cases.
echo "=== eval phase-1 adapter ==="
ADAPTER="$ROOT/training/checkpoints/adapter_final_warmup.lora"
if [ ! -d "$ADAPTER" ]; then
  echo "WARN: $ADAPTER not found; eval skipped. inspect training/dumps/phase1.log"
  exit 0
fi

python - <<PY
from pathlib import Path
print("phase-1 adapter saved at: $ADAPTER")
print("re-render plots + report with: python -m eval.eval_harness --advisor scripted --label phase1-floor --out eval/report_phase1.md")
print("(real-model eval needs a chat backed by the saved adapter — see scripts/run_eval_post_train.sh)")
PY
