#!/usr/bin/env bash
# phase 2 co-training: advisor + generator alternation. budget ~5h on A100.
# resumes from the phase 1 adapter; cuts num_episodes to 1000 (vs PLAN's 2000)
# to fit 60-credit budget.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PHASE1_ADAPTER="$ROOT/training/checkpoints/adapter_final_warmup.lora"
if [ ! -d "$PHASE1_ADAPTER" ]; then
  echo "ERROR: phase 1 adapter not found at $PHASE1_ADAPTER"
  echo "run scripts/run_phase1.sh first."
  exit 1
fi

# patch the phase2 yaml's num_episodes down to 1000 + resume_adapter pointer.
CFG="$ROOT/training/configs/advisor_phase2_budget.yaml"
python - <<PY
import yaml
from pathlib import Path

src = Path("$ROOT/training/configs/advisor_phase2.yaml").read_text()
cfg = yaml.safe_load(src)
cfg.setdefault("model", {})["resume_adapter"] = "$PHASE1_ADAPTER"
cfg.setdefault("grpo", {})["num_episodes"] = 1000
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote $CFG with num_episodes=1000, resume from $PHASE1_ADAPTER")
PY

echo "=== phase 2 co-training ==="
echo "config: $CFG"
echo "budget: ~5h on A100"
echo "==="

WANDB_FLAG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_FLAG="--wandb"
fi

python -m training.train_grpo --config "$CFG" --real-model $WANDB_FLAG | tee training/dumps/phase2.log

echo "phase 2 complete. final adapter: training/checkpoints/adapter_final_phase2.lora"
