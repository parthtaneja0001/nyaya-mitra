#!/usr/bin/env bash
# timeboxed Phase 1: ~2h on Colab T4. Qwen 2.5 0.5B + 4-bit + LoRA, 150 episodes.
# fits the hackathon submission deadline. burns no HF credit (Colab T4 is free).
#
# pre-flight (paste in a Colab cell BEFORE this):
#   import os
#   os.environ["HF_TOKEN"] = "hf_xxx"           # write-scope token
#   # optional: os.environ["WANDB_API_KEY"] = "..."
#
# success criteria:
#   - training/dumps/phase1_timeboxed_metrics.jsonl has 150 rows
#   - last row's rolling_mean_50 > first row's (any upward trend wins)
#   - training/checkpoints/adapter_step_000150.lora/ exists
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="$ROOT/training/configs/advisor_timeboxed.yaml"
mkdir -p "$ROOT/training/dumps" "$ROOT/training/checkpoints"

echo "=== timeboxed phase 1 ==="
echo "config: $CFG"
echo "model:  Qwen 2.5 0.5B Instruct (4-bit + LoRA)"
echo "budget: ~2h on T4 / ~30min on A10G"
echo "target: any upward trend in rolling_mean_50"
echo "==="

WANDB_FLAG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_FLAG="--wandb"
fi

python -m training.train_grpo --config "$CFG" --real-model $WANDB_FLAG | tee training/dumps/phase1_timeboxed.log

# quick check: did the metrics jsonl get rows?
python - <<'PY'
import json
from pathlib import Path
p = Path("training/dumps/phase1_timeboxed_metrics.jsonl")
if not p.exists():
    print("FAIL: no metrics jsonl. inspect training/dumps/phase1_timeboxed.log")
    exit(1)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"rows: {len(rows)}")
if rows:
    first = rows[0]
    last = rows[-1]
    print(f"first step={first.get('step')} reward={first.get('total_reward'):.3f}")
    print(f"last  step={last.get('step')} reward={last.get('total_reward'):.3f} rolling_mean_50={last.get('rolling_mean_50') or last.get('rolling_mean_100'):.3f}")
PY

echo "=== render plots from real metrics ==="
python scripts/render_demo_plots.py --training-jsonl training/dumps/phase1_timeboxed_metrics.jsonl

echo ""
echo "=== timeboxed run complete ==="
echo "artifacts:"
echo "  - training/dumps/phase1_timeboxed_metrics.jsonl    (the curve)"
echo "  - training/checkpoints/adapter_step_*.lora/         (snapshots)"
echo "  - demo/plots/*.png                                  (re-rendered with real data)"
echo ""
echo "next: commit + push the artifacts, then re-deploy the Space"
