#!/usr/bin/env bash
# evaluate a trained adapter against the 30 held-out cases. produces
# eval/report_<label>.md and re-renders demo/plots/ with real data.
#
# usage: scripts/run_eval_post_train.sh <adapter_path> <label>
# example: scripts/run_eval_post_train.sh training/checkpoints/adapter_final_warmup.lora phase1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER="${1:?usage: $0 <adapter_path> <label>}"
LABEL="${2:?usage: $0 <adapter_path> <label>}"

if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: adapter not found at $ADAPTER"
  exit 1
fi

OUT="eval/report_${LABEL}.md"

python - <<PY
"""evaluate the trained adapter and re-render plots."""
from pathlib import Path
import json

from eval.baselines.prompted_baseline import build_prompted_baseline
from eval.baselines.scripted_baseline import build_scripted_baseline
from eval.eval_harness import run_eval, write_report
from eval.plot import load_training_jsonl, render_all
from training._real_policy import build_unsloth_grpo_chat
from training.train_grpo import TrainConfig

cfg = TrainConfig.from_yaml(Path("$ROOT/training/configs/advisor_warmup.yaml"))
cfg.resume_adapter = Path("$ADAPTER")

print("loading trained chat...")
chat, _ = build_unsloth_grpo_chat(cfg)
trained_advisor = build_prompted_baseline(chat)

print("running scripted baseline...")
scripted_report = run_eval(advisor=build_scripted_baseline(), model_label="scripted-baseline")

print("running trained model on 30 cases...")
trained_report = run_eval(advisor=trained_advisor, model_label="trained-${LABEL}")

print("writing reports...")
write_report(scripted_report, Path("$ROOT/eval/report_scripted.md"))
write_report(trained_report, Path("$ROOT/eval/$OUT"))

print("re-rendering plots with real data...")
metrics = load_training_jsonl(Path("$ROOT/training/dumps/phase1_metrics.jsonl"))
render_all(
    training_steps=metrics,
    eval_results={
        "scripted": scripted_report["per_cohort"],
        "trained-${LABEL}": trained_report["per_cohort"],
    },
    out_dir=Path("$ROOT/demo/plots"),
)

print("=== headline numbers ===")
for label, rep in [("scripted", scripted_report), ("trained-${LABEL}", trained_report)]:
    overall = rep["overall"]
    print(f"{label}: mean={overall.mean_total_reward:.3f} gates={overall.pct_all_gates_passed:.1f}% "
          f"integrated={rep['per_cohort']['integrated'].pct_integrated_solved:.1f}%")
PY

echo "=== done ==="
echo "report:    eval/$OUT"
echo "plots:     demo/plots/"
