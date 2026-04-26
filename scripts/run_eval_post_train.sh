#!/usr/bin/env bash
# evaluate a trained adapter against the 30 held-out cases. produces
# eval/report_<label>.md and re-renders demo/plots/ with real data.
#
# usage: scripts/run_eval_post_train.sh <adapter_path> <label> [config_path] [metrics_jsonl]
# example (t4 colab):
#   scripts/run_eval_post_train.sh \
#     training/checkpoints/adapter_final_timeboxed_t4.lora \
#     t4 \
#     training/configs/advisor_t4.yaml \
#     training/dumps/phase1_t4_metrics.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER="${1:?usage: $0 <adapter_path> <label> [config_path] [metrics_jsonl]}"
LABEL="${2:?usage: $0 <adapter_path> <label> [config_path] [metrics_jsonl]}"
CONFIG="${3:-training/configs/advisor_t4.yaml}"
METRICS="${4:-training/dumps/phase1_t4_metrics.jsonl}"

if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: adapter not found at $ADAPTER"
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"
  exit 1
fi

OUT="eval/report_${LABEL}.md"

python - "$ADAPTER" "$LABEL" "$CONFIG" "$METRICS" <<'PY'
"""evaluate the trained adapter and re-render plots with three model bars
(constant + scripted + trained). reuses the t4 training config so the loaded
model matches what was actually trained."""
import sys
from pathlib import Path

from eval.baselines.prompted_baseline import build_prompted_baseline
from eval.baselines.scripted_baseline import build_scripted_baseline
from eval.eval_harness import run_eval, write_report
from eval.plot import load_training_jsonl, render_all
from training._real_policy import build_unsloth_grpo_chat
from training.train_grpo import TrainConfig

adapter_path, label, config_path, metrics_path = sys.argv[1:5]
root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()

cfg = TrainConfig.from_yaml(Path(config_path))
cfg.resume_adapter = Path(adapter_path)

print(f"loading trained chat from {adapter_path} (config={config_path})...")
chat, _ = build_unsloth_grpo_chat(cfg)
trained_advisor = build_prompted_baseline(chat)

print("running scripted baseline...")
scripted_report = run_eval(advisor=build_scripted_baseline(), model_label="scripted")

# constant 'always-finalize' baseline for context (matches render_demo_plots).
from nyaya_mitra.interface import (
    ActionPlan,
    ApplicationPath,
    FreeLegalAidContact,
    LegalRouteRecommendation,
    PlainSummary,
    SchemeRecommendation,
)
from nyaya_mitra.interface.actions import Finalize


def always_finalize_advisor(observation, state):
    plan = ActionPlan(
        schemes=[
            SchemeRecommendation(
                scheme_id="pm_kisan",
                rationale_facts=[],
                required_documents=["Aadhaar"],
                application_path=ApplicationPath(),
            )
        ],
        legal_routes=[
            LegalRouteRecommendation(
                framework_id="domestic_violence_act_2005",
                applicable_situation="generic",
                forum="magistrate",
                procedural_steps=["x"],
                free_legal_aid_contact=FreeLegalAidContact(
                    authority="DLSA", contact_id="dlsa_ludhiana"
                ),
                required_documents=["id"],
            )
        ],
        most_important_next_step="contact dlsa",
        plain_summary=PlainSummary(language="en", text="please follow up."),
    )
    return Finalize(plan=plan)


print("running constant baseline...")
constant_report = run_eval(advisor=always_finalize_advisor, model_label="constant")

print(f"running trained model on 30 cases (this is the slow part — ~15min on T4)...")
trained_report = run_eval(advisor=trained_advisor, model_label=f"trained-{label}")

print("writing reports...")
write_report(scripted_report, Path("eval/report_scripted.md"))
write_report(trained_report, Path(f"eval/report_{label}.md"))

print("re-rendering plots with three-bar comparison...")
metrics = load_training_jsonl(Path(metrics_path))
render_all(
    training_steps=metrics,
    eval_results={
        "constant": constant_report["per_cohort"],
        "scripted": scripted_report["per_cohort"],
        f"trained": trained_report["per_cohort"],
    },
    out_dir=Path("demo/plots"),
)

print()
print("=== headline numbers (overall, all 30 cases) ===")
for name, rep in [("constant", constant_report), ("scripted", scripted_report), ("trained", trained_report)]:
    overall = rep["overall"]
    integrated = rep["per_cohort"]["integrated"]
    print(
        f"{name:>9}: mean_reward={overall.mean_total_reward:.3f} "
        f"pct_finalized={overall.pct_finalized:.0f}% "
        f"pct_gates_clean={overall.pct_all_gates_passed:.0f}% "
        f"integrated_solved={integrated.pct_integrated_solved:.0f}%"
    )
PY

echo ""
echo "=== done ==="
echo "report:    eval/report_${LABEL}.md"
echo "plots:     demo/plots/  (now with constant + scripted + trained bars)"
