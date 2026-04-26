"""render the 6 demo plots from the local-smoke training jsonl + scripted+vanilla
baseline eval. produces demo/plots/*.png with real (not placeholder) data so the
demo dir has shipped artifacts even before a real GPU training run.

once a real Phase 1 run lands `training/dumps/phase1_metrics.jsonl`, re-run this
script with `--training-jsonl training/dumps/phase1_metrics.jsonl` to overwrite
the plots with real-training data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.eval_harness import run_eval  # noqa: E402
from eval.metrics import CohortMetrics  # noqa: E402
from eval.plot import load_training_jsonl, render_all  # noqa: E402


def _eval_results_from_run(report: dict) -> dict[str, CohortMetrics]:
    return dict(report["per_cohort"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-jsonl",
        default="training/dumps/local_smoke_metrics.jsonl",
        help="path to a metrics jsonl from a training run",
    )
    ap.add_argument("--out-dir", default="demo/plots")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    training_jsonl = ROOT / args.training_jsonl

    print(f"loading training metrics from {training_jsonl}...")
    if training_jsonl.exists():
        steps = load_training_jsonl(training_jsonl)
        print(f"  loaded {len(steps)} steps")
    else:
        print(f"  not found — plots will be placeholders for the training-side panels")
        steps = []

    print("running scripted baseline eval over 30 held-out cases...")
    from eval.baselines.scripted_baseline import build_scripted_baseline

    scripted_report = run_eval(advisor=build_scripted_baseline(), model_label="scripted")
    print(f"  scripted overall mean reward: {scripted_report['overall'].mean_total_reward:.3f}")

    print("running 'always-finalize' constant baseline (proxy for vanilla)...")
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

    vanilla_report = run_eval(advisor=always_finalize_advisor, model_label="constant")
    print(f"  constant overall mean reward: {vanilla_report['overall'].mean_total_reward:.3f}")

    eval_results = {
        "constant": _eval_results_from_run(vanilla_report),
        "scripted": _eval_results_from_run(scripted_report),
    }

    print(f"rendering 6 plots to {out_dir}...")
    paths = render_all(training_steps=steps, eval_results=eval_results, out_dir=out_dir)
    for name, p in paths.items():
        size = p.stat().st_size if p.exists() else 0
        print(f"  {name}  ({size} bytes)")

    print("done.")


if __name__ == "__main__":
    main()
