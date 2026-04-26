# Submission checklist

Final actions to land the hackathon submission. Most of the engineering is in place; this is the wall-clock checklist for what only the user can run.

## 0. Pre-flight (already done)

- [x] env (Track A): runnable, FastAPI, /healthz, deterministic
- [x] KB: 12 schemes + 8 frameworks + 20 DLSAs across 10 states
- [x] 14 seed profiles + 30 held-out eval cases
- [x] rewards (Track B): 11 components + 4 gates + aggregator
- [x] GRPO training pipeline (TRL + Unsloth + LoRA + Colab notebook)
- [x] eval harness + 2 baselines + plot rendering
- [x] demo blog post + video script
- [x] 499 tests pass, ruff clean

## 1. Run Phase 1 training (~$24, 6h)

The Colab notebook is at `training/train_grpo_colab.ipynb`. Open it, set **Runtime → Change runtime type → A100 GPU**, then:

1. **Cell 7 (auth)**: replace `os.environ["HF_TOKEN"] = ...` with your write-scope HF token from https://huggingface.co/settings/tokens
2. (Optional) `os.environ["WANDB_API_KEY"] = ...` for live W&B charts
3. **Run all** (Runtime → Run all)
4. Wait for the smoke run (Cell 11) to succeed before letting Phase 1 (Cell 15) start
5. Phase 1 takes ~6h on A100; watch `rolling_mean_100` cross 0.30 around step 300

If Phase 1 mean reward stays below 0.10, **stop**. Don't run Phase 2 on a failed adapter. Inspect `training/dumps/phase1_metrics.jsonl` and `training/dumps/step_*` transcript dumps; common issues:
- The smart-canned citizen sim returns the same utterance shape too often → reduce `num_generations` in the config
- The reward hits gates too often early → check format / hallucination patterns

## 2. (Optional, conditional on #1 success) Run Phase 2 (~$20, 5h)

Only if Phase 1 rolling-100 mean exceeded 0.30. Run Cell 19. 1000 episodes of co-trained adversarial. Stop early if generator collapses to a single mode (diversity penalty in metrics goes to ~0).

## 3. Re-render demo plots from real data (~$2, 30 min)

After training, the metrics are in `training/dumps/phase1_metrics.jsonl` (and `phase2_metrics.jsonl` if Phase 2 ran). Re-render:

```
python scripts/render_demo_plots.py --training-jsonl training/dumps/phase1_metrics.jsonl
```

This overwrites the placeholder plots in `demo/plots/` with real-data ones. Six PNGs, each labeled with axes + units.

## 4. Re-run eval against the trained adapter

```
./scripts/run_eval_post_train.sh training/checkpoints/adapter_final_warmup.lora phase1
```

Produces `eval/report_phase1.md` with the real headline numbers. The headline:
- **Mean total reward** — should be above 0.509 (the scripted baseline floor)
- **Integration solve rate** — target above 50%
- **Sensitivity F1** — target above 0.15

## 5. Deploy the env to HF Space (~$2, 30 min)

```
export HF_TOKEN=hf_xxxxxxxxxxxxx
export HF_SPACE_REPO=parthtaneja0001/nyaya-mitra-env
./scripts/deploy_space.sh
```

The script logs in, creates the Space (Docker SDK) if missing, pushes current branch as Space's `main`, prints the URL. ~2 minutes for the Docker build to complete on HF's side.

Smoke test once up:
```
curl https://parthtaneja0001-nyaya-mitra-env.hf.space/healthz
# → {"status":"ok"}
```

## 6. Refresh the README and demo dir with real numbers

After steps 3-5, update `README.md` and `demo/blog_post.md` placeholders that say "fill in real numbers after Phase 1 training run" with the actual numbers from `eval/report_phase1.md`.

The 3 demo transcripts in `demo/transcripts/` are currently scripted-baseline-only. Re-run the trained adapter against `int_001`, `leg_003`, `wel_005` and append the trained-side transcript to each (the existing `transcript_renderer.py` outputs the structure).

## 7. Record the video (under 2 minutes)

The script is at `demo/video_script.md`. Beats sized for 90 seconds:
- 0:00-0:08 problem hook (Hindi citizen + tangled scheme websites)
- 0:08-0:18 why LLMs are hard here
- 0:18-0:35 four gates + schema invariant
- 0:35-0:52 live transcript scroll (the trained agent on a hard case)
- 0:52-1:10 the numbers (real Phase 1 plots)
- 1:10-1:25 architecture (two-track + cross-track-imports test)
- 1:25-1:30 URLs

Upload to YouTube as unlisted, link from the README.

## 8. Submit

Per the hackathon submission requirements:
- ✅ OpenEnv-compliant environment hosted on HF Spaces
- ✅ Working training script using TRL + Unsloth (in `training/train_grpo.py` + Colab)
- ✅ Evidence of training (loss + reward plots)
- ✅ Mini-blog (`demo/blog_post.md`) — publish to HF Hub or GitHub Pages
- ✅ <2 min YouTube video
- ✅ README with motivation, env explanation, results

Repo: https://github.com/parthtaneja0001/nyaya-mitra

Submission link goes in the hackathon submission portal with all of the above linked.

## Known gaps (document these explicitly)

- **Real LLM citizen sim**: smart-canned stand-in shipped; PLAN.md A.4 wants a frozen Qwen 2.5 3B Instruct or Llama 3.2 3B. Not done — the storytelling acknowledges this.
- **18 more KB schemes / 7 more frameworks**: target was 30+15; we're at 12+8. Not done — the architecture supports growth via JSON+checker pattern.
- **46 more seed profiles**: target was 60; we're at 14. Not done.
- **128 more extractor goldens**: target was 200+; we're at 72. Not done.
- **Phase 3 frozen-generator eval**: stretch goal, dependent on Phase 2 success.

These are documented in the README and `docs/what_this_is_not.md`. The submission positions them honestly as out-of-scope-for-hackathon-timeline rather than oversights.
