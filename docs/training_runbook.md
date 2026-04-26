# Training runbook

How to spend 60 HF credits ($60) on a real GRPO training run for nyaya-mitra.

## TL;DR

```
total budget: $60 / ~15 hours of A100 large at $4/hr
  $4    1h smoke run on Qwen 2.5 0.5B (verify pipeline)
  $24   6h phase 1 warmup, Qwen 2.5 3B, 500 episodes
  $20   5h phase 2 co-training, 1000 episodes
  $2    0.5h eval rerun + plot regeneration
  $2    0.5h HF Space deploy + smoke
  $8    2h buffer for re-runs / debugging
```

If everything works first try, you finish with $8 unused. If smoke run finds a
bug, the first $4 saved you $50.

## Decision tree

```
Have you run scripts/run_smoke.sh successfully?
├─ no  → start there. don't burn credits on phase 1 with broken code.
├─ yes
   └─ Phase 1 rolling-100 mean ever exceeded 0.3?
      ├─ no  → stop. inspect transcripts in training/dumps/. don't run phase 2
      │        on a phase-1 adapter that didn't learn anything.
      └─ yes → run phase 2.
```

## Setup (do this on the A100 host before timing starts)

```bash
git clone https://github.com/parthtaneja0001/nyaya-mitra.git
cd nyaya-mitra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[track_a,track_b,dev,train]"
pip install vllm  # speeds up Unsloth fast_inference path

# verify
pytest tests/track_b -q  # should be 192 passing
python -c "from training._real_policy import build_unsloth_grpo_chat; print('imports ok')"

# auth
export HF_TOKEN=hf_xxx
huggingface-cli login --token $HF_TOKEN
export WANDB_API_KEY=...  # optional
```

## Phase 0: smoke run (~$4, 1h)

```bash
./scripts/run_smoke.sh
```

What it does:
- Loads Qwen 2.5 **0.5B** Instruct (small enough to fail fast).
- 50 episodes against the easy curriculum.
- Confirms: model loads, rollout completes, reward fires, GRPO update runs,
  metrics jsonl writes, adapter saves.

Success criteria:
- `training/dumps/smoke.log` ends with no traceback.
- `training/dumps/smoke_metrics.jsonl` has 50 rows.
- `training/checkpoints/adapter_step_000025.lora/` exists.
- last row's `total_reward` is a finite float.

If any of these fail, fix the bug locally, then re-run. You have $56 left.

## Phase 1: warmup (~$24, 6h)

```bash
./scripts/run_phase1.sh
```

What it does:
- Loads Qwen 2.5 **3B** Instruct, 4-bit + LoRA via Unsloth.
- 500 episodes against the easy/medium/hard mix in `advisor_warmup.yaml`.
- Per-step logs to `training/dumps/phase1_metrics.jsonl` + W&B if configured.
- Snapshots LoRA adapter every 200 steps.
- Aborts early if rolling-100 mean stays below 0.10 after step 500 (PLAN B.5).

Watch for:
- `rolling_mean_100` should cross **0.30** by step ~300.
- gate trigger frequency should drop after ~step 150 (model learns format).
- transcript dumps in `training/dumps/step_000100/` etc — eyeball one every
  ~hour to confirm the model isn't doing something pathological.

If phase 1 mean reward is still below 0.10 at step 500: **don't run phase 2.**
Instead spend $2 on:

```bash
./scripts/run_eval_post_train.sh training/checkpoints/adapter_final_warmup.lora phase1-failed
```

Inspect `eval/report_phase1-failed.md`. The components breakdown will show
which dimension is broken (likely `scheme_recall` or `fact_coverage`).

## Phase 2: co-trained adversarial (~$20, 5h)

```bash
./scripts/run_phase2.sh
```

What it does:
- Resumes from `adapter_final_warmup.lora`.
- Runs 1000 episodes (cut from PLAN's 2000 for budget).
- Generator + advisor alternate per the `alternation` block in `phase2.yaml`.

Watch for:
- The diversity-penalty signal in the generator's reward should keep
  generator outputs varied (no mode collapse).
- Advisor's rolling reward should stay above phase-1 final or improve;
  if it crashes, the generator is producing pathological cases.

## Eval + plot regeneration (~$2, 30min)

```bash
./scripts/run_eval_post_train.sh training/checkpoints/adapter_final_phase2.lora phase2
```

What it does:
- Runs the trained model against all 30 held-out eval cases.
- Runs the scripted baseline for comparison.
- Renders `eval/report_phase2.md`.
- Regenerates all 6 PNGs in `demo/plots/` with **real** training and eval data.

The headline number in `eval/report_phase2.md`:
- **Mean total reward** — should be **above 0.509** (the scripted baseline floor).
- **Integration solve rate** — should be **above 50%** for real RL win.
- **Sensitivity F1** — should be **above 0.15** (the scripted baseline's gap).

## HF Space deploy (~$2, 30min)

```bash
export HF_SPACE_REPO=parthtaneja0001/nyaya-mitra-env
./scripts/deploy_space.sh
```

What it does:
- Logs into HF with `HF_TOKEN`.
- Creates the Space if it doesn't exist (Docker SDK).
- Pushes current branch to Space's `main`.
- Prints the Space URL.

The HF Space hosts the env (not the trained model — that's separate). Submitting
the env to the hackathon means hosting it here.

## Buffer ($8, ~2h)

Reality: something will go wrong. Common modes:
- OOM on 3B at the chosen `max_completion_length`. Drop to 384 tokens.
- Unsloth fast-inference path uses vllm which fights with PEFT in some configs.
  Fall back to `model.generate(...)` without vllm by setting `use_vllm: false`
  in the config (cuts speed ~2x but unblocks).
- `chat.grpo_step` raises during a long run. Inspect `training/dumps/phase1.log`;
  the train loop catches and logs without crashing, but burst failures are bad.

## What to do if you run out of credits before phase 2

**Skip phase 2.** Phase 1 alone with good curves is enough to win on
"showing improvement in rewards." Do the eval + plot rerender + HF Space
deploy with whatever budget remains. The headline in `eval/report_phase1.md`
becomes the demo number. Document it as "phase 1 only; phase 2 future work."

## What to do if smoke run finds a bug too late

Fix it locally on CPU (orchestration is fully testable without GPU — see
`tests/track_b/test_train_grpo.py` and `tests/track_b/test_grpo_hook.py`),
push, pull on the A100 host, re-run smoke. Each smoke run is $4. Budget two.

## Submission deliverables produced by this runbook

After all phases complete you will have:
- `training/checkpoints/adapter_final_phase1.lora/` (or `_phase2.lora/`)
- `training/dumps/phase1_metrics.jsonl` (real W&B-exportable training log)
- `training/dumps/phase2_metrics.jsonl`
- `eval/report_phase2.md` — real headline numbers
- `demo/plots/*.png` — six real-data PNGs
- `demo/transcripts/*.md` — baseline-vs-trained side-by-side (already exist
  for scripted; rerun against trained for the demo set)
- HF Space URL hosting the env

That's the full submission package per PLAN B.9.
