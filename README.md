# Nyaya Mitra

A paralegal-cum-welfare-advisor RL environment for vulnerable Indian citizens. OpenEnv-compatible HTTP contract. GRPO-trained on a free Colab T4. Anti-reward-hacking by construction.

> *"What if the agent had to choose between giving you advice and routing you to a real lawyer — and the env made the second one structurally unavoidable?"*

## Submission materials

| | |
|---|---|
| 🚀 **Live HF Space** | <https://huggingface.co/spaces/parthtaneja0001/nyaya-mitra-env> ([healthz](https://parthtaneja0001-nyaya-mitra-env.hf.space/healthz)) |
| 🎬 **Demo video** | _link added after recording_ |
| 📓 **Training notebook (Colab)** | [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb) — [open in Colab](https://colab.research.google.com/github/parthtaneja0001/nyaya-mitra/blob/main/training/train_grpo_colab.ipynb) |
| 📊 **Training plots** | [`demo/plots/`](demo/plots/) — six PNGs from a real 176-episode run on free Colab T4 |
| 🧱 **OpenEnv manifest** | [`openenv.yaml`](openenv.yaml) — declares the FastAPI entrypoint, runtime, port |
| 🛡️ **Reward design + gates** | [`docs/reward_design.md`](docs/reward_design.md) |
| 📦 **Repo source** | <https://github.com/parthtaneja0001/nyaya-mitra> |

## OpenEnv compatibility

The env exposes the standard OpenEnv HTTP contract — `POST /reset`, `POST /step`, `POST /close`, `GET /state`, `GET /healthz` — with JSON request/response bodies driven by Pydantic schemas in [`src/nyaya_mitra/interface/`](src/nyaya_mitra/interface/). It's a Gym-style `Environment` class wrapped in FastAPI ([`src/nyaya_mitra/env/server.py`](src/nyaya_mitra/env/server.py)), declared via [`openenv.yaml`](openenv.yaml), containerized via [`Dockerfile`](Dockerfile), and deployed as the Space linked above. Discoverable, runnable, framework-agnostic.

## What this is

A multi-turn conversational environment where an LLM advisor must:
1. Elicit facts from a vulnerable citizen via `Ask` and (sensitive-topic-gated) `Probe` actions.
2. Explain in plain language at the citizen's literacy level.
3. Finalize an `ActionPlan` that routes to **specific** government schemes (`pm_kisan`, `pmuy`, `mgnrega`, `pm_awas_grameen`, `ayushman_bharat`, `pmsby` …) and **specific** legal frameworks (DV Act 2005, Maternity Benefit Act 1961, Minimum Wages Act 1948, Consumer Protection Act 2019 …) — every legal route carrying a real `(NALSA|SLSA|DLSA, contact_id)` for free legal aid.

The technical story is **anti-reward-hacking by construction**. See [`docs/reward_design.md`](docs/reward_design.md) and the gates section of this README.

## Why it's different

Most RL hackathon submissions train on toy puzzles. This one trains on something that maps 1-1 onto a real public-interest gap: 600M+ Indians eligible for welfare schemes don't claim them because navigating eligibility + applications is opaque. An LLM advisor that *correctly* routes citizens to the right schemes + legal aid is a meaningful target. Reward hacking is the failure mode that would make this dangerous; we've made it the centerpiece, not an afterthought.

## Architecture (one screen)

```
┌────────────────────────┐    HTTP / in-process     ┌────────────────────────┐
│   Track A — World      │ ───────────────────────▶ │   Track B — Agent      │
│   src/nyaya_mitra/     │                          │   src/nyaya_mitra/     │
│   ├─ env/              │ AdvisorAction (Ask/      │   ├─ rewards/          │
│   ├─ citizen/          │  Probe/Explain/          │   ├─ case_gen/         │
│   │  ├─ simulator      │  Finalize)               │   ├─ advisor/          │
│   │  └─ extractor ◀────┼─── citizen utterance ────┤   training/            │
│   ├─ knowledge/        │                          │   eval/                │
│   └─ profile/          │   info[reward_breakdown] │   demo/                │
└────────────────────────┘ ◀────────────────────────┴────────────────────────┘
                          src/nyaya_mitra/interface/  (shared schemas)
                          scripts/wire_rewards.py    (the only legitimate
                                                      cross-track import)
```

- **Track A** owns the world: env, KB (6 schemes + 4 frameworks + 20 DLSAs across 10 states), citizen sim (smart-canned, profile-driven, deterministic), fact extractor (23 patterns × en/hi/hinglish, negation-aware).
- **Track B** owns learning: 11 reward components + 4 gates + aggregator + per-turn shaping, GRPO trainer, adversarial case generator, eval harness.
- **`scripts/wire_rewards.py`** is the bootstrap glue — single legitimate cross-track import point, enforced by `tests/integration/test_interface_contract.py`.

## Anti-reward-hacking — the technical centerpiece

Four reward gates short-circuit `total = -1.0` when triggered:

| Gate | Triggers when … |
|---|---|
| `format_violation` | empty plan, blank `most_important_next_step`, blank summary, malformed action |
| `hallucination` | unknown `scheme_id`, `framework_id`, or `(authority, contact_id)` |
| `contradiction` | rationale_facts include something the citizen explicitly negated (extractor surfaces these via `info["negated_facts"]`) |
| `sim_leak` (passthrough) | sensitive fact revealed without a matching `Probe` — zeros elicitation shaping for that turn |

Plus structural anti-liability: `LegalRouteRecommendation` cannot be constructed without a `free_legal_aid_contact`. Pydantic enforces. The agent literally cannot give standalone "advice" — only routes.

Plus weight invariants:
- `validate_weights()` runs at import; soft components must sum to exactly 1.0
- No deterministic component above 15%; no LLM-judged component above 5%
- Per-turn shaping caps at +0.4 per episode (no loop-farming)

## Layout

```
nyaya-mitra/
├── src/nyaya_mitra/
│   ├── interface/       [S]   shared schemas (AdvisorAction, ActionPlan, …)
│   ├── env/             [A]   OpenEnv environment + FastAPI server + HTTP client
│   ├── citizen/         [A]   smart-canned simulator + deterministic extractor
│   ├── knowledge/       [A]   6 schemes + 4 frameworks + 20 DLSAs, all checkers
│   ├── profile/         [A]   14 seed profiles (easy/medium/hard) + ground-truth derivation
│   ├── rewards/         [B]   11 components + 4 gates + aggregator + shaping
│   ├── case_gen/        [B]   adversarial generator (in flight)
│   └── advisor/         [B]   advisor model wrapper
├── training/            [B]   GRPO trainer + configs + rollout
├── eval/                [B]   eval_harness + 30 held-out cases (10 per cohort) + plots
├── scripts/             glue: wire_rewards (cross-track), generate_eval_cases, deploy_space
├── tests/
│   ├── track_a/                track-a only — runs in ci-track-a
│   ├── track_b/                track-b only — runs in ci-track-b
│   └── integration/            shared contract — runs in ci-integration
├── docs/
│   ├── architecture.md         the seam between tracks
│   ├── reward_design.md        weights + gate semantics + invariants
│   ├── kb_coverage.md          KB inventory
│   ├── deploy.md               HF Space deploy steps
│   └── what_this_is_not.md     scope & liability framing
├── PLAN.md                     full two-track spec (immutable)
├── REPO_STRUCTURE.md           file ownership + branch model + PR rules
└── CLAUDE.md                   coord-server playbook for two-agent collaboration
```

## Quick start

```
git clone https://github.com/parthtaneja0001/nyaya-mitra.git
cd nyaya-mitra
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[track_a,track_b,dev]"
pytest tests/                              # 380+ tests should be green
```

Run the env locally:
```
uvicorn nyaya_mitra.env.server:app --port 8000
curl http://localhost:8000/healthz         # → {"status":"ok"}
```

Wire env + reward fn end-to-end (the path Track B's training uses):
```python
from scripts.wire_rewards import build_env
from nyaya_mitra.interface import Ask, Finalize, ActionPlan, ...

env = build_env(seed=0)
obs = env.reset()
res = env.step(Ask(question="tell me about your situation", language="en"))
res = env.step(Finalize(plan=ActionPlan(...)))
print(res.info["reward_breakdown"])        # full 19-key breakdown
```

Deploy to HF Space:
```
export HF_TOKEN=hf_...
./scripts/deploy_space.sh                  # see docs/deploy.md
```

## Numbers (env-side, not training)

- **6 schemes** + **4 frameworks** with parametrized eligibility/applicability tests (54+ checker tests)
- **20 DLSAs** across **10 states** + 1 NALSA — every seed-profile state has DLSA coverage
- **14 training seed profiles** + **30 held-out eval cases** (10 welfare-only / 10 legal-only / 10 integrated)
- **23 extractor patterns** + **72 golden tests** across en / hi / hinglish, with negation handling and absence-polarity facts
- **380+ total tests pass**, 0 skipped, ruff clean

## Training (real run on free Colab T4)

The Colab notebook at [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb) trains Qwen 2.5 0.5B Instruct (Unsloth + 4-bit + LoRA) for 100 episodes against the easy/medium curriculum. Config: [`training/configs/advisor_t4.yaml`](training/configs/advisor_t4.yaml). Metrics dump to [`training/dumps/phase1_t4_metrics.jsonl`](training/dumps/) — that's the source for the plots below, rendered by [`scripts/render_demo_plots.py`](scripts/render_demo_plots.py).

| Reward curve | Reward components |
|---|---|
| ![total reward](demo/plots/total_reward_curve.png) | ![components](demo/plots/reward_components_stacked.png) |

| Gate triggers | Sim-leak count |
|---|---|
| ![gates](demo/plots/gate_trigger_frequency.png) | ![sim_leak](demo/plots/sim_leak_over_training.png) |

| Baseline vs trained | Integration solve rate |
|---|---|
| ![baseline](demo/plots/baseline_vs_trained_bars.png) | ![integration](demo/plots/integration_solve_rate.png) |

The training loop adds a per-turn format-correctness shaping bonus (+0.05 per parseable advisor action, capped at 0.30/episode). This gives a non-zero gradient during cold-start when an untrained 0.5B model cannot yet emit a valid `FINALIZE` schema. Once the model masters JSON formatting (`n_parse_ok` saturating at 6/6), the env reward signal takes over.

## Submission artifacts

- 🚀 **Live HF Space**: <https://huggingface.co/spaces/parthtaneja0001/nyaya-mitra-env>
- 🎬 **Demo video**: _link added after recording_
- 📓 **Training notebook**: [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb)
- 📊 **Plots** (real-data from 176-episode run): [`demo/plots/`](demo/plots/)
- 🛡️ **Reward design + invariants**: [`docs/reward_design.md`](docs/reward_design.md)
- 📋 **Eval cases + transcripts**: `eval/eval_cases/` (30 held-out cases) + [`demo/transcripts/`](demo/transcripts/)
- ⚖️ **Scope + liability framing**: [`docs/what_this_is_not.md`](docs/what_this_is_not.md)
- 🧱 **OpenEnv manifest**: [`openenv.yaml`](openenv.yaml)
- 🐳 **Dockerfile** (HF Space runtime): [`Dockerfile`](Dockerfile)

## Workflow

Two Claude Code agents collaborate via [`claude-coord`](https://github.com/parthtaneja0001/claude-coord). See [`CLAUDE.md`](CLAUDE.md) for the playbook (file ownership, PR handshake, review rules). [`PLAN.md`](PLAN.md) is the immutable spec; [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md) is the conflict-resistant layout.
