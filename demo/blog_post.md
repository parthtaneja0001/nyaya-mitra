---
title: "Nyaya Mitra: an RL environment where reward hacking is the technical centerpiece"
authors: parthtaneja0001
tags: [openenv, grpo, reinforcement-learning, india, welfare, legal-aid, anti-reward-hacking]
---

# Nyaya Mitra: an RL environment where reward hacking is the technical centerpiece

> *Built for the OpenEnv Hackathon (India 2026). Themes: 3.2 (Personalized Tasks), 4 (Self-Improvement), 2 (Long-Horizon).*

## The gap

Over 600 million Indians are eligible for at least one welfare scheme they don't claim — and a similar number have a legal grievance (DV, denied maternity leave, underpayment, defective product) that a paralegal could route to free legal aid. The bottleneck is not money; it's a conversational layer that elicits facts, knows the schemes + frameworks, and routes correctly without giving "advice" that could harm.

That's an LLM job. So why isn't this solved?

Because the failure mode of an LLM advisor is *worse* than no advisor:

- Hallucinated scheme IDs send people to non-existent offices
- Confidently-wrong eligibility wastes a citizen's bus fare and trust
- "I am a lawyer, here is what I think you should do" attracts liability
- Pretty-sounding plans optimized for a judge model can still be useless

So when we trained an LLM advisor with GRPO, we made the failure mode the *centerpiece* of the design.

## The environment

`Nyaya Mitra` is an OpenEnv-compliant multi-turn environment where a citizen (a deterministic profile-driven sim — never an LLM during training, by construction) brings a vague welfare or legal grievance. The advisor's only outputs are:

- `Ask` (open question)
- `Probe` (sensitive-topic-gated question — DV / caste / disability / immigration / mental health)
- `Explain` (literacy-level-aware teaching)
- `Finalize(plan: ActionPlan)` — the terminal, structured plan

The plan must list specific `scheme_id`s from a 12-scheme KB and specific `framework_id`s from an 8-framework KB. Every legal route **must** include a `free_legal_aid_contact` to a real (NALSA / SLSA / DLSA, contact_id) tuple; the schema refuses construction without one. The agent literally cannot give standalone "advice" — only routes.

## How we made reward hacking the headline

Four hard reward gates short-circuit `total = -1.0`:

1. **`format_violation`** — empty plan, blank summary, malformed action.
2. **`hallucination`** — any unknown `scheme_id`, `framework_id`, or `(authority, contact_id)` in the plan. Hand-wave a contact and you score -1.
3. **`contradiction`** — a `rationale_fact` cited in the plan was *negated* by the citizen during the conversation. Track A's deterministic fact extractor surfaces both positive and negative mentions; the gate compares.
4. **`sim_leak` (passthrough)** — the citizen sim accidentally volunteered a sensitive fact without a matching `Probe`. The reward zeros the elicitation shaping for that turn (we don't reward the agent for the sim's mistake) but doesn't fail the episode.

Plus structural and weight invariants:

- **Schema invariant**: `LegalRouteRecommendation` requires `free_legal_aid_contact` at construction time. Pydantic enforces.
- **Weight invariant**: `validate_weights()` runs at module import; soft components must sum to exactly 1.0; no deterministic component above 15%, no LLM-judged component above 5%.
- **Per-turn shaping cap**: positive shaping caps at +0.4 per episode (no loop farming).
- **Cross-track imports**: an AST walker test mechanically forbids Track A modules from importing Track B internals and vice versa. Single legitimate cross-track point: `scripts/wire_rewards.py` (the bootstrap glue).

The agent that learns to "exploit the reward without solving the task" — the exact failure mode the OpenEnv guide flags as #1 — gets -1.0 instantly. The agent that learns to elicit facts, route to real DLSA contacts, and explain at the citizen's literacy level wins.

## What it looks like in practice

A wary 27-year-old garment worker in Bihar, pregnant, denied maternity leave, with DV at home, low literacy, prefers Hindi. She opens with: *"मुझे काम पर परेशानी है, और घर में भी कुछ ठीक नहीं है।"* (work problem, also things at home aren't ok)

The advisor's job over up to 20 turns:

1. `Ask` her about the work issue → she discloses pregnancy and the denial → extractor adds `pregnant_or_postpartum` and `denied_maternity_benefit` to elicited_facts
2. `Probe` (sensitive_topic="dv") on the home situation → she discloses (only after 2 advisor turns of trust building, since she's wary)
3. `Explain` at literacy="low" the difference between maternity benefit (Inspector / Labour Commissioner under MB Act 1961) and DV protection orders (Magistrate under DV Act 2005) — without legal jargon (jargon detector docks the reward)
4. `Finalize` a plan that:
   - Recommends `pmuy` + `pmsby` + `apy` + `pmjjby` (welfare she qualifies for)
   - Routes `maternity_benefit_act_1961` → DLSA Muzaffarpur
   - Routes `domestic_violence_act_2005` → DLSA Muzaffarpur
   - Cites `dv_present` and `denied_maternity_benefit` in `rationale_facts` (both elicited, both must be in the transcript)

Score this and you get a single number. Hallucinate `dlsa_xyz` → -1. Cite `dv_present` if the citizen *denied* it during the conversation → -1.

## The numbers (env-side)

- **12 schemes** + **8 frameworks** with parametrized eligibility/applicability tests (90+ checker tests)
- **20 DLSAs** across **10 states** + 1 NALSA
- **14 training seed profiles** + **30 held-out eval cases** (10 welfare-only / 10 legal-only / 10 integrated)
- **23 extractor patterns** + **72 golden tests** across en / hi / hinglish, with negation handling and absence-polarity facts
- **499 tests pass green**, 0 skipped, ruff clean
- **Determinism guard**: same seed → same sha256 over the full transcript + ground-truth + reward breakdown

## The numbers (training)

[fill in real numbers after Phase 1 training run; current plots in `demo/plots/` are from a 30-episode FakeChat smoke that exercises the rendering pipeline]

## What this is not

- Not a substitute for a lawyer (every route ends at NALSA/SLSA/DLSA — schema invariant)
- Not a benefits-application service (cites canonical .gov.in URLs and offline offices only)
- Not a chatbot (terminal output is the structured `ActionPlan`, not free text)
- Not exhaustive (12/30 schemes, 8/15 frameworks)
- Not a real LLM citizen sim yet (smart-canned + deterministic, swap planned)
- Not done (case_gen + Phase 2 co-training are stretch goals beyond the hackathon timeline)

## The repo

[`github.com/parthtaneja0001/nyaya-mitra`](https://github.com/parthtaneja0001/nyaya-mitra)

Two-track collaboration via `claude-coord`: Track A owns world (env, KB, sim, profiles, extractor); Track B owns learning (rewards, gates, aggregator, GRPO trainer, eval, demo). Single legitimate cross-track import is `scripts/wire_rewards.py`. CI is split three ways so a flaky test on one side never blocks the other.

## Run it locally

```
git clone https://github.com/parthtaneja0001/nyaya-mitra.git && cd nyaya-mitra
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[track_a,track_b,dev]"
pytest tests/                              # 499 tests, 0 skipped
uvicorn nyaya_mitra.env.server:app         # the env over HTTP
```

The Colab training notebook lives at `training/train_grpo_colab.ipynb`.

## Why we built this for the hackathon

OpenEnv's #1 failure mode (per the official hackathon guide) is reward hacking. Most hackathon submissions wave at it; we made it the technical story. The judging criteria reward "environments that push the frontier of what we can train LLMs to do." A welfare-and-legal-aid advisor that *cannot lie about contacts* and *cannot rationalize from facts the citizen denied* is closer to a real-world deployment than a chess RL env, and it tests the exact LLM weakness the hackathon set out to address.
