# Architecture

The seam between Track A (Data + World) and Track B (Agent + Learning). Jointly owned `[S]`; edits go through a cross-track-approved PR.

## The three layers

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Track A — World                              Track B — Agent        │
│  src/nyaya_mitra/                             src/nyaya_mitra/       │
│  ├─ env/         (FastAPI + reset/step)       ├─ rewards/            │
│  ├─ citizen/     (sim + extractor)            │  ├─ components/      │
│  ├─ knowledge/   (KB + checkers)              │  ├─ gates/           │
│  └─ profile/     (seeds + ground-truth)       │  ├─ aggregator       │
│                                               │  ├─ shaping          │
│                                               │  └─ kb_protocol      │
│                                               ├─ case_gen/           │
│                                               └─ advisor/            │
│                                               training/, eval/, demo/│
│                                                                      │
│  ─────────────────  src/nyaya_mitra/interface/  ─────────────────    │
│                          (shared schemas)                            │
│  AdvisorAction │ ActionPlan │ CitizenObservation │ CitizenProfile    │
│  reward_keys   │ kb_schemas                                          │
│                                                                      │
│  ─────────────────  scripts/wire_rewards.py  ─────────────────       │
│                  (the only cross-track import)                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## What crosses the seam

The interface package is purely Pydantic + literal types — no logic. Both tracks import freely from `nyaya_mitra.interface`. Neither track imports from the other's `src/` packages; this is **mechanically enforced** by `tests/integration/test_no_cross_track_imports` (an AST walk over `src/nyaya_mitra/`).

The single legitimate cross-track point is `scripts/wire_rewards.py:build_env(...)` — a helper that constructs a `NyayaMitraEnv` with both Track B's `compute_reward` and `compute_shaping` injected as constructor args. This is the import-allowed entry point Track B's training loop uses.

The contract:
- Track A's env exposes `reward_fn: RewardFn | None` and `shaping_fn: ShapingFn | None` constructor args. Both default to `None` (env runs cleanly without rewards).
- Track B publishes `make_env_reward_fn(kb)` and `compute_shaping(...)` matching the env's expected callable shapes.
- Track A's KB satisfies the read-only `nyaya_mitra.rewards.kb_protocol.KnowledgeBase` Protocol via the `DuckTypedKB` adapter.

## Per-turn info passthrough (env → reward)

The env writes the following into `info` on each step; the aggregator reads them at terminal step:

| key | when | purpose |
|---|---|---|
| `info["sim_leak"]` (per-step bool) | always on `step()` | sim_leak gate (passthrough; zeros elicitation shaping) |
| `info["negated_facts"]` (per-step list) | always | contradiction gate consumes via `Turn.info["negated_facts"]` |
| `info["format_violation"]` (terminal bool) | terminal | format gate hint |
| `info["shaping_running"]` (terminal dict) | terminal | accumulated per-turn shaping deltas |
| `info["reward_breakdown"]` (terminal dict) | terminal, populated by env's call to `reward_fn` | the final 19-key breakdown |

All optional with safe defaults; the reward fn tolerates a missing field (returns 0.0 for the corresponding key) so old transcripts don't break.

## File-ownership tags

Every file in the repo is tagged `[A]`, `[B]`, or `[S]` in `REPO_STRUCTURE.md`. CODEOWNERS enforces that `[A]` files require @parthtaneja0001 review, `[B]` files require @aPassie review, and `[S]` files require both.

## CI structure

Three workflows so a flaky test on one side never blocks the other:
- `ci-track-a.yml` — runs `pytest tests/track_a/` on track-A file changes
- `ci-track-b.yml` — runs `pytest tests/track_b/` on track-B file changes
- `ci-integration.yml` — runs `pytest tests/integration/` plus `ruff check` on every PR to `main`

The integration workflow is the gate: if `test_no_cross_track_imports` or any contract test fails, the seam has drifted and the merge is blocked.

## Anti-reward-hacking invariants (where they live)

| Invariant | Where it's enforced |
|---|---|
| Every legal route includes free legal aid contact | `interface/plan.py` Pydantic schema |
| Soft reward components sum to 1.0 | `rewards/weights.py:validate_weights()` (runs at import) |
| No LLM-judge component > 5% | same |
| No deterministic component > 15% | same |
| Hallucinated scheme/framework/contact_id → `total = -1.0` | `rewards/gates/hallucination.py` |
| Citizen-contradicted rationale_facts → `total = -1.0` | `rewards/gates/contradiction.py` |
| sim_leak (sensitive disclosed without matching Probe) | env's `_detect_sim_leak` + `rewards/gates/sim_leak_passthrough.py` |
| Per-turn shaping capped at +0.4/episode | `rewards/shaping.py:cap_positive_shaping` + `rewards/aggregator.py` |
| Cross-track imports forbidden | `tests/integration/test_no_cross_track_imports` (AST walk) |
| Same seed → same transcript | `tests/track_a/test_determinism.py` (sha256 hash compare) |

See [`docs/reward_design.md`](reward_design.md) for the full reward decomposition + the test invariants Track B maintains.

## When to change the interface

If you find yourself wanting to add a field to `interface/plan.py` or `interface/observations.py`:
1. Open an `[interface]` PR — both tracks must approve.
2. Update `tests/integration/test_interface_contract.py` in the same commit.
3. Coordinate the producer side (Track A: env emits) and consumer side (Track B: reward reads) updates to land before / together with the interface change, never after.

Most "I want to add a field" instincts should land on one side of the seam, not in the interface itself.
