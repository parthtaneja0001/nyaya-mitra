# Architecture

Nyaya Mitra is an OpenEnv-compliant environment plus a reward function plus a training pipeline. Three layers:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HTTP layer (canonical OpenEnv)                                          │
│    openenv.core.env_server.http_server.create_app(...)                   │
│    /reset · /step · /state · /metadata · /schema · /docs · /healthz     │
│    /mcp · /ws (canonical OpenEnv multi-transport)                        │
├──────────────────────────────────────────────────────────────────────────┤
│  OpenEnv-conformant env  src/nyaya_mitra/env/openenv_env.py             │
│    NyayaEnvironment(Environment[NyayaAction, NyayaObservation, NyayaState])
│      ├── rubric: Sequential(Gate, Gate, Gate, WeightedSum(11 components))│
│      ├── reset(seed, episode_id) → NyayaObservation                      │
│      ├── step(NyayaAction) → NyayaObservation (reward + done on it)      │
│      └── state → NyayaState  (public; no profile/ground-truth leak)      │
├──────────────────────────────────────────────────────────────────────────┤
│  Domain layer  src/nyaya_mitra/                                          │
│    citizen/   ── extractor + smart-canned simulator + 72 golden tests   │
│    knowledge/ ── 6 schemes + 4 frameworks + 20 DLSAs across 10 states   │
│    profile/   ── 14 seed profiles, derive_ground_truth from KB checkers │
│    rewards/   ── 11 components + 4 gates + per-turn shaping             │
│                  + openenv_rubric.py (Sequential/Gate/WeightedSum tree)  │
│    case_gen/  ── adversarial profile generator (Phase 2)                 │
└──────────────────────────────────────────────────────────────────────────┘
```

## OpenEnv conformance

- `NyayaEnvironment` subclasses `openenv.core.env_server.interfaces.Environment[ActT, ObsT, StateT]`.
- `NyayaAction` subclasses `openenv.core.env_server.types.Action`.
- `NyayaObservation` subclasses `openenv.core.env_server.types.Observation` — `done` and `reward` are fields on the observation, per the OpenEnv contract.
- `NyayaState` subclasses `openenv.core.env_server.types.State` (carries `episode_id` + `step_count`).
- HTTP server is `openenv.core.create_app(env, action_cls, observation_cls)` — not a hand-rolled FastAPI.
- `openenv.yaml` matches the canonical reference (`spec_version: 1`, `type: space`, `runtime: fastapi`).
- `tests/integration/test_openenv_conformance.py` mechanically verifies inheritance + rubric structure + route exposure + manifest format (10 tests).

## Rubric system (composable, OpenEnv-native)

Per the OpenEnv RFC 004 design, `env.rubric` is a tree of `Rubric` nodes:

```python
Sequential(                                # fail-fast on gates
    Gate(FormatRubric()),                  # → 0 if plan malformed
    Gate(HallucinationRubric()),           # → 0 if unknown scheme/framework/contact id
    Gate(ContradictionRubric()),           # → 0 if rationale_facts contradict citizen
    WeightedSum(                           # → weighted sum of 11 components
        rubrics=[SchemePrecision, SchemeRecall, LegalPrecision, LegalRecall,
                 DocumentAccuracy, ProceduralCorrectness, FactCoverage,
                 IntegrationBonus, SensitivityCorrectness, TurnEfficiency,
                 DignityJudge],
        weights=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.15, 0.05, 0.03, 0.05],
    ),
)
```

Training infra introspects scores via `for name, r in env.rubric.named_rubrics(): r.last_score`.

## Per-turn info passthrough (env → reward)

The internal env writes the following into `info` on each step; the aggregator reads them at terminal step:

| key | when | purpose |
|---|---|---|
| `info["sim_leak"]` (per-step bool) | always on `step()` | sim_leak gate (passthrough; zeros elicitation shaping) |
| `info["negated_facts"]` (per-step list) | always | contradiction gate consumes via `Turn.info["negated_facts"]` |
| `info["format_violation"]` (terminal bool) | terminal | format gate hint |
| `info["shaping_running"]` (terminal dict) | terminal | accumulated per-turn shaping deltas |
| `info["reward_breakdown"]` (terminal dict) | terminal | the 19-key breakdown |

All optional with safe defaults; missing fields default to 0.

## CI

Single workflow `.github/workflows/ci.yml`: installs `[env,rewards,dev]`, runs `ruff check .`, runs `pytest tests`. Green on every push to main + every PR.

## Anti-reward-hacking invariants

| Invariant | Where it's enforced |
|---|---|
| Every legal route includes free legal aid contact | `interface/plan.py` Pydantic schema |
| Soft reward components sum to 1.0 | `rewards/weights.py:validate_weights()` (runs at import) |
| No LLM-judge component > 5% | same |
| No deterministic component > 15% | same |
| Hallucinated id → `total = -1.0` | `rewards/gates/hallucination.py` |
| Citizen-contradicted rationale_facts → `total = -1.0` | `rewards/gates/contradiction.py` |
| sim_leak (sensitive disclosed without matching Probe) | env's `_detect_sim_leak` + `rewards/gates/sim_leak_passthrough.py` |
| Per-turn shaping capped at +0.4/episode | `rewards/shaping.py:cap_positive_shaping` |
| Same seed → same transcript | `tests/track_a/test_determinism.py` (sha256 hash compare) |
| OpenEnv conformance | `tests/integration/test_openenv_conformance.py` |

See [`reward_design.md`](reward_design.md) for the full reward decomposition.
