# Reward design

Source of truth for weights, gate semantics, and the public api the env consumes.

## Public api

`nyaya_mitra.rewards` exposes exactly four things:

- `compute_reward(ctx: RewardContext) -> RewardBreakdown`
- `make_env_reward_fn(kb, *, extra_info=None, max_turns=20) -> Callable`
- `compute_shaping(turn_index, action, revealed_this_turn, sim_leak, citizen_literacy) -> dict[str, float]`
- `KnowledgeBase` (Protocol)

Wiring happens via `scripts/wire_rewards.py:build_env(...)` which imports `nyaya_mitra.rewards.kb_adapter.DuckTypedKB`, wraps the `KnowledgeBase` instance, and passes the result into `make_env_reward_fn(kb)`. The returned callable matches the env's `RewardFn` signature `(profile, plan, transcript, elicited_facts) -> dict[str, float]`.

## Weights (PLAN B.2 #3 corrected decomposition)

Soft components — sum to 1.0:

| Component | Weight | Type |
|---|---|---|
| `scheme_precision` | 0.10 | deterministic |
| `scheme_recall` | 0.10 | deterministic |
| `legal_precision` | 0.10 | deterministic |
| `legal_recall` | 0.10 | deterministic |
| `document_accuracy` | 0.10 | deterministic |
| `procedural_correctness` | 0.10 | deterministic |
| `fact_coverage` | 0.12 | deterministic |
| `integration_bonus` | 0.15 | deterministic, scoped |
| `sensitivity_correctness` | 0.05 | deterministic |
| `turn_efficiency` | 0.03 | deterministic |
| `dignity_judge` | 0.05 | LLM-judged, capped |

Caps: deterministic ≤ 0.15, LLM-judge ≤ 0.05. `validate_weights()` runs at import.

`harm_penalty` is a separate additive (-0.05 per harmful suggestion, capped at -0.20). It is NOT in the soft budget and is bypassed when a hard gate fires (gate-fail short-circuits to -1.0).

## Per-turn shaping (PLAN B.2 #4)

Shaping is computed by the env each step (via `compute_shaping(...)`) and accumulated into a running dict that the env passes to the aggregator at terminal step under `info["shaping_running"]`. The aggregator caps positive shaping at +0.4 per episode (negatives uncapped) and adds it to the total.

| Shaping key | Trigger | Value |
|---|---|---|
| `shaping_ask_fact` | `Ask` causes a new fact to enter `elicited_facts` | +0.02 |
| `shaping_probe_sensitive` | `Probe` topic matches a revealed sensitive fact (no sim_leak) | +0.05 |
| `shaping_late_turn` | turn index ≥ 15 | -0.03 |
| `shaping_jargon` | `Explain` for low-literacy citizen contains legal jargon | -0.10 |

## Hard gates (short-circuit total to -1.0)

1. `gate_format_violation` — empty plan, blank `most_important_next_step`, blank summary, or env-set `info["format_violation"]=true`.
2. `gate_hallucination` — any unknown `scheme_id`, `framework_id`, or `(authority, contact_id)` in the plan.
3. `gate_contradiction` — any `rationale_facts` entry not in `elicited_facts`, or any entry that appears in a citizen turn's `info["negated_facts"]`.

`gate_sim_leak` is **not** a hard gate. It is a passthrough that records the count of leaked turns (where `info["sim_leak"]=true`) and zeroes the elicitation shaping (`shaping_ask_fact`, `shaping_probe_sensitive`) when any leak occurred. Negative shaping is preserved.

## Gate priority

Format → hallucination → contradiction. Sim-leak passthrough applies regardless of gate state (it adjusts shaping in the breakdown for monitoring even when the total is forced to -1.0). The breakdown still emits all component scores when a gate fires — this aids debugging.

## Reward overlap fix

The original spec had `wrong_suggestion_penalty` as a separate soft component, double-counting precision. Removed. Replaced by `harm_penalty` (additive, conservative — only fires for valid-but-wrong suggestions, never for hallucinated ids since those gate to -1).

## Anti-reward-hacking properties (test invariants)

The following properties are tested and must not regress:

- "suggest everything" loses on precision and accumulates harm_penalty.
- "suggest nothing" loses on recall.
- Hallucinated ids hard-fail to -1.0.
- Skipping elicitation tanks fact_coverage and zeroes integration_bonus.
- Dignity judge alone cannot lift the total above ~0.05 (test: `test_no_dignity_dominance_with_max_judge`).
- Per-turn shaping cannot exceed +0.4 over an episode (test: `test_positive_cap_scales_proportionally_when_exceeded`).
- Sim-leak credit is removed without ending the episode (test: `test_sim_leak_zeroes_elicitation_shaping`).

## What the env passes through

The env surfaces these per-turn signals; the aggregator reads them at terminal step:

- `info["sim_leak"]: bool` — set by env's `_detect_sim_leak`.
- `info["negated_facts"]: list[str]` — set by the extractor's negation pass. Contradiction gate consumes via `Turn.info["negated_facts"]`.
- `info["format_violation"]: bool` — for actions rejected at schema level (defaults False).
- `info["shaping_running"]: dict[str, float]` — accumulated shaping deltas keyed by `SHAPING_KEYS`.

All of these are optional with safe defaults. The reward function tolerates a missing field by defaulting to a safe value (shaping defaults to zeros, contradiction falls back to "must appear in elicited_facts").

## Relevant-facts contract (extractor → reward)

`fact_coverage` and the integration bonus rely on knowing which fact IDs each scheme/framework's checker reads from the profile. **Canonical source: `src/nyaya_mitra/profile/relevant_facts.py:_RELEVANT_BY_ID`.** The mirror in `src/nyaya_mitra/rewards/kb_adapter.py:_DEFAULT_RELEVANT_FACTS` must agree exactly with it.

Two regression tests guard this:

- `test_every_scheme_has_relevant_facts_entry` / `test_every_framework_has_relevant_facts_entry` — KB growth without map growth fails CI.
- `test_kb_adapter_agrees_with_track_a_relevant_facts` — drift between the two maps fails CI.

The long-term plan is to move the mapping into KB JSON so neither side owns a separate dict; tracked as a separate `[interface]` task on the board.

Current contract — fact IDs the extractor emits when the citizen reveals these realities:

| Source | Fact IDs |
|---|---|
| `pm_kisan` | `occupation_farmer`, `land_small` |
| `pmuy` | `gender_female`, `bpl_household`, `no_lpg` |
| `ayushman_bharat` | `secc_listed`, `urban_occupational_category` |
| `mgnrega` | `adult`, `residence_rural`, `willing_unskilled_work` |
| `pm_awas_grameen` | `residence_rural`, `secc_listed`, `kuccha_or_houseless` |
| `pmsby` | `adult_18_70`, `has_bank_account` |
| `domestic_violence_act_2005` | `gender_female`, `dv_present` |
| `consumer_protection_act_2019` | `is_consumer`, `consumer_grievance` |
| `maternity_benefit_act_1961` | `gender_female`, `formally_employed`, `pregnant_or_postpartum`, `denied_maternity_benefit` |
| `minimum_wages_act_1948` | `is_wage_worker`, `wages_below_minimum` |

New IDs require coordinated update of (1) the eligibility/applicability checker, (2) `profile/relevant_facts.py`, (3) `rewards/kb_adapter.py`, and (4) the extractor.

## Testing

99 reward tests in `tests/track_b/`:

- `test_components.py` — 56 golden checks (≥5 per component).
- `test_gates.py` — 17 gate triggers including the "sneaky" negated-fact case.
- `test_shaping.py` — 12 shaping-rule + cap checks.
- `test_aggregator.py` — 14 end-to-end checks: gate dominance, weight invariants, sim-leak passthrough, env-fn signature.

Plus 6 integration contract tests (1 newly wired: `test_aggregator_emits_all_keys`).
