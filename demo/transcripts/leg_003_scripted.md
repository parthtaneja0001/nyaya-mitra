# leg_003 — scripted baseline

**seed**: 203 · **difficulty**: _eval/legal_only · **finalized**: ✓ · **total reward**: 0.492 · **turns**: 5 · **sim_leak**: 0

## Conversation
> Citizen: The laptop I purchased is defective. The seller refuses replacement.
- **PROBE** (dv, en): Is anyone at home hurting you?
> Citizen: No, that's not the case for me.
- **ASK** (en): Do you have a small landholding or do you do farm work?
> Citizen: Yes, I am an adult.
- **ASK** (en): Do you have a small landholding or do you do farm work?
> Citizen: I'm willing to do unskilled manual work if there's pay.
- **ASK** (en): Do you have a small landholding or do you do farm work?
> Citizen: I'm in the 18-70 age range.
- **FINALIZE** — see plan below.

## Action plan
**Schemes**
- `mgnrega` — rationale: adult, willing_unskilled_work
- `pmsby` — rationale: adult_18_70

**Legal routes**
- `consumer_protection_act_2019` → forum: District Consumer Disputes Redressal Commission → NALSA `nalsa_central`

**Most important next step**: contact your district legal services authority for free legal aid

_Summary (en): we will help you apply for the schemes you qualify for and connect you to free legal aid._

## Reward breakdown
| component | value |
|---|---|
| `dignity_judge` | 1.000 |
| `document_accuracy` | 0.198 |
| `fact_coverage` | 1.000 |
| `gate_contradiction` | 0.000 |
| `gate_format_violation` | 0.000 |
| `gate_hallucination` | 0.000 |
| `gate_sim_leak` | 0.000 |
| `harm_penalty` | -0.100 |
| `integration_bonus` | 0.000 |
| `legal_precision` | 1.000 |
| `legal_recall` | 1.000 |
| `procedural_correctness` | 0.475 |
| `scheme_precision` | 0.000 |
| `scheme_recall` | 1.000 |
| `sensitivity_correctness` | 1.000 |
| `shaping_ask_fact` | 0.000 |
| `shaping_jargon` | 0.000 |
| `shaping_late_turn` | 0.000 |
| `shaping_probe_sensitive` | 0.000 |
| `total` | 0.492 |
| `turn_efficiency` | 0.167 |