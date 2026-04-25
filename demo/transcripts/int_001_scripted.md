# int_001 — scripted baseline

**seed**: 301 · **difficulty**: _eval/integrated · **finalized**: ✓ · **total reward**: 0.492 · **turns**: 5 · **sim_leak**: 0

## Conversation
> Citizen: Mai kheti karti hoon, BPL hain, ghar mein bhi pareshani hai.
- **PROBE** (dv, hinglish): Is anyone at home hurting you?
> Citizen: Abhi yeh baat nahi karna chahti.
- **ASK** (hinglish): Do you have a small landholding or do you do farm work?
> Citizen: Mai mahila hoon.
- **ASK** (hinglish): Do you have a small landholding or do you do farm work?
> Citizen: Gaon mein rehte hain hum.
- **ASK** (hinglish): Do you have a small landholding or do you do farm work?
> Citizen: Gas connection nahi hai, chulha pe khana banate hain.
- **FINALIZE** — see plan below.

## Action plan
**Schemes**
- `pmuy` — rationale: bpl_household, no_lpg
- `pm_kisan` — rationale: occupation_farmer
_(no legal routes)_

**Most important next step**: contact your district legal services authority for free legal aid

_Summary (en): we will help you apply for the schemes you qualify for and connect you to free legal aid._

## Reward breakdown
| component | value |
|---|---|
| `dignity_judge` | 0.750 |
| `document_accuracy` | 0.500 |
| `fact_coverage` | 0.455 |
| `gate_contradiction` | 0.000 |
| `gate_format_violation` | 0.000 |
| `gate_hallucination` | 0.000 |
| `gate_sim_leak` | 0.000 |
| `harm_penalty` | -0.000 |
| `integration_bonus` | 0.000 |
| `legal_precision` | 1.000 |
| `legal_recall` | 0.000 |
| `procedural_correctness` | 1.000 |
| `scheme_precision` | 1.000 |
| `scheme_recall` | 0.500 |
| `sensitivity_correctness` | 0.000 |
| `shaping_ask_fact` | 0.000 |
| `shaping_jargon` | 0.000 |
| `shaping_late_turn` | 0.000 |
| `shaping_probe_sensitive` | 0.000 |
| `total` | 0.492 |
| `turn_efficiency` | 0.000 |