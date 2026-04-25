# wel_005 — scripted baseline

**seed**: 105 · **difficulty**: _eval/welfare_only · **finalized**: ✓ · **total reward**: 0.664 · **turns**: 5 · **sim_leak**: 0

## Conversation
> Citizen: मैं किसान हूं, BPL परिवार है, गैस नहीं है।
- **PROBE** (dv, hi): क्या घर में आपको कोई परेशानी या मार-पीट है?
> Citizen: नहीं, ऐसा कुछ नहीं है।
- **ASK** (hi): क्या आपकी छोटी जमीन है या खेती का काम है?
> Citizen: मैं महिला हूं।
- **ASK** (hi): क्या आपकी छोटी जमीन है या खेती का काम है?
> Citizen: मैं गाँव में रहती हूं।
- **ASK** (hi): क्या आपकी छोटी जमीन है या खेती का काम है?
> Citizen: मेरे पास छोटा खेत है, बहुत कम जमीन है।
- **FINALIZE** — see plan below.

## Action plan
**Schemes**
- `pm_kisan` — rationale: land_small, occupation_farmer
- `pmuy` — rationale: bpl_household, no_lpg
_(no legal routes)_

**Most important next step**: contact your district legal services authority for free legal aid

_Summary (en): we will help you apply for the schemes you qualify for and connect you to free legal aid._

## Reward breakdown
| component | value |
|---|---|
| `dignity_judge` | 0.750 |
| `document_accuracy` | 0.500 |
| `fact_coverage` | 0.600 |
| `gate_contradiction` | 0.000 |
| `gate_format_violation` | 0.000 |
| `gate_hallucination` | 0.000 |
| `gate_sim_leak` | 0.000 |
| `harm_penalty` | -0.000 |
| `integration_bonus` | 0.000 |
| `legal_precision` | 1.000 |
| `legal_recall` | 1.000 |
| `procedural_correctness` | 1.000 |
| `scheme_precision` | 1.000 |
| `scheme_recall` | 0.500 |
| `sensitivity_correctness` | 1.000 |
| `shaping_ask_fact` | 0.000 |
| `shaping_jargon` | 0.000 |
| `shaping_late_turn` | 0.000 |
| `shaping_probe_sensitive` | 0.000 |
| `total` | 0.664 |
| `turn_efficiency` | 0.167 |