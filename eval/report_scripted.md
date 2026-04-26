# Eval report — scripted

## Headline

- Cases: 30
- Mean total reward: **0.469**
- Median: 0.449
- P25 / P75: 0.380 / 0.571
- All gates passed: **100.0%**
- Integrated cases solved: **0.0%**

## Per-cohort

| Cohort | n | mean reward | gates passed | finalized | mean turns | sensitivity F1 |
|---|---|---|---|---|---|---|
| welfare_only | 10 | 0.568 | 100.0% | 100.0% | 5.0 | 0.90 |
| legal_only | 10 | 0.416 | 100.0% | 100.0% | 5.0 | 0.80 |
| integrated | 10 | 0.422 | 100.0% | 100.0% | 5.0 | 0.15 |

## Reward components (means)

| Cohort | scheme P | scheme R | legal P | legal R | turn eff |
|---|---|---|---|---|---|
| welfare_only | 0.81 | 0.38 | 1.00 | 1.00 | 0.23 |
| legal_only | 0.00 | 1.00 | 1.00 | 0.40 | 0.45 |
| integrated | 1.00 | 0.20 | 1.00 | 0.45 | 0.00 |

## Gate triggers (count)

| Cohort | format | hallucination | contradiction | sim leak |
|---|---|---|---|---|
| welfare_only | 0 | 0 | 0 | 0 |
| legal_only | 0 | 0 | 0 | 0 |
| integrated | 0 | 0 | 0 | 0 |

## Episode summary

### welfare_only

| seed | reward | finalized | turns | sim_leak |
|---|---|---|---|---|
| 101 | 0.577 | ✓ | 5 | 0 |
| 102 | 0.579 | ✓ | 5 | 0 |
| 103 | 0.370 | ✓ | 5 | 0 |
| 104 | 0.504 | ✓ | 5 | 0 |
| 105 | 0.626 | ✓ | 5 | 0 |
| 106 | 0.583 | ✓ | 5 | 0 |
| 107 | 0.699 | ✓ | 5 | 0 |
| 108 | 0.573 | ✓ | 5 | 0 |
| 109 | 0.571 | ✓ | 5 | 0 |
| 110 | 0.603 | ✓ | 5 | 0 |

### legal_only

| seed | reward | finalized | turns | sim_leak |
|---|---|---|---|---|
| 201 | 0.460 | ✓ | 5 | 0 |
| 202 | 0.403 | ✓ | 5 | 0 |
| 203 | 0.510 | ✓ | 5 | 0 |
| 204 | 0.412 | ✓ | 5 | 0 |
| 205 | 0.400 | ✓ | 5 | 0 |
| 206 | 0.323 | ✓ | 5 | 0 |
| 207 | 0.403 | ✓ | 5 | 0 |
| 208 | 0.323 | ✓ | 5 | 0 |
| 209 | 0.370 | ✓ | 5 | 0 |
| 210 | 0.557 | ✓ | 5 | 0 |

### integrated

| seed | reward | finalized | turns | sim_leak |
|---|---|---|---|---|
| 301 | 0.464 | ✓ | 5 | 0 |
| 302 | 0.429 | ✓ | 5 | 0 |
| 303 | 0.570 | ✓ | 5 | 0 |
| 304 | 0.370 | ✓ | 5 | 0 |
| 305 | 0.439 | ✓ | 5 | 0 |
| 306 | 0.379 | ✓ | 5 | 0 |
| 307 | 0.476 | ✓ | 5 | 0 |
| 308 | 0.383 | ✓ | 5 | 0 |
| 309 | 0.370 | ✓ | 5 | 0 |
| 310 | 0.343 | ✓ | 5 | 0 |
