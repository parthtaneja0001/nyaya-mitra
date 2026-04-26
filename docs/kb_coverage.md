# KB coverage

KB inventory. Refresh on every KB change.

## Schemes (6 / 30 target)

| scheme_id | category | ministry | verified_on | tests | notes |
|---|---|---|---|---|---|
| `pm_kisan` | agriculture_rural | Agriculture & Farmers Welfare | 2026-04-25 | 6 | farmer with cultivable land, professional / income-tax exclusions |
| `pmuy` | women_child | Petroleum & Natural Gas | 2026-04-25 | 6 | adult woman, BPL, no existing LPG |
| `mgnrega` | labor_livelihood | Rural Development | 2026-04-25 | 5 | adult rural resident willing to do unskilled work |
| `pm_awas_grameen` | agriculture_rural | Rural Development | 2026-04-25 | 6 | rural, SECC-listed, kuccha or houseless, no pucca house |
| `ayushman_bharat` | health | Health and Family Welfare | 2026-04-25 | 5 | SECC household or urban occupational category |
| `pmsby` | labor_livelihood | Finance | 2026-04-25 | 6 | age 18-70, savings bank account |

## Frameworks (4 / 15 target)

| framework_id | category | forum | verified_on | tests | notes |
|---|---|---|---|---|---|
| `domestic_violence_act_2005` | women_protection | Magistrate (First Class) | 2026-04-25 | 4 | woman with domestic violence present or in history |
| `minimum_wages_act_1948` | labor | s. 20 Authority (Labour Commissioner) | 2026-04-25 | 4 | wage worker paid below state-notified minimum |
| `maternity_benefit_act_1961` | women_protection | Inspector / Labour Commissioner | 2026-04-25 | 6 | pregnant or postpartum woman employee denied benefit |
| `consumer_protection_act_2019` | consumer | District/State/National Consumer Commission | 2026-04-25 | 6 | consumer with defective goods, deficient service, unfair practice, or misleading ad |

## DLSA directory

Minimal v1: NALSA central + Punjab SLSA + DLSA Ludhiana.

## Outstanding

- 24 schemes to add (target 30)
- 11 frameworks to add (target 15)
- Bring per-checker test count up to >=8 each
- Expand DLSA directory with at least one DLSA per major state
- Add fact-extractor patterns for the new fields (rural, secc, wage_worker, pregnant, defective_goods, etc.)
