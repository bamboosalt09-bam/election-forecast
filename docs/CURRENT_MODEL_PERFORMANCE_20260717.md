# Current Model Performance (2026-07-17)

> **Critical validity warning (2026-07-18):** `slot_A` is the realized winner
> in all five scored elections, while `slot_A`, `slot_B`, `slotA_prior`, and
> `slotB_prior` are active predictors without a frozen pre-election slot
> assignment protocol. The nested `4.613162%p` regional MAE and `3.441419%p`
> national point MAE are reproducible historical diagnostics, but they must not
> be cited as outcome-blind forecast performance. See
> `docs/SLOT_PREDICTOR_LEAKAGE_AUDIT_20260718.md`.

## Boundary

- active workspace: `C:\english_folder\poll_project`
- scored elections: 2002, 2007, 2012, 2017, 2022
- rolling warmup: 1997 only
- 2025 outcome: excluded
- primary regional metric: contest-vote weighted candidate-region MAE within election,
  followed by an equal-election macro average
- active nested-selected electorate gains: preference `0.04`; terrain, turnout, nonvoter `0`
- historical fixed electorate experiment: preference gain `0.04`, evaluated post hoc

Observed contest votes are used only as post-election evaluation weights. The national
aggregation is also a post-election diagnostic, not a deployable pre-election metric.

## Electorate-layer audit

The old signal normalization forced every election's strongest candidate contrast to one.
The corrected version preserves candidate-tone magnitude and confidence and estimates core
support from direct party ballots separately from candidate ballots.

The active `direct_party_layers` profile now keeps `2,602` direct-party rows separate:
Assembly proportional `1,933`, metropolitan-council proportional `413`, and local-council
proportional `256`. Candidate-ballot excess is treated as personal/swing evidence instead of
raising party attachment. Raw layer reassignment is capped at `2.5%p`; observed final maxima
are `2.70%p` for core and `2.50%p` for critical support, both within the `3%p` policy cap.

| Frozen outer evaluation | Weighted macro MAE |
|---|---:|
| Pre-layer baseline | 4.627768%p |
| Capped strict nested learner | 4.613162%p |
| Uncapped nested gain learner (rejected) | 5.470805%p |
| Non-active fixed structural experiment | 4.593551%p |

The capped learner uses gain `0` for 2002 and 2007 because fewer than two earlier scored
elections are available, then selects `0.04` for 2012, 2017, and 2022. It improves the strict
nested macro by `0.014606%p` and passes every adoption gate. The fixed experiment applies
`0.04` post hoc even to 2002 and 2007, so its lower `4.593551%p` value is not nested evidence.
The active v14 explanatory issue overlay is a separate bounded theory-driven override; nested
selection itself still chose overlay gain `0`.

An additional uncapped, one-parameter nested learner was tested. It expands the gain search
until a constrained interior optimum is found and excludes every target election from its
own tuning set. Its strict outer-fold macro MAE was `5.470805%p`, an deterioration of
`0.843037%p`; consequently it was not adopted. Its artifacts are stored separately under
`outputs/electorate_nested_learning_uncapped`.

| Election | Baseline | Capped strict nested | Improvement |
|---|---:|---:|---:|
| 2002 | 3.5712 | 3.5712 | +0.0000 |
| 2007 | 6.3463 | 6.3463 | +0.0000 |
| 2012 | 6.2182 | 6.1936 | +0.0246 |
| 2017 | 5.3232 | 5.2624 | +0.0608 |
| 2022 | 1.6800 | 1.6923 | -0.0123 |

## Active rolling performance

| Election | Row MAE | Contest-vote weighted MAE |
|---|---:|---:|
| 2002 | 4.0323 | 3.5422 |
| 2007 | 4.2877 | 4.6079 |
| 2012 | 4.0415 | 4.6000 |
| 2017 | 3.7920 | 4.1480 |
| 2022 | 2.1299 | 1.6924 |
| Overall | **3.7089** | **3.7181** equal-election macro |

The active gain is `0.04`, selected by the capped strict nested learner. Compared
with the pre-correction historical snapshot (`3.627322%p`), current overall rolling
performance is worse because the former 2012 gain depended heavily on the invalid maximum
normalization.

Other current diagnostics:

- LOEO row MAE: `4.98%p`
- Ridge R2: `0.876`
- Ridge alpha: `1.20`
- tests: `308 passed`
- through-2022 selection-boundary audit: PASS
- strict deep PIT audit: PASS, outcome invariance `215/215`

## National rolling diagnostic

| Election | Candidate | Predicted | Actual | Absolute error |
|---|---|---:|---:|---:|
| 2002 | Roh Moo-hyun | 49.464 | 51.217 | 1.753 |
| 2002 | Lee Hoi-chang | 50.536 | 48.783 | 1.753 |
| 2007 | Lee Myung-bak | 54.952 | 54.140 | 0.812 |
| 2007 | Chung Dong-young | 32.887 | 29.089 | 3.799 |
| 2007 | Lee Hoi-chang | 12.160 | 16.771 | 4.611 |
| 2012 | Park Geun-hye | 56.044 | 51.773 | 4.271 |
| 2012 | Moon Jae-in | 43.956 | 48.227 | 4.271 |
| 2017 | Moon Jae-in | 45.035 | 47.476 | 2.441 |
| 2017 | Hong Joon-pyo | 32.476 | 27.773 | 4.703 |
| 2017 | Ahn Cheol-soo | 22.489 | 24.751 | 2.262 |
| 2022 | Yoon Suk-yeol | 50.516 | 50.380 | 0.136 |
| 2022 | Lee Jae-myung | 49.484 | 49.620 | 0.136 |

Candidate-row national MAE is `2.578921%p`; equal-election national macro MAE is
`2.473813%p`.

## 2022 diagnosis

The old layer moved candidate A by about `+0.825%p` nationally and raised weighted regional
MAE from `1.6800` to `1.7924%p`. The capped strict nested model uses gain `0.04`, with national
candidate error `0.138%p` and weighted regional MAE `1.6922%p`.

In the fixed experiment the national direction is correct. Its residual problem is regional uniformity: the shift
helps Seoul, Busan, Daegu, Daejeon, Chungcheong, Gangwon, and Gyeongbuk, but harms Ulsan,
Gyeongnam, Incheon, Gyeonggi, Jeju, and Honam where the baseline error had a different local
direction. Region-level details are in
`outputs/electorate_layer_experiment/pres_2022_region_diagnostics.csv`.

## Artifacts

- `outputs/electorate_layer_experiment/summary.json`
- `data/config/electorate_layers_fixed_experiment.json`
- `outputs/electorate_layer_experiment/fixed_structural_predictions.csv`
- `outputs/electorate_layer_experiment/pres_2022_region_diagnostics.csv`
- `outputs/electorate_layer_experiment/history_source_audit.csv`
- `outputs/electorate_nested_learning/summary.json`
- `outputs/electorate_nested_learning/nested_comparison.csv`
- `outputs/electorate_nested_learning_uncapped/summary.json`
- `presidential_issue_engine/report/tables/issue_vote_engine_rolling_predictions.csv`
- `presidential_issue_engine/report/tables/issue_vote_engine_rolling_national_summary.csv`
