# Automatic footprint and response v10

## Regression diagnosis

The first district compiler linearly accumulated office and constituency
history at province scope.

- In 2022, three Seongnam mayoral records and one Gyeonggi governor record were
  all treated as Gyeonggi-wide evidence. A lost 2006 mayoral race also received
  an office-base weight. This saturated Lee Jae-myung's affinity at 0.85 and
  moved the Gyeonggi prediction by about 1.57 percentage points.
- In 2012, three Park Geun-hye constituency records were summed into a 0.325
  Daegu-wide affinity, while one Moon Jae-in constituency became a 0.246
  Busan-wide affinity. The national diagnostic improved slightly, but the
  population-weighted regional MAE regressed.

## Footprint-controlled v9

The automatic compiler now applies these rules before province roll-up:

1. A constituency's scope is the square root of its valid-vote share within
   the province.
2. A municipal executive office uses the municipality's share of prior
   Assembly constituencies in the province.
3. A lost executive candidacy does not create an office base.
4. Repeated records use bounded union rather than linear summation.
5. A governor win remains province-wide evidence.

This changes the main affected affinities:

| Candidate-region | First district compiler | Footprint v9 |
|---|---:|---:|
| Park Geun-hye-Daegu | 0.325 | 0.081 |
| Moon Jae-in-Busan (2012) | 0.246 | 0.068 |
| Lee Jae-myung-Gyeonggi | 0.850 | 0.708 |
| Sim Sang-jung-Gyeonggi | 0.821 | 0.150 |

With the active 0.40 response, 2012 regional MAE is 2.1987 percentage points
and 2022 is 1.4434. The earlier 2012/2022 regressions from province-wide
overexpansion are therefore removed.

## Prior-only response v10

The fixed 0.60 diagnostic response is replaced by a strict prior-only selector.
For each target election, it:

- scores gains 0.40, 0.50, 0.60, and 0.70 on earlier completed election folds;
- never reads the target election outcome;
- shrinks the selected gain toward 0.50 according to the number of earlier
  activated contest regimes;
- applies a matching log-shift cap and swing cap;
- retains the coefficient-free cumulative-rejection router.

Selected gains:

| Target | Selected gain |
|---|---:|
| 2002 | 0.500 |
| 2007 | 0.500 |
| 2012 | 0.600 |
| 2017 | 0.633 |
| 2022 | 0.650 |

## Strict nested result

| Variant | Regional weighted MAE | National candidate MAE | Winner accuracy |
|---|---:|---:|---:|
| Active v16 | 3.3817 | 1.8417 | 0.80 |
| Automatic footprint + prior response | **3.3128** | **1.6185** | **0.80** |

Election-level regional MAE:

| Election | Active | Automatic candidate | Change |
|---|---:|---:|---:|
| 2002 | 3.7484 | 3.8354 | +0.0870 |
| 2007 | 4.9237 | 5.1585 | +0.2348 |
| 2012 | 2.1992 | 2.2201 | +0.0208 |
| 2017 | 4.5431 | 3.9067 | -0.6364 |
| 2022 | 1.4939 | 1.4434 | -0.0506 |

## Decision and remaining automation

Keep this as a candidate, not the active model. The 2007 regression is
concentrated and has a factual cause: pre-2007 official election history does
not encode Lee Hoi-chang's Chungcheong biographical affinity. Post-election
2008 constituency history cannot be used. A dated pre-election biographical
source is required before the manual Chungcheong alignment can be removed.

The next automatic replacements should be implemented in this order:

1. dated candidate biography and regional-affinity evidence;
2. third-candidate source-lane pressure;
3. KOSIS age population plus NEC age-turnout weights;
4. withdrawal compliance and target split from earlier comparable events;
5. transcript-derived mega-issue intensity and conservative taxonomy;
6. prior-only selection for the remaining behavioral gains.

Safety bounds and numerical caps should remain explicit rather than be fitted.

Artifacts:

- `outputs/footprint_candidate_base_v9/`
- `outputs/footprint_candidate_base_v9_ablation/`
- `outputs/automatic_contest_response_v10_ablation/`
- `outputs/automation_status_v3/`
