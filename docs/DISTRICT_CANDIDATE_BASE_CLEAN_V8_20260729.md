# Clean district candidate-base v8 audit

## Why v6 was not an isolated comparison

The first district-first evaluator also enabled role-aware slot assignment and
the speech-derived v4 candidate context. Its reported 3.5014/2.0565 percentage
point metrics remain reproducible for that full variant, but they do not
measure only the regional-base replacement.

The clean v8 evaluator holds the active v16 pipeline fixed:

- default preliminary slot assignment (`role_aware=False`);
- active `data/raw` candidate, issue, and treatment inputs;
- active strict nested fitting and postprocessing;
- no post-2022 outcome;
- only `candidate_regional_base.csv` is replaced.

The input-manifest comparison found one changed input hash, exactly the
candidate regional-base file. There were zero unexpected differences.

## Results

| Variant | Regional weighted MAE | National candidate MAE | Winner accuracy |
|---|---:|---:|---:|
| Active v16 | 3.3817 | 1.8417 | 0.80 |
| District base, active response 0.40 | 3.4429 | 1.9308 | 0.80 |
| District base, balanced response 0.60 | 3.2874 | 1.6505 | 0.80 |
| District base, response 0.60 plus rejection routing | **3.2501** | **1.5671** | **0.80** |

Election-level comparison for the strongest diagnostic candidate:

| Election | Active regional MAE | Candidate regional MAE | Active national MAE | Candidate national MAE |
|---|---:|---:|---:|---:|
| 2002 | 3.7484 | 3.7955 | 3.1286 | 3.1294 |
| 2007 | 4.9237 | 4.7233 | 2.5487 | 1.6270 |
| 2012 | 2.1992 | 2.2226 | 0.2564 | 0.1980 |
| 2017 | 4.5431 | 3.9553 | 3.2492 | 2.5185 |
| 2022 | 1.4939 | 1.5541 | 0.0254 | 0.3625 |

The response increase is not an election ID exception. It applies only when
the outcome-blind contest-regime gate activates. In this sample, the effective
change is concentrated in the 2007 cumulative-rejection and 2017 rupture
regimes. Rejection-beneficiary routing has no new fitted coefficient.

## Remaining 2017 residual

The diagnostic candidate predicts:

| Candidate | Predicted | Actual | Error |
|---|---:|---:|---:|
| Moon Jae-in | 45.45 | 47.48 | -2.03 |
| Hong Joon-pyo | 31.55 | 27.77 | +3.78 |
| Ahn Cheol-soo | 23.00 | 24.75 | -1.75 |

The remaining structure is clear: rejected conservative-camp flexible mass is
still too high, while both alternatives are low. The current rejection router
sends that mass only to the dominant major candidate. A future generalized
router should distribute it among evidence-qualified alternatives using
strictly pre-election conversion capacity, political-lane affinity, and third
candidate viability.

## Decision

Do not promote automatically. The 0.60 response was selected after examining
the same five scored elections, and the personal-constituency signal still
needs an electorate-footprint control before province roll-up. Retain this as
the leading candidate model and keep active v16 unchanged.

Artifacts:

- `outputs/district_candidate_base_clean_v8_ablation/summary.csv`
- `outputs/district_candidate_base_clean_v8_ablation/by_election.csv`
- `outputs/district_candidate_base_clean_v8_ablation/national_predictions.csv`
- `outputs/district_candidate_base_clean_v8_ablation/input_manifest_diff.csv`
- `outputs/district_candidate_base_clean_v8_ablation/decision.json`
