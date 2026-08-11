# Current Model Performance (2026-07-27)

## Active policy

`active_strict_nested_v9_party_context_cohesion`

- 1997 is rolling warmup only.
- 2002, 2007, 2012, 2017, and 2022 are scored development folds.
- Every target election is excluded from its own Ridge fit.
- The same `structural_mega_shock_regime` pipeline is applied to every fold and
  is the declared future deployment pipeline.
- Each structural, mega-issue, government-shock, or contest-regime component is
  bounded and becomes a no-op without qualifying dated evidence.
- 2025 outcomes are prohibited from fitting, tuning, ablation, and comparison.
- Regional and national metrics use observed target-election contest votes and
  are therefore post-election aggregation diagnostics.

## Aggregate metrics

| Metric | V8 | V9 | Change |
|---|---:|---:|---:|
| Regional weighted macro MAE | 3.9499%p | **3.8584%p** | -0.0915%p |
| National candidate macro MAE | 2.6298%p | **2.4769%p** | -0.1529%p |
| Winner accuracy | 80% | **80%** | 0%p |

## Election metrics

| Election | Regional weighted MAE | National candidate MAE | Winner correct |
|---|---:|---:|:---:|
| 2002 | 4.0173%p | 3.3455%p | No |
| 2007 | 6.4407%p | 4.4472%p | Yes |
| 2012 | 2.6468%p | 1.2717%p | Yes |
| 2017 | 4.6042%p | 3.2122%p | Yes |
| 2022 | 1.5828%p | 0.1079%p | Yes |

## National predictions

| Election | Candidate | Predicted | Actual | Error |
|---|---|---:|---:|---:|
| 2002 | Roh Moo-hyun | 47.872% | 51.217% | -3.345%p |
| 2002 | Lee Hoi-chang | 52.128% | 48.783% | +3.345%p |
| 2007 | Lee Myung-bak | 47.944% | 54.140% | -6.197%p |
| 2007 | Chung Dong-young | 35.760% | 29.089% | +6.671%p |
| 2007 | Lee Hoi-chang | 16.297% | 16.771% | -0.474%p |
| 2012 | Park Geun-hye | 50.502% | 51.773% | -1.272%p |
| 2012 | Moon Jae-in | 49.498% | 48.227% | +1.272%p |
| 2017 | Moon Jae-in | 44.158% | 47.476% | -3.318%p |
| 2017 | Hong Joon-pyo | 32.591% | 27.773% | +4.818%p |
| 2017 | Ahn Cheol-soo | 23.251% | 24.751% | -1.500%p |
| 2022 | Yoon Suk-yeol | 50.272% | 50.380% | -0.108%p |
| 2022 | Lee Jae-myung | 49.728% | 49.620% | +0.108%p |

## Interpretation

V9 corrects the interpretation of party-internal context. V8 treated centered
party support and fragmentation as a direct adjustment to total candidate
support. V9 instead lets weak context release only a bounded share of existing
candidate-aligned core and critical-support mass into the regional flexible
pool. Core defection is capped at 2%, critical-support defection at 15%, and
zero-confidence context is an exact no-op.

The correction improves four elections and slightly worsens 2017. This mixed
movement is consistent with a general structural rule rather than a targeted
election override.

The five-election score remains development-sample evidence. Numeric gains and
stage definitions were developed while through-2022 outcomes were visible, so
the result must not be presented as an untouched external holdout.

## Verification

- full test suite: `368 passed`
- strict PIT deep audit: PASS, target-outcome invariance `215/215`
- slot-predictor leakage audit: PASS
- through-2022 selection boundary audit: PASS
- active fold audit: target excluded, realized-slot predictors absent
- all five folds use one fixed deployment stage; no target-specific stage choice
- active input manifest contains SHA-256 hashes for every CSV read
- Assembly match-level PIT audit remains unavailable in the current audit run:
  `assembly_matches_present=0`

## Artifacts

- `outputs/active_presidential_nested_v9/summary.json`
- `outputs/active_presidential_nested_v9/by_election.csv`
- `outputs/active_presidential_nested_v9/national_predictions.csv`
- `outputs/active_presidential_nested_v9/nested_predictions.csv`
- `outputs/active_presidential_nested_v9/stage_selection_audit.csv`
- `outputs/active_presidential_nested_v9/input_manifest.csv`
- `docs/PARTY_CONTEXT_COHESION_V9_20260727.md`
