# Current Model Performance (2026-07-16)

> Historical snapshot. The maximum-normalized electorate signal in this file was corrected
> on 2026-07-17. Use `docs/CURRENT_MODEL_PERFORMANCE_20260717.md` for current values.

## Evaluation Boundary

- active workspace: `C:\english_folder\poll_project`
- scored and tuning elections: 2002, 2007, 2012, 2017, 2022
- rolling warmup only: 1997
- post-2022 presidential outcomes: excluded
- predictors: 10 fixed Ridge predictors, 11 fitted parameters including intercept
- active Ridge alpha: 1.20
- active post-model electorate layer: core / critical / swing issue sensitivity
- active electorate gains: terrain 0.00, preference 0.04, turnout 0.00, nonvoter 0.00

The primary regional metric first weights region-candidate absolute errors by observed
`contest_votes` within each election and then gives the five elections equal weight. It is
a post-election diagnostic because actual turnout supplies the evaluation weights. The
historical row-unweighted result remains a secondary compatibility statistic.

## Primary Metrics

| Evaluation | MAE | RMSE | Median AE | Maximum AE | R2 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Strict nested, contest-vote weighted macro | **4.5155%p** | n/a | n/a | n/a | n/a | primary model-selection estimate |
| Strict nested, row unweighted | 4.4191%p | n/a | n/a | n/a | n/a | compatibility reference |
| Final selection-sample rolling, row unweighted | 3.6909%p | 4.5672%p | 3.2971%p | 11.5233%p | 0.9576 | chronological but selected on same sample |
| Final selection-sample rolling, contest-vote weighted macro | 3.6273%p | n/a | n/a | n/a | n/a | post-election weighted diagnostic |
| LOEO, row unweighted | 5.08%p | n/a | n/a | n/a | n/a | stability only; later elections train earlier targets |
| Full fit, adjusted deterministic prediction | 4.9003%p | 5.9959%p | 4.5015%p | 13.2578%p | 0.9269 | in-sample description |
| Full-fit Monte Carlo expected share | 6.4264%p | 7.8578%p | 5.2213%p | 18.4041%p | 0.8745 | nonlinear draw-mean diagnostic |

The deterministic point prediction and Monte Carlo expected share are different definitions.
The official point MAE is not calculated from the Monte Carlo `pred` column.

## Strict Nested Result

The frozen pre-layer baseline was `4.627768%p`. The selected electorate layer is
`4.515503%p`, an improvement of `0.112265%p`.

| Election | Baseline weighted MAE | Electorate-layer weighted MAE | Improvement |
|---|---:|---:|---:|
| 2002 | 3.5712%p | 3.5712%p | 0.0000%p |
| 2007 | 6.3463%p | 6.3463%p | 0.0000%p |
| 2012 | 6.2182%p | 5.6653%p | +0.5529%p |
| 2017 | 5.3232%p | 5.2024%p | +0.1208%p |
| 2022 | 1.6800%p | 1.7924%p | -0.1124%p |

The 2002 and 2007 outer folds keep the electorate gain at zero because fewer than two prior
scored elections are available for layer selection. The 2022 worsening is retained in the
reported result and is not hidden by an aggregate average.

All declared adoption guards passed:

- nested weighted improvement at least 0.10%p
- no election worsened by more than 0.50%p
- 2022 worsened by no more than 0.25%p
- third-candidate regional shape correlation did not fall by 0.03
- Gyeongbuk weighted MAE did not worsen by more than 0.50%p

## Active Rolling Diagnostics

| Election | Row-unweighted MAE | Contest-vote weighted MAE |
|---|---:|---:|
| 2002 | 3.8781%p | 3.4078%p |
| 2007 | 4.4427%p | 4.6914%p |
| 2012 | 3.7840%p | 4.1604%p |
| 2017 | 3.7417%p | 4.0846%p |
| 2022 | 2.2840%p | 1.7924%p |
| Overall | 3.6909%p | 3.6273%p equal-election macro |

The rolling national aggregation below uses observed regional contest votes and is not a
deployable pre-election metric.

| Election | Candidate | Predicted | Actual | Absolute error |
|---|---|---:|---:|---:|
| 2002 | Roh Moo-hyun | 49.845% | 51.217% | 1.372%p |
| 2002 | Lee Hoi-chang | 50.155% | 48.783% | 1.372%p |
| 2007 | Lee Myung-bak | 55.277% | 54.140% | 1.137%p |
| 2007 | Chung Dong-young | 32.839% | 29.089% | 3.750%p |
| 2007 | Lee Hoi-chang | 11.883% | 16.771% | 4.887%p |
| 2012 | Park Geun-hye | 55.407% | 51.773% | 3.634%p |
| 2012 | Moon Jae-in | 44.593% | 48.227% | 3.634%p |
| 2017 | Moon Jae-in | 45.344% | 47.476% | 2.132%p |
| 2017 | Hong Joon-pyo | 32.348% | 27.773% | 4.574%p |
| 2017 | Ahn Cheol-soo | 22.308% | 24.751% | 2.443%p |
| 2022 | Yoon Suk-yeol | 51.025% | 50.380% | 0.646%p |
| 2022 | Lee Jae-myung | 48.975% | 49.620% | 0.646%p |

Candidate-row national MAE is `2.5190%p`; equal-election national macro MAE is `2.3920%p`.

## Electorate Layer

The layer is an ecological decomposition, not observed individual voter labels.

- durable core: lower-quartile, long-half-life historical bloc support
- critical support: support above the durable core, shrunk by historical volatility
- swing mass: residual regional voting mass
- nonvoter reservoir: explicit but inactive until official prior regional turnout history exists

Issue classes are economy, housing, integrity/candidate, regime, social, security, coalition,
and regional. Candidate direction comes from PIT-safe own-party support/defense versus
cross-party attack posture. Issue salience and sentence character scale the response. Fixed
templates enforce lower preference elasticity for the core than for critical and swing mass.

The existing regional partisan moderation remains active. The old aggregate party-tone
adjustment is replaced to prevent duplicated candidate-treatment signals. Terrain anchoring
was tested and rejected (`gain=0`). Turnout and nonvoter gains are zero because the official
turnout-history input currently has zero rows.

## Coefficients and Collinearity

The Ridge coefficients are unchanged. Prior-family VIF remains very high: partisan prior
147.75, slot-A prior 77.47, and slot-B prior 71.31. Ridge regularization stabilizes prediction,
but these coefficients must not be interpreted as independent causal effects. The electorate
layer is post-model and adds no fitted Ridge coefficient.

## Prediction Intervals

The production full-fit interval table now includes the active electorate transformation for
every coefficient draw.

| Nominal level | Mean interval width / coverage | Predictive width / coverage |
|---|---:|---:|
| 90% | 3.51%p / 17.1% | 16.35%p / 64.8% |
| 95% | 4.19%p / 21.1% | 19.30%p / 73.9% |
| 99% | 5.47%p / 26.6% | 24.61%p / 84.4% |

These in-sample coverages are not calibrated enough to claim nominal 90/95/99% coverage.
Point MAE is unaffected by this interval-calibration limitation.

The active strict rolling interval path was also rerun with 100 draws per fold. It exactly
reproduced the active rolling point MAE of `3.690907%p`. At residual scale 2.0, exploratory
90/95/99% coverage was 90.5% / 95.0% / 96.5%, with mean widths 15.75 / 17.62 / 20.27%p.
The earlier 1,000-draw values are not reused because that rerun exceeded the execution limit;
100 draws are sufficient for path consistency but not final interval calibration.

## Verification Artifacts

- electorate selection summary: `outputs/electorate_layer_experiment/summary.json`
- nested shadow predictions: `outputs/electorate_layer_experiment/nested_shadow_predictions.csv`
- active config: `data/config/electorate_layers.json`
- canonical rolling rows: `presidential_issue_engine/report/tables/issue_vote_engine_rolling_predictions.csv`
- rolling national diagnostic: `presidential_issue_engine/report/tables/issue_vote_engine_rolling_national_summary.csv`
- full-fit Monte Carlo table: `presidential_issue_engine/report/tables/issue_vote_engine.csv`
- active rolling interval audit: `outputs/predictor_interval_audit/rolling_interval_experiment.csv`

Latest verification: `257 passed`. Strict PIT target-outcome invariance passed for 215/215
rows, and the through-2022 selection-boundary audit passed.
