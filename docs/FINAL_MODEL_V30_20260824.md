<!-- active-model-version: v30 -->
# Final presidential model V30 — forecast-time regional weighting

V30 is V29 with one change: the two terminal transforms weight each candidate's
national level by the **previous** election's regional valid votes rather than
by the target election's own turnout. Nothing else moves — the Ridge stack, the
predictors, the shock structure, the V28 external-model boundary, both transform
forms and the gain are V29's.

## Why

`contest_votes` is the target election's regional turnout, known only after the
count. A postprocess reading it consumes an outcome of the election it is
predicting. The 2025 prospective path already refused it and substituted prior
volumes; the scored panel did not, so the historical figures described something
no forecast could have produced and the two paths weighted differently. They now
weight the same way.

## Frozen development metrics

- regional equal-election macro MAE: `2.5664447526782004%p`
- national equal-election macro MAE: `0.7204374174124484%p`
- winner accuracy: `0.8`
- scored rows: `232`
- post-2022 outcomes used: `false`

Both improved against V29's `2.5736074405126663` and `0.7262497116354087`. That
is an outcome, not the justification: the decision was taken on the availability
of the weight, and the cost projected at decision time was `+0.0119%p`.

By election, regional weighted MAE:

| election | V29 | V30 |
| --- | ---: | ---: |
| pres_2002 | 2.957 | 2.964 |
| pres_2007 | 3.730 | 3.711 |
| pres_2012 | 2.387 | 2.387 |
| pres_2017 | 2.607 | 2.585 |
| pres_2022 | 1.187 | 1.186 |

## The 1997 warmup table

2002's predecessor is 1997, outside the scored results table, so its regional
turnout ships separately as
`presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv`.

Source: 국사편찬위원회 한국사데이터베이스, 제15대 대통령선거 (1997-12-18). The
transcription was verified by summation — electorate `32,290,416`, votes cast
`26,042,633`, valid votes `25,642,438`, each matching the published national
total exactly. Valid votes are the column used, matching what `contest_votes`
counts in the panel.

## Error columns now describe the shipped model

`err_pp` and `abs_err_pp` were computed from `official_pred`, a pre-layer
baseline whose name read like the opposite. All 232 rows differed from
`layer_pred`, by 5.89 percentage points on average, so the artifact's own error
columns implied a regional macro of 6.35 where the headline says 2.57.

The figures were never wrong for what they measured; the names were wrong about
what that was. In V30 `err_pp` and `abs_err_pp` describe `layer_pred`, and the
baseline is carried as `baseline_pre_layer_pred` with matching error columns.

## AI boundary

Unchanged from V28 and re-audited:

- hosted inference API: none
- downloaded model weights at runtime: none
- external neural encoder at runtime: none
- external-model-derived active input: one compact frozen candidate-issue aggregate
- fitted component: scikit-learn Ridge plus deterministic project transforms

## Evidence boundaries

2002–2022 remain a development panel, not an untouched holdout. The 2025
artifact remains a corrected D-1 demonstration.

The scored panel is still defined by which candidates cleared roughly 1% of the
actual vote — a declared modelling scope, recorded with its consequences in
`DIAGNOSIS_SCORING_SCOPE_20260824.md`. The headline metric still weights by
`contest_votes`, which is a post-hoc diagnostic weighting disclosed as such and
entering no prediction.
