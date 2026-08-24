<!-- active-model-version: v30 -->
# A proposed metric change, investigated and rejected

## Status

- Date: 2026-08-25
- Status: **rejected**; the reported headline is unchanged
- Nothing in the model, the predictions or any frozen artifact was altered
- Reproduce: `python scripts/evaluate_metric_weighting_sensitivity.py`

## The proposal

V30 stopped letting the model read `contest_votes` — the target election's own
regional turnout, which does not exist until the count. The proposal was to
apply the same reasoning to the report: make the headline metric weight by the
previous election's regional volumes too, so that the published figure would be
one a forecaster could have computed on the eve of the election.

It was attractive because it moves the numbers the *unfavourable* way
(regional `2.5664` → `2.6705`), which is not the direction metric-shopping
goes, and because V30 had just shipped 1997's turnout, so prior-election
weighting became defined for all five scored elections rather than four.

## Why it is wrong

The two cases are not the same, and the difference is the whole point.

**A model reading `contest_votes` is a leak.** The prediction consumes an
outcome of the election it is predicting. V30 fixed that, correctly.

**A metric aggregating by `contest_votes` is not a leak.** National vote share
is *defined* as the vote-weighted mean of regional shares, and the weights are
the votes actually cast. The model predicts regional shares; it does not
predict turnout and never claimed to. Aggregating with the real votes hands the
model no information — it states the target quantity correctly, and isolates
share error from turnout-mix error the model was not forecasting.

The objection that sank the proposal came from applying its own logic
symmetrically. Reweighting the *realised* national result by prior-election
volumes produces "what the result would have been had regions turned out like
last time" — a counterfactual that never happened. If that is illegitimate for
the realised side, it is equally illegitimate for the predicted side: both are
aggregations of regional shares into a national one, and both need the same
weights to mean anything.

## What the investigation did establish

Swapping the aggregation weight is a sensitivity measurement, and one result is
worth keeping.

| figure | as reported (`contest_votes`) | swapped weight |
| --- | ---: | ---: |
| regional equal-election macro MAE | `2.566445%p` | `2.670461%p` |
| winner accuracy | `0.8` | `0.6` |

The lost call is 2022, in national levels:

| | `contest_votes` | swapped weight |
| --- | ---: | ---: |
| 윤석열 predicted (slot A) | `48.6564` | `48.4793` |
| 이재명 predicted (slot B) | `48.5250` | `48.6859` |
| predicted margin, 윤석열 − 이재명 | `+0.1314` | `−0.2066` |
| realised margin | `+0.7410` | — |

**The 2022 winner call rests on a 0.13%p predicted margin, and swapping the
aggregation weight moves that margin by 0.34%p — more than the margin itself.**
The model does not separate the two leading candidates in 2022. The call comes
out right because the actual turnout mix leaned the way it did, not because the
prediction distinguishes them.

This is not caused by V30. The same ordering is already present at
`v26_pre_regional_polarization_pred`, three stages earlier, and is unchanged
through the rest of the chain.

## An internal inconsistency, noted

V30's terminal transform preserves the **forecast-time-weighted** national
level exactly — measured shift `0.000000000000` — while the pipeline reports
the `contest_votes`-weighted level, which the transform moves by up to
`0.005%p`. The transform conserves one aggregate and the report publishes
another.

The quantity is small and it is not a leak in either direction, but the two
ought to be the same thing and are not. Recorded here rather than changed:
changing it means a new version, and the effect is two orders of magnitude
below the errors being reported.

## What was kept from this work

- `EXPERIMENT_EXTERNAL_MODEL_INPUT_REMOVAL_20260825.md` — the removal cost of
  the external-model-derived input, re-measured on V30 instead of quoting a
  V27-era figure.
- `evaluate_ex_ante_weighting.py` now covers all five scored elections, because
  1997 supplies the first one's predecessor. The ex-ante weightings stay what
  they were: companion figures published beside the headline, not the headline.

## Related

- `EXPERIMENT_V30_FORECAST_TIME_WEIGHTS_20260824.md` — the model-side change,
  which stands
- `DIAGNOSIS_ERROR_CANCELLATION_20260822.md` — the other reason a national
  figure should not be read as regional accuracy
- `DIAGNOSIS_SCORING_SCOPE_20260824.md` — the panel-membership rule, likewise
  recorded rather than changed
