<!-- active-model-version: v30 -->
# V30: weighting the terminal transforms with what a forecaster has

## Status

- Date: 2026-08-24
- Status: **promoted**; predecessor V29 frozen and unchanged
- Post-2022 outcomes used: none

## What was wrong

V27 and V29 each end in a transform that expands or adjusts a candidate's
regional deviations around that candidate's own national level. To locate that
level they took a weighted mean, and the weight was `contest_votes` — the
**target election's own regional turnout**.

That number exists only once the votes are counted. A postprocess reading it
consumes an outcome of the election it is predicting.

The project already knew this without stating it. The 2025 prospective path
refuses `contest_votes` and substitutes the previous election's regional
volumes, because a forecast cannot have the target's turnout. The scored panel
kept using it, so the historical figures described something no forecast could
have produced, and the two paths weighted differently.

## The change

Every scored election now weights by its predecessor's regional valid votes.
Nothing else moves: the Ridge stack, the predictors, the shock structure, the
V28 external-model boundary, both transform forms and the gain are V29's.

2002's predecessor is 1997, a warmup election outside the scored results table,
so its regional turnout is carried in
`presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv`. That
avoids the alternative — equal-region weights for the first fold — which cost
`+0.0866%p` on 2002 alone when measured.

## The 1997 table

Source: 국사편찬위원회 한국사데이터베이스, 제15대 대통령선거 (1997-12-18),
`https://db.history.go.kr/id/tcct_1997_12_18_0010`.

Sixteen regions with electorate, votes cast and valid votes. The transcription
was checked by summation rather than by eye, and all three columns reproduce the
published national totals exactly:

| column | sum | published |
| --- | ---: | ---: |
| 선거인수 | 32,290,416 | 32,290,416 |
| 투표수 | 26,042,633 | 26,042,633 |
| 유효투표수 | 25,642,438 | 25,642,438 |

That check matters: an earlier attempt to take the same figures from a secondary
source produced regional numbers that summed **8,864 votes short** of the
published total, and there was no way to tell which region was wrong. Those
numbers were discarded rather than adjusted.

Valid votes are used, to match what `contest_votes` counts in the scored panel.
Electorate and votes cast are retained because electorate is published *before*
an election and is the natural weight if a stricter forecast-time rule is ever
wanted.

## Result

| metric | V29 | V30 |
| --- | ---: | ---: |
| regional equal-election macro MAE | 2.5736074405126663 %p | **2.5664447526782004 %p** |
| national equal-election macro MAE | 0.7262497116354087 %p | **0.7204374174124484 %p** |
| winner accuracy | 0.8 | 0.8 |
| prediction rows | 232 | 232 |

By election, regional weighted MAE:

| election | V29 | V30 |
| --- | ---: | ---: |
| pres_2002 | 2.957 | 2.964 |
| pres_2007 | 3.730 | 3.711 |
| pres_2012 | 2.387 | 2.387 |
| pres_2017 | 2.607 | 2.585 |
| pres_2022 | 1.187 | 1.186 |

**Both headline figures improved.** That is recorded as an outcome, not as the
justification. The change was made because the old weight was not available at
forecast time, and it would have been made had the numbers moved the other way —
the measurement was taken before the decision, and the projected cost at the
time was `+0.0119%p`, which is what equal-region weighting for 2002 would have
cost. Sourcing 1997 removed that.

The leak was wide open and carried almost nothing. Regional vote shares move
slowly between elections — 1997 and 2002 regional sizes correlate at `0.996` —
so the weight barely differed. That is the argument for closing it cheaply, not
an argument that it did not matter.

## What this does not fix

The **metric** still weights by `contest_votes`. That is a post-hoc diagnostic
weighting, already disclosed as such in the README alongside ex-ante
alternatives, and it does not enter any prediction. Only the transforms are
changed here.

The scoring panel is still defined by which candidates cleared roughly 1% of the
**actual** vote. That is a declared modelling scope — two-way contest versus one
with a viable third — codified rather than chosen case by case, and it is
recorded in `DIAGNOSIS_SCORING_SCOPE_20260824.md` rather than changed here.
