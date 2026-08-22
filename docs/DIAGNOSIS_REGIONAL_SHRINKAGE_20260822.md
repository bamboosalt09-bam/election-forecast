# Regional shape is compressed in proportion to the third candidate's size

## Status

- Date: 2026-08-22
- Status: diagnosis only; **no change made, and none proposed here**
- Post-2022 outcomes used: none beyond the scored panel this evaluates

The question that prompted this: is 2007's error an under-corrected regression
toward the regional mean? Yes - and it is not specific to 2007.

## The measurement

For each candidate, the standard deviation of the predicted regional shares
against the standard deviation of the realised ones. A ratio below 1 means the
model produces less regional spread than actually occurred; the slope of actual
on predicted is the factor the predicted deviations would need.

| election | candidate | actual sd | predicted sd | ratio | slope |
| --- | --- | ---: | ---: | ---: | ---: |
| 2007 | 이명박 | 21.95 | 18.72 | 0.853 | 1.141 |
| 2007 | 정동영 | 27.52 | 21.11 | 0.767 | 1.276 |
| **2007** | **이회창** | **9.56** | **4.05** | **0.423** | **2.043** |
| 2017 | 문재인 | 12.14 | 9.03 | 0.744 | 1.269 |
| 2017 | 홍준표 | 15.97 | 11.89 | 0.744 | 1.312 |
| 2017 | 안철수 | 4.88 | 4.32 | 0.885 | 0.822 |
| 2002 | 노무현 | 24.27 | 23.80 | 0.981 | 1.014 |
| 2002 | 이회창 | 23.01 | 24.46 | 1.063 | 0.936 |
| 2012 | 박근혜 | 21.30 | 22.10 | 1.038 | 0.949 |
| 2022 | 이재명 | 18.91 | 17.91 | 0.947 | 1.050 |
| 2022 | 윤석열 | 18.66 | 18.33 | 0.982 | 1.014 |

2002, 2012 and 2022 show no compression at all. 2007 and 2017 do, and 이회창
2007 is extreme: the model reproduces **42 %** of his realised regional spread.

## It scales with the third candidate

| election | third candidate share | two-major ratio | all-candidate ratio |
| --- | ---: | ---: | ---: |
| 2012 | 0.00 | 1.038 | 1.038 |
| 2022 | 2.41 | 0.964 | 1.098 |
| 2002 | 3.92 | 1.022 | 1.047 |
| **2007** | **16.77** | **0.810** | **0.681** |
| **2017** | **24.75** | **0.744** | **0.791** |

Third-candidate share against the two majors' compression: **r = -0.975**;
against every candidate's compression, r = -0.866. Near-monotone across the
panel, and the two compressed elections are exactly the two with a substantial
third candidate.

At n = 5 this is a pattern with a plausible mechanism, **not evidence**. It
should not be quoted as an estimated relationship.

## What it is not

**It is not a 충청 home-base gap for 이회창.** That was the first hypothesis and
it is only a quarter of the story. 이회창's three largest under-predictions are
충남 -13.72, 대전 -10.71 and 충북 -6.72, which is a real 충청 concentration the
model flattens. But setting those three regions to their realised values moves
2007's regional weighted MAE only from 4.272 to 3.931, while getting 이회창
right everywhere moves it to 2.930. The 충청 piece is 0.341 of a 1.342
opportunity.

He is also under-predicted in 경남 -5.06 and 부산 -4.10 and **over**-predicted
in 울산 +3.27. Under in his strong regions and over in his weak ones is
compression, not a missing home-base term.

**The Chungcheong identity layer exists and does not cover him.**
`data/raw/chungcheong_identity_alignment.csv` has two rows - 노무현 2002 on a
dated administrative-capital commitment, 박근혜 2012 on a dated party merger -
both event-based evidence with sources. 이회창's 충청 tie is biographical, a
different evidence type, and no row exists for it.

## Why no fix is proposed here

This gap was found by reading 2007's residuals. Adding a row for 이회창, or
adding a dispersion correction scaled by third-candidate size, would be
hypothesis development driven by scored outcomes. That is legitimate research
and it is what most of the V24 and V25 record already is, but it must be
declared rather than presented as a prior structural insight.

Two further cautions specific to this one.

A dispersion correction indexed on third-candidate size would be fitted on
**two** elections, since only 2007 and 2017 show the effect. That is the same
observation shortage every structural layer already has.

And the compression is measured against realised regional variance, so a
correction calibrated to close it is calibrated directly on the outcome
variance it is scored against. That is a tighter loop than the existing layers,
which key on pre-election facts.

## Reproduce

The measurement is a few lines against
`outputs/active_presidential_nested_v26/nested_predictions.csv`: group by
election and candidate, compare `layer_pred.std()` with `actual.std()`.
