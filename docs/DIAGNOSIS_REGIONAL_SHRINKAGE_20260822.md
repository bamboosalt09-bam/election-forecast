# Regional shape is compressed in proportion to the third candidate's size

## Status

- Date: 2026-08-22
- Status: diagnosis only; **no change made, and none proposed here**
- **Mechanism superseded** by `DIAGNOSIS_STRONGHOLD_ERRORS_20260822.md`.
  The measurements below stand, but the explanation - one phenomenon
  scaling with third-candidate size - was too coarse. The correlation is
  a symptom of two unrelated defects that both happen to occur in the two
  elections with a large third candidate, which is why the swept rescale
  helped 2007 and 2017 while hurting 2002 and 2022.
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

## The correction was built and measured, and is not adopted

Asked whether this needs a model change, the answer is measured rather than
argued. `scripts/evaluate_regional_dispersion_calibration.py` expands each
candidate's regional deviations around their own national level by
`1 + gain * predicted_third_share` and renormalises. The index is the model's
own predicted third share, available at forecast time, so the transform reads
no outcome; the gain is swept rather than fitted.

### Where the compression enters

Before the sweep, one thing had to be settled: which stage compresses.

| stage | 2002 | 2007 | 2012 | 2017 | 2022 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shadow_pred` | 1.066 | **0.689** | 0.984 | **0.811** | 1.224 |
| `layer_pred` (final) | 1.047 | **0.681** | 1.038 | **0.791** | 1.098 |

It is there in the earliest modelled stage and the postprocess layers neither
create nor remove it. A correction would therefore have to live in the fitted
base, which is the highest-risk place to put one: every election depends on it,
unlike the bolt-on transforms all previous structural work touched.

### Shrinkage is not by itself the defect

A regularised predictor should have less variance than the outcome; the
conditional mean does too. Predicted spread below realised spread is what Ridge
is supposed to produce, and reproducing the full outcome variance would be
overfitting.

What marks 2007 and 2017 is the slope of realised on predicted: 1.141, 1.276
and 2.043 in 2007, 1.269 and 1.312 in 2017, against roughly 1.0 everywhere
else. Calibrated shrinkage gives slope 1. So those two elections are
miscalibrated, and the other three are not.

### The sweep

| gain | regional macro | national macro | dispersion ratio | winners |
| ---: | ---: | ---: | ---: | ---: |
| **0.00** (shipped) | **2.7122** | **0.7210** | 0.9308 | 4/5 |
| 0.25 | 2.6584 | 0.7210 | 0.9499 | 4/5 |
| 0.50 | 2.6103 | 0.7210 | 0.9690 | 4/5 |
| 0.75 | 2.5936 | 0.7210 | 0.9881 | 4/5 |
| 1.00 | **2.5829** | 0.7228 | 1.0056 | 4/5 |
| 1.50 | 2.5973 | 0.7312 | 1.0373 | 4/5 |
| 2.00 | 2.6424 | 0.7519 | 1.0678 | 4/5 |

Regional weighted MAE by election, at the best gain:

| election | gain 0.00 | gain 1.00 | change |
| --- | ---: | ---: | ---: |
| pres_2007 | 4.272 | 3.793 | **-0.479** |
| pres_2017 | 3.025 | 2.687 | **-0.338** |
| pres_2012 | 2.378 | 2.378 | 0.000 |
| **pres_2002** | 2.752 | 2.913 | **+0.161** |
| **pres_2022** | 1.134 | 1.144 | +0.010 |

### Why not

**It improves the two compressed elections and degrades the two that were not.**
The net gain is positive, but the gain was chosen from the same five outcomes,
so this is fitting the panel by construction. It is the mirror image of the
pattern the V26 record already flags on 2002 - helping where the correction was
aimed and leaving or worsening the rest.

**The national metric does not improve.** 0.7210 goes to 0.7228. The headline
V26 was promoted on does not move, and winner accuracy is 4/5 at every gain
including 2.00.

**The optimum is flat.** Gains from 0.75 to 1.50 span 2.5829 to 2.5973, a range
of 0.014. Locating a minimum on a surface that flat is selecting noise.

The apparent coincidence - that the best gain is also where the dispersion
ratio reaches 1.0 - is arithmetic rather than corroboration. Minimising
absolute error and matching outcome variance are closely related objectives, so
they meet near the same point by construction.

**And it would go in the fitted stage** to fix a pattern with two supporting
elections. That is the worst available ratio of risk to evidence in the model.

### What the sweep is good for

Not as a correction, but as confirmation of the diagnosis. That gain > 0 helps
exactly 2007 and 2017, does nothing to 2012, and hurts 2002 and 2022 is
independent evidence that the compression is real in the first two and absent
in the others. The measurement earns its place as a diagnostic even though the
transform does not earn promotion.
