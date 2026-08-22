# The national metric is largely error cancellation

## Status

- Date: 2026-08-22
- Status: diagnosis only; no change made
- Explains the result recorded across
  `EXPERIMENT_DISPERSION_ALTERNATIVES_20260822.md` and
  `EXPERIMENT_PERSON_HISTORY_AND_CORE_20260822.md`

Seven corrections were tried against regional error. Every one improved regional
accuracy and degraded the national metric. That looked like a trade-off between
two kinds of accuracy. It is not.

## The arithmetic

The national candidate metric compares a vote-weighted mean of regional
predictions with the same mean of realised shares, so signed regional errors
offset inside it. Define

    cancellation = 1 - |weighted signed error| / weighted absolute error

At 0 the regional errors all point the same way and the national figure carries
the full regional error. At 1 they offset exactly and the national figure is
zero regardless of how wrong each region is.

## What the panel shows

| election | cancellation | national error | regional absolute |
| --- | ---: | ---: | ---: |
| **pres_2002** | **0.240** | **2.342** | 2.752 |
| pres_2022 | 0.584 | 0.274 | 1.134 |
| pres_2007 | 0.848 | 0.661 | **4.272** |
| pres_2017 | 0.922 | 0.201 | 3.025 |
| pres_2012 | **0.947** | **0.127** | 2.378 |

Cancellation against national error: **r = -0.869**.
Regional accuracy against national error: **r = 0.169**.

**The national metric tracks how much the errors offset, not how accurate the
regions are.** 2007 has the worst regional error in the panel at 4.272 and a
good national figure at 0.661, because 84.8 % of its regional error cancels.

The extreme case is 홍준표 2017: a national error of **+0.029 %p** against a
mean absolute regional error of **3.932 %p**, a cancellation of **0.993**. He is
about 12 points short in 대구 and 경북 and correspondingly long elsewhere, and
the two halves erase each other.

2002 is the mirror image and explains its own outlier status: its cancellation
is 0.240, the lowest in the panel, because its errors are systematic rather than
compensating. It is the worst national fold not because its regions are worst -
they are mid-panel at 2.752 - but because nothing offsets.

## Why this explains the seven failed corrections

Compression produces offsetting errors **by construction**. A candidate whose
regional spread is too narrow is short in their strongholds and long in hostile
territory, and those errors are opposite in sign. So compression is invisible to
the national metric, and any correction that removes compression necessarily
removes the cancellation the national figure was made of.

The corrections were not breaking a calibration. They were removing the
arithmetic that produced it.

That reframes the trade-off recorded earlier. It is not that regional shape and
national level are two competing goods this model cannot serve at once. It is
that the national figure was never independent evidence about regional levels,
so improving the regions was always going to move it.

## What this means for reading the headline

The published national candidate macro of 0.7210 %p should not be read as
evidence that regional levels are right. Across the panel 69.1 % of regional
error cancels inside it on average, and in two elections more than 92 %.

Three things follow.

The national figure and the regional figure are not two independent checks on
the same model. Wherever both are quoted, the cancellation should be quoted
beside them.

A correction that improves regional error while worsening national error is not
obviously a bad trade, and the earlier rejections should be reread with that in
mind. They were rejected partly on the national metric moving, and that metric
is a weaker signal than it looked.

And the pattern to watch for is a fold like 2002 - low cancellation with a
mid-panel regional error - because that is the signature of systematic rather
than compensating error, which is the kind that does not average away.

## What is not claimed

That the national metric is meaningless. Winner accuracy depends on national
levels, and the model gets four of five. Cancellation makes the *magnitude* of
the national error a poor proxy for regional accuracy; it does not make the
ordering of candidates within an election arbitrary.

And with five folds, r = -0.869 is a pattern rather than an estimate.
