<!-- active-model-version: v32 -->
# Why the model under-states a candidate's own region

## Status

- Date: 2026-08-26
- **Measurement only.** No model change is proposed as settled here, and none
  was made.
- Everything below is measured on the scored panel, 2002–2022. **The 2025
  outcome is not used anywhere in this document.**
- Claims are graded **observed**, **demonstrated**, **inferred** or
  **unresolved**

## The observation being tested

That the model under-predicts a candidate in the region that is their base —
홍준표 in TK, the liberal candidate in Honam, 안철수 in Honam in 2017.

## It is true in some cases and inverted in others

**Observed.** 2017, error in percentage points, negative meaning
under-predicted:

| | 문재인 | 안철수 | 홍준표 |
| --- | ---: | ---: | ---: |
| 대구 | +3.94 | +2.97 | **−6.91** |
| 경북 | +2.68 | +2.90 | **−5.58** |
| 광주 | +1.06 | **−1.35** | +0.29 |
| 전북 | −2.19 | **+1.38** | +0.81 |
| 전남 | +4.71 | **−4.57** | −0.14 |

홍준표 in TK is under-predicted, heavily. 안철수 is under-predicted in 광주 and
전남 but **over**-predicted in 전북. 문재인 in Honam is **over**-predicted
except in 전북.

**Inferred.** In 전남 the two errors are near mirror images (`+4.71` and
`−4.57`). Within a region the shares must sum to one, so this is not a failure
to see that 전남 is liberal — it is a failure to divide the liberal vote between
two candidates in the same lane.

## The bias is specific to nationally weak candidates

**Observed.** Taking the 26 rows where a candidate's regional share exceeds
their own national share by more than 20 points:

| candidate's national share | rows | mean error |
| --- | ---: | ---: |
| **under 35%** | 6 | **−8.01pp** |
| 35% or more | 20 | **+0.02pp** |

Correlation between national share and residual error on those rows:
**+0.677**.

| | national | region | error |
| --- | ---: | ---: | ---: |
| 2007 이회창 | 16.8% | 충남 37.5% | **−11.46** |
| 2007 정동영 | 29.1% | Honam 86.5% | **−8.03** |
| 2017 홍준표 | 27.8% | TK 56.1% | **−6.24** |
| 2002 노무현 | 49.2% | Honam 94.0% | −4.95 |
| 2012 문재인 | 48.2% | Honam 89.6% | +2.71 |
| 2022 이재명 | 48.4% | Honam 85.5% | −1.88 |

## It is a shape problem, not a level problem

**Demonstrated.** The national levels are almost all correct. 홍준표 2017:
predicted `27.7%`, actual `27.8%` — an error of `−0.04pp`, while 대구 is out by
`−6.91pp`. Replacing every candidate's national level with the true one and
keeping the predicted regional shape lowers panel regional MAE only from
`3.142` to `2.873` (−0.27pp).

So the model knows how big each candidate is. It distributes that size too
evenly across regions.

## What the expansion layer can and cannot express

**Observed.** The expansion factor is one scalar per election, indexed on the
predicted third-candidate share:

| election | third share | factor |
| --- | ---: | ---: |
| 2002 | 4.5% | 1.045 |
| 2007 | 16.3% | 1.163 |
| 2012 | 0.0% | **1.000** |
| 2017 | 24.5% | 1.245 |
| 2022 | 2.8% | 1.028 |

**Observed.** The factor each candidate needed, measured as the log-space slope
from the pre-expansion deviation to the realised one:

| | national | needed | applied |
| --- | ---: | ---: | ---: |
| 2017 홍준표 | 27.8% | **~1.30** | 1.245 |
| 2017 문재인 | 47.5% | **~0.95** | 1.245 |
| 2007 정동영 | 29.1% | ~1.19 | 1.163 |
| 2012 문재인 | 48.2% | ~0.85 | **1.000** |
| 2012 박근혜 | 51.8% | ~0.80 | **1.000** |
| 2022 이재명 | 48.4% | ~0.91 | 1.028 |
| 2022 윤석열 | 49.2% | ~0.92 | 1.028 |

Two structural limits follow, and neither is a matter of choosing a better
number:

1. **One factor is shared by candidates that need opposite corrections.** In
   2017 the same `1.245` was applied to a candidate needing `1.30` and one
   needing `0.95`. Because regional shares sum to one, the surplus given to one
   is the deficit of the other — which is the 전남 mirror above.
2. **The layer can only expand.** In 2012 and 2022 every major candidate needed
   a factor **below 1**, and the layer's floor is `1.0`. Three of five elections
   needed the opposite of what it can do.

## The tails need more than the middle — a lead, not a law

**Observed.** Comparing the log-space slope on moderately deviating rows
(`0.15 ≤ |d| < 0.5`) with strongly deviating ones (`|d| ≥ 0.5`):

| | middle | tail | difference |
| --- | ---: | ---: | ---: |
| 2017 홍준표 | 1.172 | **1.342** | **+0.169** |
| 2007 이명박 | 1.041 | 1.288 | +0.247 |
| 2012 문재인 | 0.612 | 1.042 | +0.429 |
| 2022 이재명 | 0.789 | 1.014 | +0.225 |
| 2022 윤석열 | 0.947 | 1.002 | +0.055 |
| 2012 박근혜 | 0.885 | 0.849 | −0.037 |
| 2002 이회창 | 1.015 | 0.949 | −0.066 |

Mean `+0.146`, positive in **5 of 7**.

**Unresolved.** Five of seven is not a demonstration — under a fair coin that
happens about a quarter of the time, and only seven candidate-elections have
enough rows in both bands. The direction is consistent and the magnitude is
about 15% of the slope, but this is a lead for a designed experiment, not an
established property. A constant factor in log space cannot express it; a form
that is convex in the deviation can.

## Two things this is not

**The reconciliation is not eating the expansion.** Measured in percentage
points it looks as though it does — for 홍준표 2017, 대구 keeps only 67.8% of
the points the expansion added while 광주 keeps 112%. That reading is an
artifact of the unit. The expansion acts on log deviations, and a multiplicative
level correction is a translation in log space, so it leaves the shape alone. In
log space the reconciliation removes `+0.0010` of slope on average across the
panel, and for 홍준표 it **adds**, taking the realised factor from `1.255` to
`1.281`.

An earlier draft of this analysis asserted the opposite, in the percentage-point
reading, and it was wrong. It is recorded because the mistake is easy to repeat:
a transform must be measured in the space it operates in.

**The residual is not a function of deviation size alone.** Pooled across
candidates, the correlation between `|log deviation|` and the needed factor is
`+0.048` — apparently nothing. Pooling hides it: each candidate has a different
overall factor, so mixing them averages the within-candidate slope away. The
tail-versus-middle comparison above is the same question asked within
candidates.

## A warning about the estimator

The natural quantity — `needed factor = d_actual / d_predicted` — divides by the
deviation, and explodes wherever a candidate sits near their own national level.
For 홍준표 2017 it returns `+5.16` for 경기, `+6.73` for 인천 and `−4.04` for
충북, all of which have `|d| < 0.03` and carry no information. Two intermediate
conclusions in this investigation were drawn from band statistics contaminated
by exactly those rows and had to be withdrawn. Every number in this document
that describes a slope is an origin regression, which never divides.

## The experiment, and why the line is closed

**Demonstrated.** Three forms were applied to the pre-expansion predictions and
reconciled exactly the way the shipped model reconciles. Each reuses the factor
the model already computes, `c = 1 + predicted_third_share`; none introduces a
new constant, because a constant chosen against this panel would be fitted to
it.

| form | regional weighted | regional equal | concentrated rows |
| --- | ---: | ---: | ---: |
| shipped V32 | 2.5007 | 3.1569 | −1.83 |
| A — current, `d → c·d` | 2.5019 | 3.1552 | −1.82 |
| B — the same `c` in log-odds | 2.5217 | 3.1759 | −1.95 |
| C — `d → c·d + (c−1)·d·|d|` | **2.4918** | **3.1473** | **−0.71** |

Form A reproduces the shipped artifact, which is the check that the offline
reconciliation is faithful. **Log-odds is worse** and is rejected. The convex
form C is better, and it more than halves the bias on the concentrated rows —
the direction the tail measurement predicted.

But the size is the point:

- panel regional macro improves by `0.0089pp` on `2.5007`, about a third of one
  percent;
- 홍준표's 대구 goes `48.35 → 50.13` against an actual `55.25`, closing `1.8` of
  a `6.9` point gap.

**Demonstrated, and this closes the line.** An *oracle* was then run: the
per-candidate factor fitted directly on the realised result — the best any
member of this family could possibly do, and not a proposal, since it reads the
answer.

```
shipped V32                      2.5007pp
oracle per-candidate factor      2.3900pp
the most any such form could win 0.1107pp
```

**The entire terminal-transform family has a ceiling of `0.11pp`, about 4% of
the regional error.** The legitimate parameter-free convex form captures
`0.009pp` of that ceiling. Whatever is producing a `6.9` point miss in 대구 is
not reachable by rescaling deviations after the fact; it is in how the core
model distributes a candidate across regions in the first place.

So the answer to "would a different special function fix this" is **no, and by
a margin large enough not to need a tie-breaker**. The version stays V32.

## What would have to be true for a fix

A per-candidate expansion, or a form convex in the deviation, would address what
is measured here. Neither is proposed as settled, for one reason: the amount
would have to come from somewhere. Deriving it from the needed factors above is
fitting to the scored panel, which is the same objection that ended the two
earlier attempts recorded in `DIAGNOSIS_REGIONALISM_DEAD_ENDS_20260825.md`. A
form whose strength follows from structure rather than from these residuals is
the precondition, and it does not exist yet.

## Related

- `DIAGNOSIS_REGIONALISM_DEAD_ENDS_20260825.md` — two abandoned approaches
- `EXPERIMENT_V31_MULTIPLICATIVE_EXPANSION_20260825.md` — why the form is
  multiplicative
- `PRES_2025_V32_POST_ELECTION_EVALUATION.md` — the same under-statement showing
  up in the 2025 demonstration, scored after the fact
