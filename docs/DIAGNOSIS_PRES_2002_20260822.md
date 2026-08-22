# Where the 2002 error actually is

## Status

- Date: 2026-08-22
- Status: diagnosis only; **no change made**
- Post-2022 outcomes used: none beyond the scored panel this already evaluates

2002 has been described in the V26 record as "the worst election in the panel"
and "the first place to look next". Both are true and both understate it.

## It is the only winner miss

| candidate | actual | V26 | error |
| --- | ---: | ---: | ---: |
| 노무현 | 49.21 | 45.70 | **-3.51** |
| 이회창 | 46.87 | 49.86 | **+2.99** |
| 권영길 | 3.92 | 4.45 | +0.53 |

The model puts 이회창 ahead by 4.16 points. The published 4/5 winner accuracy
is this one election, so 2002 is not merely the largest MAE - it is the entire
winner error.

## The error is in the base stage, not the postprocess stack

| stage | 노무현 | 이회창 | gap |
| --- | ---: | ---: | ---: |
| `shadow_pred` (earliest) | 40.00 | 43.66 | -3.66 |
| `pre_hierarchy_pred` | 35.64 | 38.81 | -3.17 |
| `terrain_pred` | 37.35 | 54.46 | -17.11 |
| `anchored_pred` | 39.55 | 45.50 | -5.95 |
| `layer_pred` (final) | 45.70 | 49.86 | -4.16 |
| **actual** | **49.21** | **46.87** | **+2.34** |

The reversal is present in the first modelled stage and never recovers. The
three structural postprocesses move the two-major gap from -3.66 to -4.16, so
the stack makes 2002 slightly **worse**, not better.

This settles a question the V26 and ablation records left open. Tuning the
veto, the ceiling or the refusal cannot fix 2002 - none of them addresses the
two majors here, and 2002's mega intensity is 0.6837, still below the
activation gate, so the graded-intensity work does not reach it either. Every
structural lever worked on so far is inert on the one election that decides the
winner metric.

## The preliminary stage gets the direction right

`candidate_slot_assignments_v2` puts 노무현 in slot A with a preliminary
expected share of 0.5724 against 이회창's 0.4276. The two-way direction is
correct - it overshoots 노무현, whose realised two-way share was 0.512, but it
has the right candidate ahead. **The Ridge stage then reverses it.**

## The candidate-prior exponent is not the cause

`CANDIDATE_STRENGTH_EXPONENT = 2.0` squares the candidate weight, which
amplifies differences between candidates. It is a reasonable thing to suspect,
and it is not what is happening here:

| election | slot A weight | slot B weight | squared ratio |
| --- | ---: | ---: | ---: |
| **pres_2002** | 0.6237 | 0.6055 | **1.0609** |
| pres_2007 | 0.6040 | 0.5737 | 1.1085 |
| pres_2012 | 0.5914 | 0.5413 | 1.1938 |
| pres_2017 | 0.6640 | 0.5811 | 1.3060 |
| pres_2022 | 0.5773 | 0.5364 | 1.1581 |

2002 has the *smallest* squared ratio of the five. The two candidates are
nearly equal in structural weight, so the exponent barely separates them there.
If the exponent deserves a sensitivity test - and it does - 2017 is where it
bites, not 2002.

## What is not established

Which predictor drives the reversal. Of the six slot-free predictors, only two
are carried into `nested_predictions.csv`:

    landscape_bloc_alignment   노무현 0.0529   이회창 0.0834   diff -0.0305
    landscape_centrist         노무현 0.0773   이회창 0.0774   diff -0.0001

`landscape_bloc_alignment` does favour 이회창, but `issue_advantage`, `rif`,
`partisan_prior` and `landscape_inferred_prior` are not in the output frame, so
this is a partial view and must not be quoted as the answer.

## Next step

A predictor-level attribution for 2002 needs the Ridge design matrix and
coefficients at the fold that predicts it, which means instrumenting the base
stage rather than reading its outputs. That is the work this diagnosis points
to, and it is a different kind of change from everything done so far: the
structural layers are bolt-on transforms, while this reaches the fitted stage.

Two cautions for whoever does it. 2002 is one election, and a change to the
base stage touches all five, so the risk profile is the opposite of the
structural work - broad rather than narrow. And 2002 is also the only election
where two postprocess layers meet, so its residual is entangled with the one
interaction the panel can observe at all.
