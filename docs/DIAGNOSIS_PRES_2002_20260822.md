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

## The fold that predicts 2002 trains on one election

This is the finding, and it is definitional rather than statistical.

| target | training elections | count | max VIF | regional | national | winner | share of national macro |
| --- | --- | ---: | ---: | ---: | ---: | :-: | ---: |
| **pres_2002** | pres_1997 | **1** | **1.0000** | 2.7519 | **2.3416** | ✗ | **65 %** |
| pres_2007 | +pres_2002 | 2 | 1.0068 | 4.2718 | 0.6610 | ✓ | 18 % |
| pres_2012 | +pres_2007 | 3 | 15.0221 | 2.3782 | 0.1271 | ✓ | 4 % |
| pres_2017 | +pres_2012 | 4 | 13.8326 | 3.0251 | 0.2011 | ✓ | 6 % |
| pres_2022 | +pres_2017 | 5 | 11.0600 | 1.1341 | 0.2741 | ✓ | 8 % |

Six predictors are fitted on a single warmup election. The maximum predictor
VIF of exactly 1.0000 is the signature: with one election there is no
cross-election variation for the design to be collinear in. Collinearity only
appears from the third fold, where VIF reaches 15.

Two consequences.

**The error is a level error, not a shape error.** 2002's regional weighted MAE
of 2.7519 is mid-panel - 2007 is worse at 4.2718 - while its national MAE of
2.3416 is the worst by more than three times. The model gets the pattern across
regions roughly right and the balance between the two majors wrong, which is
what a fold with almost no fitted level information would be expected to do.

**One fold carries 65 % of the national macro.** The headline 0.7210 is
0.4683 from 2002 and 0.2527 from everything else. An equal-election macro
averages a fold trained on one election with a fold trained on five, which is
closer to averaging the first point of a learning curve with its steady state
than to averaging five comparable measurements.

Training depth correlates with national MAE at r = -0.782 across the five
folds. At n = 5 that is a pattern consistent with the mechanism, **not
evidence**, and it should not be quoted as a result.

## What this rules out as a fix

Nothing here is tunable. The shallowest fold is what strict chronological
nesting is for: 2002 is the first scored election, so it can only be predicted
from what precedes it. Deepening it would mean adding earlier elections to the
panel or relaxing the chronological rule, and the second would destroy the
property the design exists to guarantee.

The useful response is reporting rather than repair. Wherever the macro is
quoted, the per-fold training depth belongs beside it, so a reader can see that
the first fold is not a comparable measurement. `scripts/diagnose_fold_training_depth.py`
produces the table above.

## Next step

A predictor-level attribution is now much less interesting than it looked. With
one training election the fitted coefficients are barely identified, so asking
which predictor drove the reversal would be reading structure into a fit that
has almost none. The question was well posed for a fold with real training
behind it; for this one it is close to meaningless.

What remains worth doing, in order of value and in decreasing order of safety:

1. **Report the per-fold training depth beside the macro.** Costs nothing,
   changes nothing, and stops the headline from implying five comparable
   measurements.
2. **Report the macro with and without the shallowest fold** - not to claim the
   better number, but because 0.7210 and 0.3158 describe different things and
   quoting only the first hides that one fold is 65 % of it.
3. Leave 2002 alone. Every structural lever is inert on it, the base stage
   cannot be given information it does not have, and the one change that would
   help - more elections before 2002 - is not available.

2002 is also the only election where two postprocess layers meet, so its
residual is entangled with the single layer interaction this panel can observe.
That is a reason to be careful with it, not a reason to work on it.
