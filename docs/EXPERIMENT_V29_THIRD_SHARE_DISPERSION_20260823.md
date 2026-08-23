# V29: expanding the dispersion a third candidate compresses

## Status

- Date: 2026-08-23
- Status: **promoted**; predecessor V28 frozen and unchanged
- Gain: 1.0, the parameter-free value - not swept
- Post-2022 outcomes used: none

## Why this sat unpromoted

The mechanism is not new. It is mechanism 1 of
[EXPERIMENT_DISPERSION_ALTERNATIVES_20260822](EXPERIMENT_DISPERSION_ALTERNATIVES_20260822.md),
which concluded that **all four** dispersion corrections degrade the national
metric, and adopted none.

Two separate faults kept it there, and each hid the other.

**The evidence had expired.** That conclusion was measured against V26 and
never re-measured. V27 then added core-weighted inherited regional dispersion
and V28 froze it, while the diagnostic that produced the conclusion kept
reading `outputs/active_presidential_nested_v26` from a hardcoded path. Making
the diagnostics follow the active pointer (PR #21) is what surfaced this; it is
the first case where doing so reversed a conclusion rather than refreshing a
number.

**And the measured cost was an artifact.** The evaluation harness held its own
copy of the transform, which clipped negative shares at a floor and then
renormalised. Clipping injects vote mass, so the candidate national levels
moved and the national metric degraded - by 0.0066pp at gain 1.0. That was read
as an intrinsic tension between regional shape and national level, and it is
the finding the earlier record generalised from. It was the floor.

With the boundary handled properly the national macro is `0.726249712` at
**every** gain from 0.00 to 2.00, identical to nine decimals. There was no
tension to trade against.

The harness no longer keeps its own copy; the sweep now calls the shipped
transform, so a measurement cannot again describe a mechanism the model does
not apply.

## The mechanism

Each candidate's regional deviations are expanded around that candidate's own
vote-weighted national level, and each region is then renormalised:

    scaled = level + (1 + gain * predicted_third_share) * (pred - level)

Three properties follow from the form rather than from tuning.

**It reads no outcome.** The index is the model's own predicted third-placed
national level, available at forecast time.

**It conserves the national level.** Candidate levels sum to one in every
region, so a factor applied uniformly leaves each region summing to one exactly
and the renormalisation is a no-op. Measured, the largest candidate-level shift
across the five elections is `5.6e-15` percentage points - machine epsilon.

**And it scopes itself.** The quantity that diagnoses the compression is the
quantity that sizes the correction, so no separate index decides where to act:

| election | predicted third share | factor | applied |
| --- | ---: | ---: | ---: |
| pres_2002 | 0.0445 | 1.0445 | 1.0445 |
| pres_2007 | 0.1646 | 1.1646 | 1.1646 |
| pres_2012 | **0.0000** | **1.0000** | **1.0000** |
| pres_2017 | 0.2461 | 1.2461 | **1.1474** |
| pres_2022 | 0.0282 | 1.0282 | 1.0282 |

2012 has no third candidate and is left exactly where it was.

### The feasibility cap

2017 is capped below its nominal factor. At 1.2461 the expansion drives
홍준표's 광주 and 전남 predictions below zero, and a negative share is not a
dispersion. The expansion stops at the largest factor the election admits.

The cap is per election, not per candidate. That is structural: the level
conservation above holds because one factor applies to everyone, so the
regional sums stay at one. Capping only the candidate who needs it would give
two candidates different factors, the regional sums would drift, the
renormalisation would stop being neutral, and the levels would move - which is
the failure the cap exists to prevent. Measured both ways: per-candidate
capping leaves a 0.124pp level shift in 2017, uniform capping leaves 5.6e-15.

The cap introduces no constant. It is where the transform's own definition runs
out.

## Why gain 1.0 and not the gain that scores best

The sweep against V28, with the corrected transform:

| gain | regional macro | national macro | dispersion ratio | winners |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 (V28) | 2.638411 | 0.726250 | 0.9505 | 4/5 |
| 0.25 | 2.584351 | 0.726250 | 0.9705 | 4/5 |
| 0.50 | **2.555129** | 0.726250 | 0.9906 | 4/5 |
| 0.75 | 2.555757 | 0.726250 | 1.0044 | 4/5 |
| **1.00** | 2.573607 | 0.726250 | 1.0141 | 4/5 |
| 1.50 | 2.624770 | 0.726250 | 1.0336 | 4/5 |
| 2.00 | 2.725671 | 0.726250 | 1.0531 | 4/5 |

A gain of 0.50 gives a better regional figure than the promoted value, by
0.018pp. It is not promoted, for the only reason that matters here: 0.50 is a
constant chosen by sweeping the same five outcomes it is then scored against,
and its advantage is smaller than the selection freedom used to find it.

At a gain of exactly 1 the expansion factor is `1 + predicted_third_share`.
There is no constant, so there is nothing for the panel to select. The
functional form remains a choice - linear in the third share - but a form
argued from the diagnosis is a weaker commitment than a magnitude fitted to
five outcomes.

Since the national metric is flat across the whole sweep, this choice costs
nothing on that axis. It costs 0.018pp of regional accuracy relative to the
best in-sample gain, and buys a parameter that was never fitted.

## Result

| metric | V28 | V29 |
| --- | ---: | ---: |
| regional equal-election macro MAE | 2.638411 %p | **2.573607 %p** |
| national equal-election macro MAE | 0.726250 %p | **0.726250 %p** |
| winner accuracy | 0.8 | 0.8 |
| prediction rows | 232 | 232 |

By election, regional weighted MAE:

| election | V28 | V29 |
| --- | ---: | ---: |
| pres_2002 | 2.795 | 2.957 |
| pres_2007 | 4.044 | 3.730 |
| pres_2012 | 2.387 | **2.387** |
| pres_2017 | 2.796 | 2.607 |
| pres_2022 | 1.171 | 1.187 |

The two compressed elections improve by 0.31 and 0.19 percentage points. 2012
is untouched. 2002 and 2022 pay 0.16 and 0.02.

## What this does not establish

The gap being closed was found by reading the residuals of these same five
outcomes, so the *decision to build a dispersion correction at all* is
in-sample even though the gain is not. Only a pre-registered application to an
unscored election would test that, and none is available.

2002 worsens, and it is the fold least able to afford it: one training
election, VIF 1.0000, and the largest share of the national macro. Expanding
its deviations expands a level error. The transform cannot tell the difference
between compressed dispersion and a mis-levelled fold, and 2002 is the second.
