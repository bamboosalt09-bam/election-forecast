# Electorate Layer Model

## Purpose

The layer is a constrained ecological decomposition of regional voting mass. It does not
claim that aggregate election results identify individual voters. It separates durable core,
critical support, and swing support so that the same issue signal can have different response
elasticities.

## Point-in-time history

For every target election, historical rows must satisfy:

```text
source_election_date < target_election_date
```

The audit artifact is `outputs/electorate_layer_experiment/history_source_audit.csv`. For the
2002 target, the eligible sources are the 1992 and 1997 presidential elections; the 1992,
1996, and 2000 Assembly district elections; and the June 2002 local election. All are earlier
than the December 2002 presidential election. The current history has no 1995 or 1998 local
rows, so the model does not claim to use them.

## Two-channel mass estimation

Direct party ballots and candidate ballots are no longer pooled into one quantile.

Direct-party channel:

- Assembly proportional vote
- metropolitan council proportional vote
- local council proportional vote

Candidate-ballot channel:

- presidential vote, including 1997 warmup evidence
- Assembly district vote
- council district vote
- governor and mayor vote

Education elections have zero weight. Direct-party ballots estimate the durable party floor.
Candidate ballots are a fallback and stabilizer when direct-party history is short.

For each region and bloc:

```text
direct_core = max(direct weighted Q25,
                  direct mean - 1.5 * weighted MAD / sqrt(effective N))
candidate_core = candidate weighted Q25
direct_reliability = direct effective N / (direct effective N + 2)

core_raw = reliability * direct_core
         + (1 - reliability) * candidate_core

recent_base = reliability * direct_mean
            + (1 - reliability) * candidate_mean

critical_raw = max(recent_base - core_raw, 0) * volatility persistence
swing_mass = 1 - sum(candidate core and critical mass)
```

The lower-confidence term can raise the direct-party floor only when repeated elections make
it statistically stable. A single direct-party election cannot promote its mean into core
support. Independent core is zero; third-bloc core is conservatively shrunk.

## Issue signal correction

The previous implementation divided candidate preference signals by each election's maximum
preference magnitude. That forced the strongest contrast in every election to one. A tiny
candidate-tone difference in 2012 therefore became as strong as the much larger 2022
difference, erasing the confidence and magnitude information computed earlier.

The corrected signal is divided by issue-attention scale instead:

```text
preference_component = attention
                     * centered candidate stance
                     * stance confidence

issue_preference_signal = centered preference component
                        / election issue-attention scale
```

This preserves relative issue salience while retaining the absolute candidate-tone gap and
confidence. Neutral and informational sentences increase attention but do not receive an
arbitrary vote direction.

## Response

The deterministic engine prediction is the baseline. Core sensitivity is lower than critical
support sensitivity, and critical support sensitivity is generally lower than swing
sensitivity.

```text
susceptibility[k] =
    core_mass * core_sensitivity[k]
  + critical_mass * critical_sensitivity[k]
  + swing_mass * swing_sensitivity[k]

log_shift = preference_gain
          * issue_preference_signal[k]
          * susceptibility[k] / baseline_share

prediction = normalize(baseline * exp(sum(log_shift)))
```

### Active direct-party separation

The active mass profile is `direct_party_layers`. It uses `2,602` direct-party rows from
Assembly proportional, metropolitan-council proportional, and local-council proportional
elections. These ballots define party attachment independently from candidate ballots.

```text
party base = weighted direct-party vote
durable core = lower direct-party floor, conservatively stabilized by candidate history
critical support = party base above the durable floor
personal/swing evidence = candidate-ballot mean - direct-party mean
```

Candidate popularity can no longer raise the estimated party base. A single update may move
at most `2.5%p` in the raw core or critical assignment; after regional compositional
normalization the observed maxima are `2.70%p` for core and `2.50%p` for critical support,
both below the final `3%p` policy cap.

### Layer-separation audit (2026-07-17)

The response was further decomposed into three explicit paths:

- core rigidity: reduce issue elasticity of durable support
- critical defection: increase only the response to negative candidate/party issue signals
- swing mobility: increase symmetric movement of the residual swing mass

The paths have separate diagnostics (`core_preference_log_shift`,
`critical_preference_log_shift`, and `swing_preference_log_shift`) and are available through
one bounded `layer_separation` parameter plus a declared response profile. The active config
keeps `layer_separation=0`; therefore this implementation does not change the released point
predictions unless a profile passes nested transfer checks.

The primary `critical_defection` profile improved every applicable outer fold but only by
`0.000556%p` in equal-election nested macro MAE (`4.613552 -> 4.612996`). Core rigidity was
not selected, swing mobility worsened the macro, and the original combined response worsened
it by `0.003033%p`. These effects are too small for adoption. Reproducible artifacts are in
`outputs/electorate_layer_profile_experiment`.

The direct-party separation changed the strict nested macro from legacy `4.613552%p` to
`4.613162%p`. The improvement is too small to claim an accuracy gain, but it passed the
non-inferiority, maximum-election-worsening, point-in-time, and final-reclassification gates.
It was therefore adopted as a structural interpretation correction. More aggressive
10th-percentile and broad-critical alternatives reclassified as much as `13-32%p` and remain
rejected under `outputs/electorate_mass_profile_experiment`.

The capped strict nested selector chooses an active preference gain of `0.04`. Terrain
anchoring, turnout, and nonvoter gains are zero. The historical fixed `0.04` post-hoc
experiment remains in `data/config/electorate_layers_fixed_experiment.json`; its numerical
gain now matches the active model, but its evaluation protocol does not. Official
regional turnout history still has zero rows, so no turnout effect is claimed.

## Evaluation

Two evaluations are reported separately:

1. Capped strict nested learner: gains are selected only from earlier scored folds on a
   declared `0.00..0.04` grid. It uses zero for 2002/2007 and `0.04` for 2012 onward.
   Weighted macro MAE is `4.613162%p`.
2. Fixed structural experiment: the same weak `0.04` gain is applied to every frozen outer
   prediction after the fact. Weighted macro MAE is `4.593551%p`, a `0.034217%p` improvement
   over baseline, but this is not a strict nested selection estimate.

The non-active fixed experiment by election is:

| Election | Baseline | Fixed layer | Change |
|---|---:|---:|---:|
| 2002 | 3.5712 | 3.5195 | +0.0517 |
| 2007 | 6.3463 | 6.2975 | +0.0488 |
| 2012 | 6.2182 | 6.1941 | +0.0240 |
| 2017 | 5.3232 | 5.2645 | +0.0587 |
| 2022 | 1.6800 | 1.6922 | -0.0122 |

In the active capped model, the 2022 weighted regional degradation is only `0.0123%p`, while
the national candidate error improves from `0.180%p` to `0.138%p`. Applying a similar
positive shift across regions still hurts places where the baseline was already close. This
must not be presented as strict nested or untouched out-of-sample evidence.

### Uncapped nested gain experiment

`scripts/evaluate_nested_electorate_learning.py --uncapped` removes the `0.04` search ceiling.
It starts at zero, expands the range geometrically while the optimum remains on the boundary,
and uses only earlier frozen outer-fold elections for each target. The learned target gains
were `0.88` (2012), `1.20` (2017), and `0.72` (2022). Strict weighted macro MAE worsened from
`4.627768%p` to `5.470805%p`, mainly because the 2022 target worsened by `3.926472%p`.
All adoption gates therefore failed and these gains are not part of the active engine. The
rejected outputs are in `outputs/electorate_nested_learning_uncapped`.

## Limits

- These are latent ecological masses, not survey-observed voter identities.
- Candidate stance is not fully issue-target-specific for every sentence.
- The capped learner's improvement is small and based on only three non-warmup selection folds.
- The critical-support mass is not directly observed. Broader definitions improve 2012/2017
  but currently transfer poorly to 2022, so they are not active.
- The fixed post-hoc result must not be substituted for the capped nested estimate.
- 1995 and 1998 local election rows are absent from the current history.
- Turnout and nonvoter channels remain inactive.
- 2022 has been used for this implementation audit and is not an untouched holdout.
