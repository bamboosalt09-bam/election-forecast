# Electorate Layer Separation Experiment (2026-07-17)

## Question

Can the existing ecological electorate layer distinguish durable core, critical support, and
swing voters more explicitly without using the target election outcome to tune its own
response?

## Protocol

- scored elections: 2002, 2007, 2012, 2017, 2022
- post-2022 presidential outcomes: not loaded
- outer target excluded from both preference-gain and separation selection
- regional metric: contest-vote weighted candidate-region MAE within election
- macro metric: equal-election mean
- separation grid: `0.00, 0.25, 0.50, 0.75, 1.00`
- adoption threshold: at least `0.01%p` strict nested improvement, no election worsening over
  `0.05%p`, and at least two nonzero outer selections

## Response ablation

| Profile | Nested MAE | Change vs active | Result |
|---|---:|---:|---|
| Critical defection (primary) | 4.612996 | +0.000556 | Reject: too small |
| Core rigidity | 4.613552 | +0.000000 | Reject |
| Swing mobility | 4.613903 | -0.000351 | Reject |
| Critical + swing | 4.613428 | +0.000125 | Reject: too small |
| Combined | 4.616585 | -0.003033 | Reject |

`critical_defection` now changes only negative responses. Positive critical-support responses
remain at the baseline elasticity. It improved 2012, 2017, and 2022 without worsening an
applicable fold, but the effect was practically negligible.

Artifacts: `outputs/electorate_layer_profile_experiment`.

## Mass-definition ablation

The legacy estimator produced an average critical-support mass of about `2.8%` per
candidate-region row. The follow-up primary profile separates direct party ballots from
candidate ballots and caps final layer reclassification at `3%p`.

| Mass profile | Nested MAE | Change vs active | Largest worsening | Result |
|---|---:|---:|---:|---|
| Direct-party layers (active) | 4.613162 | baseline | 0.000000 | Adopt structurally |
| Legacy | 4.613552 | -0.000390 | 0.002120 | Superseded |
| Durable floor | 4.610381 | +0.002782 | 0.007205 | Reject: layer shift over cap |
| Broad critical | 4.612164 | +0.000998 | 0.000436 | Reject: layer shift over cap |
| Durable floor + broad critical | 4.608907 | +0.004255 | 0.006092 | Reject: layer shift over cap |

The direct-party profile changes candidate-region core mass by at most `2.70%p` and critical
mass by at most `2.50%p`. The aggressive alternatives moved layers by `13-32%p`, violating
the slow-transition assumption even when their aggregate MAE was lower.

Artifacts: `outputs/electorate_mass_profile_experiment`.

## Decision

The active model now uses the bounded `direct_party_layers` mass estimator with
`layer_separation=0`. Direct party votes identify party attachment; candidate-ballot excess
remains personal/swing evidence. Additional critical-defection response is still inactive
because its transfer gain is negligible and it slightly worsens 2017 under the new masses.
