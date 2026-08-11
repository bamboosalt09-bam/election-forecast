# Incumbent burden and high-shock response

## Decision

The fixed, one-shot experiment was promoted to active policy version
`active_outcome_blind_nested_v4_incumbent_shock_response`.

This is a through-2022 development-selected policy, not an untouched holdout
result. No 2025 outcome is used by the compiler, response, or promotion input.

## Structural diagnosis

- The old terrain layer divided the entire anchor gain by national shock size.
  This was too coarse: a shock can consolidate durable supporters while moving
  critical and swing voters away from the governing camp.
- In 2007, prior direct party ballots showed a weaker liberal base than the
  blended candidate-ballot terrain, but the difference was softened by the
  layer reclassification cap.
- In 2017, the direct impeachment burden reached the conservative candidate,
  but a single linear shock shift was too weak relative to the inherited bloc
  floor.

## Forecast-safe formula

Government burden is compiled only from rows where the automatic issue profile
has an explicit `government` target and a non-zero directional attribution.
The candidate score is an attribution-evidence-weighted mean of:

`direction * association_strength * confidence`

The burden exposure is:

`(1 - max(party_base_resistance, conversion_resistance)) * party_reliability`

- `party_base_resistance` compares the candidate camp's prior direct-party base
  with the strongest prior party base in the same election.
- `conversion_resistance` measures how far the pre-election point forecast
  already exceeds that party base, capped at a 15%p buffer.
- The response never reads actual vote share or contest-vote weights.

For shocks above intensity 1.0, the already explicit direct-mega score receives
an additional `0.40 * score * (intensity - 1)` log-share response. The combined
additional shift is capped at 0.15 and each regional contest is renormalized.

## Fixed promotion test

The coefficients were declared before evaluation:

- government burden gain: `1.0`
- high-shock extra gain: `0.40`
- conversion buffer: `0.15`
- maximum additional log shift: `0.15`

Promotion required both macro metrics to improve, no election to worsen by more
than 0.05%p, no loss of winner accuracy, and improvement in both 2007 and 2017.

| Metric | Previous active | New active | Change |
|---|---:|---:|---:|
| Regional contest-vote weighted macro MAE | 4.9850%p | 4.7248%p | -0.2603%p |
| National candidate macro MAE | 3.8756%p | 3.5408%p | -0.3348%p |
| Winner accuracy | 80% | 80% | unchanged |

National candidate MAE changed by election:

- 2002: unchanged
- 2007: -0.1795%p
- 2012: unchanged
- 2017: -1.4943%p
- 2022: unchanged

The experiment artifacts are in `outputs/incumbent_shock_response/`.
`input_active_v3_predictions.csv` is the immutable pre-promotion input used by
the evaluator, preventing accidental double application after v4 activation.
