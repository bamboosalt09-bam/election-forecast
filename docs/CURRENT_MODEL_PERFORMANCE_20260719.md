# Current Model Performance (2026-07-19)

## Scope and interpretation

This document describes active policy
`active_outcome_blind_nested_v6_cumulative_regime_rejection`.

- 1997 is warmup only.
- 2002-2022 are scored outer folds and the development comparison sample.
- Each target is excluded from its own Ridge fit.
- 2025 outcomes are not used.
- Regional MAE is candidate-region absolute error weighted by observed contest
  votes within each election, then averaged equally across elections.
- National candidate shares are aggregated with observed contest votes. This is
  a post-election diagnostic, not a deployable pre-election aggregation scheme.
- Actual and predicted shares use the same scored two- or three-candidate
  denominator and each regional contest sums to 100%.

## Aggregate metrics

| Metric | Active v5 | Active v6 | Change |
|---|---:|---:|---:|
| Regional weighted macro MAE | 4.3378%p | **4.0522%p** | -0.2856%p |
| National candidate macro MAE | 3.0538%p | **2.6785%p** | -0.3753%p |
| Winner accuracy | 80% | **80%** | unchanged |

The v6 formula was fixed before its corrected single promotion evaluation, but
promotion still used these five historical outcomes. Report this as
development-selected nested performance, not untouched out-of-sample accuracy.

## Election metrics

| Election | Regional weighted MAE | National candidate MAE | Winner correct |
|---|---:|---:|:---:|
| 2002 | 4.1492%p | 3.4106%p | No |
| 2007 | 7.0596%p | 4.9477%p | Yes |
| 2012 | 2.7447%p | 1.0683%p | Yes |
| 2017 | 4.4909%p | 3.0091%p | Yes |
| 2022 | 1.8163%p | 0.9568%p | Yes |

## National predictions

| Election | Candidate | Predicted | Actual | Error |
|---|---|---:|---:|---:|
| 2002 | Roh Moo-hyun | 47.806% | 51.217% | -3.411%p |
| 2002 | Lee Hoi-chang | 52.194% | 48.783% | +3.411%p |
| 2007 | Lee Myung-bak | 46.719% | 54.140% | -7.422%p |
| 2007 | Chung Dong-young | 35.972% | 29.089% | +6.883%p |
| 2007 | Lee Hoi-chang | 17.309% | 16.771% | +0.538%p |
| 2012 | Park Geun-hye | 50.705% | 51.773% | -1.068%p |
| 2012 | Moon Jae-in | 49.295% | 48.227% | +1.068%p |
| 2017 | Moon Jae-in | 44.034% | 47.476% | -3.442%p |
| 2017 | Hong Joon-pyo | 32.287% | 27.773% | +4.514%p |
| 2017 | Ahn Cheol-soo | 23.679% | 24.751% | -1.072%p |
| 2022 | Yoon Suk-yeol | 51.336% | 50.380% | +0.957%p |
| 2022 | Lee Jae-myung | 48.664% | 49.620% | -0.957%p |

## V6 response effect

The government-burden compiler now measures negative evidence share, negative
issue breadth, and continuous rejection strength. The contest gate uses this
signal only when either prior party erosion is not repaired by candidate
conversion or an explicitly attributed rupture shock is present.

This activates through party erosion in 2007 and through rupture evidence in
2017. Low reliability protects 2002; lack of a qualifying route protects 2012
and 2022. The adjustment does not route votes to a named opponent, and it
preserves third-candidate shares exactly. See
`docs/CUMULATIVE_REGIME_REJECTION_20260719.md`.

## Verification

- full test suite: `355 passed`
- corrected frozen-v5 experiment reproduces the active v6 metrics exactly
- strict PIT audit: PASS (`24` manifest files, `1,685` manifest rows,
  `215` outcome-invariance rows)
- through-2022 selection-boundary audit: PASS (`70,874` active CSV rows)
- every fold excludes its target, uses the consistent scored denominator, and
  excludes realized-slot predictors
- regional share-sum maximum error: `2.22e-16`

The standalone slot audit still fails its older frozen-reproduction equality
guard at `0.0004818253`. The active fold audit independently verifies that old
realized-slot predictors are absent. Treat the standalone audit as unresolved
compatibility work, not as a passing check.
