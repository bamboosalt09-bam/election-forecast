# Cumulative Regime Rejection (2026-07-19)

## Purpose

The v6 layer addresses elections where a governing camp is weakened by several
coherent negative issues even when no literal `regime_change` keyword is strong
enough. It applies the same evidence principle to 2007 and 2017 without adding
an election-specific candidate correction.

## Outcome-blind signal

For each explicitly government-targeted candidate row:

```text
rejection_strength
  = max(-government_direction, 0)
  * negative_evidence_share
  * sqrt(min(unique_negative_issues / 4, 1))
```

The signal is then activated through the stronger of two forecast-time routes:

- party-erosion route: the governing candidate's prior direct-party base is
  below the strongest camp and forecast conversion does not already repair it;
- rupture route: mega-issue intensity exceeds 1.0 and has explicit negative
  attribution to the governing camp.

The route-adjusted signal is multiplied by direct-party reliability. It enters
the existing contest-regime score and directional advantage. The final response
still reallocates only the dominant/runner flexible pool above conservative
core floors. Slot C is unchanged exactly.

Fixed constants are breadth reference `4`, party-erosion width `0.08`,
conversion buffer `0.15`, rupture score reference `0.25`, expansion gain
`0.50`, and final log-shift cap `0.40`.

## Promotion result

The corrected evaluator first inverts the recorded v5 contest shift and then
applies v6 once. Applying v6 directly on final v5 predictions would double
count the old regime response and is prohibited by a regression test.

| Metric | Active v5 | Active v6 | Change |
|---|---:|---:|---:|
| Regional weighted macro MAE | 4.3378%p | 4.0522%p | -0.2856%p |
| National candidate macro MAE | 3.0538%p | 2.6785%p | -0.3753%p |
| Winner accuracy | 80% | 80% | unchanged |

National candidate MAE changes are `-1.4047%p` in 2007 and `-0.4720%p` in
2017. Predictions for 2002, 2012, and 2022 are unchanged. Slot-C predictions
in 2007 and 2017 are also unchanged.

This is development-selected evidence from the 2002-2022 sample, not an
untouched holdout result. No 2025 outcome is read or compared.

