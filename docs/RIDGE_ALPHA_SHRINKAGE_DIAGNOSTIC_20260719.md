# Ridge Alpha Shrinkage Diagnostic (2026-07-19)

## Question

Does Ridge shrinkage cause the active v6 model to understate the winning
margins in 2007 and 2017?

## Design

The active strict-nested alpha schedule is fold-specific:

```text
2002=0.3, 2007=0.3, 2012=0.8, 2017=1.2, 2022=1.2
```

Only this schedule was multiplied by
`0, 0.01, 0.03, 0.10, 0.25, 0.50, 1, 2, 4, 8`. Every candidate assignment,
predictor, PIT cutoff, structural layer, issue layer, and v6 postprocess stayed
fixed. Multiplier `1` reproduced the active predictions with maximum difference
`1.11e-16`. Multiplier `0` failed because the design is singular, so the
smallest successful near-unregularized run was `0.01`.

This is a diagnostic development grid. It must not select a new production
alpha because the same 2002-2022 outcomes are inspected across settings.

## Results

| Multiplier | Regional MAE | National MAE | All-margin bias | 2007 margin error | 2017 margin error | Winner accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 5.3323 | 2.8433 | -5.0754 | -14.5157 | -8.4246 | 80% |
| 0.10 | 4.7619 | 3.7662 | -7.8368 | -14.4971 | -8.3735 | 60% |
| 0.25 | 4.2290 | 2.8742 | -4.9040 | -14.4683 | -8.2924 | 80% |
| 0.50 | 4.3964 | 3.3327 | -7.0705 | -14.4165 | -8.1683 | 60% |
| **1.00 active** | **4.0522** | **2.6785** | **-5.8608** | **-14.3047** | **-7.9553** | **80%** |
| 2.00 | 4.0261 | 2.5715 | -5.7809 | -14.1497 | -7.6360 | 80% |
| 4.00 | 3.9709 | 2.5359 | -5.7687 | -13.9202 | -7.2493 | 80% |
| 8.00 | 3.9490 | 2.6430 | -5.7587 | -13.7001 | -6.8929 | 80% |

All values other than accuracy are percentage points. The 2007 and 2017
columns compare the predicted margin of the actual winner with the actual
margin. Negative values mean the model compresses the landslide.

The mean standardized coefficient L2 norm falls from `0.5643` at multiplier
`0.01` to `0.2688` at the active setting and `0.2095` at multiplier `8`,
confirming that the experiment actually changes shrinkage strength.

## Conclusion

The shrinkage-cause hypothesis is rejected for this pipeline. Removing most
regularization makes both target margin errors slightly more negative. Stronger
regularization improves the target margins, but even multiplier `8` leaves
errors of `-13.70%p` in 2007 and `-6.89%p` in 2017. The all-election margin
bias changes by only `0.10%p` from active `1` to multiplier `8`.

The dominant source of landslide compression therefore lies downstream or in
the information structure: contest normalization, core/flexible-pool bounds,
candidate/party strength priors, and regime-response magnitude are more likely
than Ridge coefficient shrinkage. The active alpha schedule remains unchanged.

## Verification

- successful schedules: 9; zero-alpha failure: singular design
- strict outer folds checked: 45
- target excluded and scored denominator consistent: PASS
- realized-slot and neutral-direction leakage: absent
- post-2022 target rows: none
- compositional sum maximum error: `2.22e-16`

Raw outputs are under `outputs/ridge_alpha_nested_v6/`.

