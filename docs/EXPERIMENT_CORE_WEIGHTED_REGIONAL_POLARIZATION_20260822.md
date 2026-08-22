# Core-weighted regional polarization

## Status

- Date: 2026-08-22
- Status: successful development-panel experiment; not yet promoted
- Active V26 and all frozen outputs remain unchanged
- Post-2022 outcomes used: none

## Mechanism

A regional share floor failed because inherited party support is not an
unchanging minimum.  The successful formulation instead treats regionalism as
the amount of **regional dispersion retained by concrete supporters**.

For each candidate, the model compares the vote-weighted standard deviation of
regional logits in the fitted prediction and the point-in-time
`recent_bloc_base`.  It expands only a missing dispersion gap:

    factor = 1 + gain * core_mass * direct_party_reliability
                   * max(0, prior_sd / fitted_sd - 1)

The candidate's fitted regional ordering is retained.  The prior supplies only
the inherited width, not regional vote floors or target shares.  Iterative
calibration then restores every candidate's original vote-weighted national
share and every region's unit sum.

At gain 1, the formula has a direct interpretation: the evidenced concrete
mass preserves its proportional part of the missing inherited dispersion.
There is no new threshold, floor or cap.

## Sensitivity

| gain | regional macro | national macro | over 10pp cells |
| ---: | ---: | ---: | ---: |
| 0.00 | 2.7122 | 0.7210 | 11 |
| 0.25 | 2.6862 | 0.7210 | 11 |
| 0.50 | 2.6609 | 0.7210 | 11 |
| 0.75 | 2.6370 | 0.7210 | 11 |
| **1.00** | **2.6139** | **0.7210** | **10** |
| 1.50 | 2.5720 | 0.7210 | 9 |
| 2.00 | 2.5402 | 0.7210 | 9 |
| 2.50 | 2.5168 | 0.7210 | 7 |
| 3.00 | 2.5037 | 0.7210 | 6 |
| 4.00 | 2.5220 | 0.7210 | 5 |

Gain 3 is the panel minimum, but it is not a valid promotion choice: its value
would be selected from the same five outcomes used for evaluation.  Gain 1 is
the only pre-specifiable candidate because it means one-for-one preservation
by the measured concrete and reliable share.

At gain 1 the election-level regional MAEs are:

| election | V26 | core-weighted polarization |
| --- | ---: | ---: |
| 2002 | 2.752 | 2.752 |
| 2007 | 4.272 | 4.039 |
| 2012 | 2.378 | 2.378 |
| 2017 | 3.025 | 2.782 |
| 2022 | 1.134 | 1.118 |

This is the first regionalism mechanism tested in this sequence that improves
every election it materially changes while leaving both national calibration
and winner calls unchanged.

## Interpretation and limit

The result supports the conceptual split:

- `recent_bloc_base`: inherited regional width
- `core_voting_mass`: the share capable of retaining that width
- `direct_party_reliability`: confidence that the inherited party channel
  belongs to the current candidate
- fitted prediction: current regional ordering and national candidate size

The experiment is still development evidence, not independent validation.  It
was motivated by residual inspection, and gain 1 must be declared before any
future election is scored.  Promotion should place this transform before
shock-driven core erosion so that the latter can explicitly release part of
the preserved concrete mass rather than operating on an already flattened
map.
