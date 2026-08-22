# Party-regionalism retention is not the missing correction

## Status

- Date: 2026-08-22
- Status: measured and rejected; V26 unchanged
- Post-2022 outcomes used: none

## Question

The model measures inherited party regionalism in `recent_bloc_base`, but the
fitted prediction can move far away from it.  This experiment asked whether a
minimum retained regional contrast could repair that gap without changing
candidate national size.

For each candidate, prior and prediction were centred around their own
vote-weighted national logit.  A floor was applied only where both contrasts
had the same sign:

    abs(final contrast) >= gain * reliability * abs(prior contrast)

Opposite-signed realignments were deliberately left untouched.  A second
version multiplied the floor by the model's predicted third-candidate national
share.  Iterative calibration restored every regional sum and each candidate's
original national share after either transform.

## Results

The unconditional floor failed monotonically:

| gain | regional macro | national macro | bound cells | over 10pp |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2.7122 | 0.7210 | 0 | 11 |
| 0.25 | 2.7252 | 0.7210 | 7 | 11 |
| 0.50 | 2.7392 | 0.7210 | 19 | 11 |
| 0.75 | 2.7558 | 0.7210 | 45 | 11 |
| 1.00 | 2.7343 | 0.7210 | 77 | 11 |

Third-candidate activation scoped the damage but did not identify the needed
cells:

| gain | regional macro | national macro | bound cells | over 10pp |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.7122 | 0.7210 | 0 | 11 |
| 1 | 2.7128 | 0.7210 | 2 | 11 |
| 2 | 2.7141 | 0.7210 | 10 | 11 |
| 3 | 2.7104 | 0.7210 | 26 | 11 |
| 4 | 2.6915 | 0.7210 | 38 | 11 |

At gain 4, 2017 improved from 3.025 to 2.910, but 2007 did not move and 2022
worsened.  Gain 5 improved the aggregate to 2.6696, while increasing cells over
10pp from 11 to 12.  Larger values finally moved 2007 but rapidly broke 2017.
There is no stable interval that repairs both stronghold failures.

Applied after the previously measured deviation expansion, every non-zero
retention gain worsened its 2.5829 regional macro.  The two mechanisms are not
complementary.

## Interpretation

`recent_bloc_base` describes regional shape, but it does not identify how much
of that shape survives candidate weakness, party splits and regime shocks.
Reliability and third-candidate size do not supply the missing survival
elasticity.  Treating the prior as a floor therefore hardens stale regionalism
in elections that do not need it.

The experiment rejects a direct prior-retention correction.  It strengthens
the case for keeping regional party identity and movable-mass elasticity as
separate quantities.  V26 remains unchanged; the symmetric deviation expansion
remains a downstream candidate, not a promoted model.
