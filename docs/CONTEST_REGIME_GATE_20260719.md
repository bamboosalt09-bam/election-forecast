# Contest Regime Gate (2026-07-19)

## Purpose

The previous engine classified core, critical-support, and swing masses but did
not decide whether the contest itself was close, asymmetric, or rupture-driven.
That omission compressed the 2007 and 2017 margins toward the center.

Active policy v5 adds an outcome-blind contest-regime gate after the v4
incumbent-shock response.

## Conservative core

The gate does not treat the full latent core estimate as immovable. Its floor is:

`min(core_voting_mass_effective, direct_party_core_raw) * direct_party_reliability`

This is always no greater than the existing core estimate and becomes small
when prior direct-party evidence is weak. The floor is not moved by the regime
response.

## Regime decision

Candidates receive a structural score from prior direct-party base,
conservative core, preliminary forecast share, explicit government direction,
and direct-mega burden. The two highest structural candidates define the
dominant/runner pair.

Activation additionally requires:

- average direct-party reliability above `0.50`;
- directional advantage above `0.02`;
- either a structural score gap above `0.015` or shock intensity above `1.0`.

This makes 2002 inactive because evidence reliability is low, 2012 inactive
because the dominant candidate is insufficiently certain, and 2022 inactive
because there is no directional dominance. 2007 is classified asymmetric and
2017 rupture-driven.

## Vote movement

Only the flexible shares of the dominant candidate and runner-up are moved.
The conservative core floors remain fixed, and every third-candidate share is
preserved exactly. The fixed expansion gain is `0.50`, with a `0.30` maximum
log shift.

## Fixed promotion result

No coefficient grid was selected from presidential outcomes. One fixed design
was evaluated against the frozen active-v4 snapshot.

| Metric | Active v4 | Active v5 | Change |
|---|---:|---:|---:|
| Regional weighted macro MAE | 4.7248%p | 4.3378%p | -0.3870%p |
| National candidate macro MAE | 3.5408%p | 3.0538%p | -0.4870%p |
| Winner accuracy | 80% | 80% | unchanged |

National MAE changes:

- 2002: unchanged
- 2007: -0.9765%p
- 2012: unchanged
- 2017: -1.4586%p
- 2022: unchanged

This remains a through-2022 development promotion, not untouched holdout
evidence. The frozen input and decision are under
`outputs/contest_regime_experiment/`.
