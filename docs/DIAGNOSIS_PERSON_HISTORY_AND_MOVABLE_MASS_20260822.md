# Two structural gaps behind the stronghold errors

## Status

- Date: 2026-08-22
- Status: diagnosis only; no change made
- Follows `DIAGNOSIS_STRONGHOLD_ERRORS_20260822.md`, which separated the 2007
  and 2017 failures without naming what each one is

Both gaps were proposed as hypotheses and both are confirmed by inspection
rather than by fit. Neither needs an outcome to state.

## Gap 1: there is no person-level history

The 2007 failure was described as "a party-less candidate loses his regional
prior". Inspection shows something stronger: **the model has no person-level
regional history at any point**, for any candidate.

`electorate_layers` builds every history feature with

    for (region_id, bloc), group in history.groupby(["region_id", "bloc"])

so both channels are keyed on bloc. The names invite the wrong reading:
`direct_party_*` is the bloc's party-list series and `candidate_ballot_*` is the
bloc's **candidate-ballot election type** series. Neither is the candidate's own
record. Nothing in the feature set identifies a person across elections.

이회창 in 광주 makes the consequence visible, because he ran three times:

| | bloc | `candidate_ballot_recent_base` | `candidate_ballot_effective_elections` | actual |
| --- | --- | ---: | ---: | ---: |
| 이회창 2002 | 국민의힘 | **0.0476** | **5.2416** | 0.0359 |
| **이회창 2007** | **무소속** | **0.0000** | **0.0000** | 0.0371 |

In 2002 the feature is close to what he actually took. In 2007 the same person,
now with more history than before, has **zero effective elections**. The record
did not degrade; it was never his to begin with, and his bloc changed.

Three effects compound for an independent:

1. no 무소속 history exists in 광주, so the channel returns `None` and every
   candidate feature is written as 0.0
2. `bloc not in MAJOR_PARTY_CORE_BLOCS` returns core 0.0 by rule
3. `bloc == INDEPENDENT_BLOC` multiplies persistence by 0.25

With no anchor the fitted base falls back to something generic and puts him at
11.70 in 광주 against a realised 3.71 - and that 8-point excess, together with
이명박's, is exactly 정동영's -15.69 stronghold miss.

`third_candidate_lineage.csv` already records `origin_lane = conservative` for
him, and V24 already routes inherited regional identity through dated party
rename and merger paths. Neither reaches a person whose bloc is 무소속.

## Gap 2: shock magnitude sets how much vote is movable

The regime response erodes the burdened candidate's core floor. The erosion
rate comes from the shock, and in 2017 it takes exactly one value across the
whole country:

    rupture_floor_activation unique values in 2017: [0.866]

Regional core depth **is** computed, and it varies:

| 2017 region | core floor | erosion | effective floor | veto transfer |
| --- | ---: | ---: | ---: | ---: |
| 경북 | **0.4827** | 0.866 | 0.0647 | 0.0554 |
| 대구 | **0.4584** | 0.866 | 0.0614 | 0.0527 |
| 경남 | 0.4142 | 0.866 | 0.0555 | 0.0496 |
| 울산 | 0.3520 | 0.866 | 0.0472 | 0.0409 |
| 강원 | 0.3343 | 0.866 | 0.0448 | 0.0392 |

The same 86.6 % is applied everywhere, so the absolute mass released scales with
core depth: 경북 releases 0.418, 강원 0.289. **The region with the deepest core
gives up the most vote.** That inverts what a core is for. A core floor is the
share that does not move; multiplying it by an election-level rate makes the
immovable share a function of how large it was.

This is the proposition that shock magnitude and movable mass have to be
separated, stated concretely: the model measures both - `mega_issue_intensity`
per election, `base_runner_core_floor` per region - and then lets the first
overwrite the second.

It is why 홍준표 undershoots most exactly in TK. The veto contributes 5.3 to 5.5
points of a 12.1 to 12.5 point error there, so roughly 45 %; the remainder is
in the base, which has its own reason to pull him off a 0.7007 prior.

## Why these are a different class from the corrections already rejected

The dispersion rescale was rejected because it was fitted on the two elections
that showed the effect and calibrated against the variance it was scored on.
Neither gap here has that shape.

**Gap 1 is a representation defect provable without any outcome.** That the
same person has 5.24 effective elections in one contest and 0.00 in the next is
wrong on its face. The fact needed to fix it - that 이회창 2007 is the 이회창
of 1997 and 2002 - is a documented pre-election fact, and part of it is already
recorded in the lineage table.

**Gap 2 is an internal inconsistency provable without any outcome.** A quantity
defined as the share that cannot move should not be scaled by an election-level
rate that has no regional content. The model already holds the regional
quantity; it is discarded.

Both were found by reading residuals, and that must be declared. But unlike a
dispersion gain, neither needs the panel to justify its existence - only to
measure what changing it would cost.

## What has not been established

How much either gap is worth. Gap 2's contribution in TK is bounded at roughly
45 % of the local error by the veto transfer, and Gap 1's at the 8-point excess
per rival candidate in 호남, but neither has been run through the model. Both
would be base-stage or postprocess changes and both carry the observation
shortage every structural layer here already has: the veto fires on two scored
elections and the independent case appears in one.
