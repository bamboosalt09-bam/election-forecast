# Same-Lane Affinity V13 (2026-07-28)

## Finding

The shared orientation helper contained this broad set-intersection test:

`pair intersects liberal labels and pair intersects centrist labels`

Because `liberal_centrist` appeared in both sets, a pair consisting of
`conservative` and `liberal_centrist` incorrectly received affinity `0.65`.
In 2017 this made Ahn Cheol-soo equally close to Moon Jae-in and Hong Joon-pyo
inside the older within-bloc feature builder.

## Correction

Affinity `0.65` now applies only to the exact pair
`liberal_centrist` and `centrist`. Liberal versus liberal-centrist remains
`0.70`; conservative versus liberal-centrist is `0`.

The v12 strategic transfer already had a defensive cross-camp gate. V13 makes
the shared helper consistent so the older within-bloc layer obeys the same
rule.

## Strict nested result

| Metric | V12 | V13 | Delta |
|---|---:|---:|---:|
| Regional weighted macro MAE | 3.579543%p | 3.580729%p | +0.001186%p |
| National candidate macro MAE | 2.075092%p | 2.079116%p | +0.004024%p |
| Winner accuracy | 80% | 80% | 0 |

The tiny adverse movement is retained and documented. Keeping a logically
invalid cross-camp transfer because it marginally lowers historical error would
be outcome-aware model distortion.

## Artifacts

- `presidential_issue_engine/issue_vote_engine.py`
- `tests/test_strategic_lane_transfer.py`
- `scripts/evaluate_orientation_affinity_fix_v13.py`
- `outputs/orientation_affinity_fix_v13_experiment/decision.json`
- `outputs/active_presidential_nested_v13/`
- `archives/experiments/same_lane_affinity_v13_20260728/`
