# Strategic Lane Transfer V12 (2026-07-28)

## Decision

Promote the outcome-blind strategic lane-transfer layer on top of v11.

Nonmajor stable support is not candidate concrete support, but it is not
discarded. It remains effective critical-support mass associated with a broad
ideological lane. A bounded fraction can move to an aligned major-party
candidate under wasted-vote pressure.

## Formula

For each nonmajor candidate and region:

`transfer = min(prediction, critical_reservoir * pressure)`

`pressure = major_party_gravity * (1 - wasted_vote_resistance) * (1 - donor_preliminary_share / strongest_major_preliminary_share) * confidence * lane_clarity`

Recipients must be exact major-party lineages and receive the transfer in
proportion to squared ideological affinity. Conservative-to-liberal and
liberal-to-conservative movement is forbidden. The operation is zero-sum
within each region.

## Leakage boundary

- Candidate viability comes from strict rolling preliminary expected share.
- Conversion context is filtered by `available_date` for each election.
- The join uses election and candidate name, not realized A/B/C rank.
- Realized vote share, winner, rank, and election error are not inputs.
- 2025 outcomes are not read or compared.

## Strict nested result

| Metric | V11 | V12 | Delta |
|---|---:|---:|---:|
| Regional weighted macro MAE | 3.589029%p | 3.579543%p | -0.009486%p |
| National candidate macro MAE | 2.075590%p | 2.075092%p | -0.000499%p |
| Winner accuracy | 80% | 80% | 0 |
| Equal-share deviation slope | 0.8938 | 0.8951 | +0.0013 |

Only 2017 activates. Ahn Cheol-soo transfers a regional mean `0.2363%p` and
maximum `0.6390%p` to the aligned major-party candidate. Regional 2017 MAE
improves from `4.6078%p` to `4.5604%p`. Every other election is an exact no-op.

The gain is small. Promotion is primarily a structural correction: it models
where nonmajor critical support can go instead of either deleting it or
granting it as immutable candidate concrete support.

## Artifacts

- `presidential_issue_engine/strategic_lane_transfer.py`
- `tests/test_strategic_lane_transfer.py`
- `scripts/evaluate_strategic_lane_transfer_v12.py`
- `outputs/strategic_lane_transfer_v12_experiment/decision.json`
- `outputs/active_presidential_nested_v12/`
- `archives/experiments/strategic_lane_transfer_v12_20260728/`
