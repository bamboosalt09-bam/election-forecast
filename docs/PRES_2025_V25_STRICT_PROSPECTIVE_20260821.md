# Pres 2025 strict prospective run under frozen V25 - 2026-08-21

## Boundary

This is an unscored forecast-only execution with cutoff 2025-06-02 (D-1).
No 2025 realised vote, winner, error, or performance metric was read, fitted,
selected, or written. V25 was frozen before this run.

## Historical reproduction gate

The prospective harness enters the exact bounded V25 runtime before assembling
either historical or target rows. It must reproduce the canonical historical
artifact before target output is accepted.

- compared historical rows: 232
- maximum absolute `layer_pred` difference: `2.220446049250313e-16`
- tolerance: `1e-12`
- canonical V25 SHA-256:
  `218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b`
- result: PASS

## Prospective mega-control and evidence-routing repair

The first V25 prospective artifact silently inherited the through-2022
`mega_issue_intensity.csv` and `mega_issue_taxonomy.csv` without appending a
target row. The missing lookup defaulted target intensity to `1.0`, so the
direct mega-shock gate, rupture response, and strong-incumbent-veto layer did
not receive the same input contract used by the frozen historical runtime.

The repaired runner generates exactly one target intensity and taxonomy row
from the official Assembly corpus after the D-1 `available_date` boundary. The
historical 16th-22nd Assembly extractor matched each complete speech row, while
the first prospective adapter passed already-collapsed sentence-level issue
labels to the intensity builder. That mismatch separated issue language from
impeachment, martial-law, and government-responsibility language appearing in
another sentence of the same speech. The corrected adapter reconstructs the
original speech rows and reruns the same keyword map, term weights, issue
boosts, and context rules used by the historical extractor.

The reconstructed frequency path classifies the target as
`political_realignment / 0.75`. Six dated official institutional proceedings
also satisfy the universal crisis vocabulary and proceeding-type gate, so the
V22 class-level mapping selects `institutional_crisis / 2.0`. This is a
categorical official-event input, not a numeric target seed. The frequency-only
class remains recorded separately in the audit. Direct candidate shock routing
is then restricted to issues compatible with the selected event class; an
institutional-crisis intensity can scale `regime_change`, but cannot amplify an
unrelated `withdrawal_event` merely because it has stronger attribution.

The government-target evidence path is also separated from direct candidate
strength. The government-linked profile remains available to the incumbent
burden compiler, while the direct mega-issue compiler receives a separately
generated person/party-only candidate profile.

- official source rows: 48,588
- PIT-eligible source rows: 14,985
- unique reconstructed sentence rows: 13,448
- reconstructed historical-granularity speech rows: 7,106
- historical-compatible issue-match rows: 10,159
- reconstructed `regime_change` rows: 828
- qualifying official-proceeding rows: 1,250
- qualifying meetings: 6
- qualifying speakers: 66
- continuous pre-taxonomy diagnostic intensity: `0.9702678425`
- frequency-only automatic class: `political_realignment`
- frequency-only class intensity: `0.75`
- selected semantic class: `institutional_crisis`
- selected class intensity: `2.0`
- proceeding-title semantic marker: true, adjustment applied: true
- target control availability date: `2025-05-05`
- government rows in burden profile: 17
- government rows in direct candidate profile: 0
- target outcomes used: false
- model parameters changed: false

Historical automatic-control rows are copied unchanged. `partisan_prior` is
also unchanged on all 51 target rows (`maximum absolute difference = 0`).

## Accepted third-candidate boundary

The forecast keeps V24's accepted `prediction_tilted` weak-C route. It does not
enable the rejected `affinity_only` route and does not rebind the V23 automatic
third-candidate profile/pressure pair. The target audit contains 17 lineage
ceiling rows and 17 weak same-lane refusal rows, all marked
`prediction_tilted`. The donor's documented pre-election party-origin lane now
precedes noisy speech-axis classification: Lee Jun-seok is routed as
`conservative_centrist`, not `liberal_centrist`.

## Outcome-free national composition

The national aggregation uses 2022 valid-vote regional volume as a fixed prior
regional weight source.

| Candidate | Slot | Predicted share |
|---|---|---:|
| Kim Moon-soo | A | 37.4256% |
| Lee Jae-myung | B | 56.9775% |
| Lee Jun-seok | C | 5.5969% |

These values are forecasts, not scores. They must not be compared with the
realised 2025 result inside model selection or documentation of the repair.

## Integrity checks

- target rows: 51 (17 regions x 3 candidates)
- regional compositions sum to 1.0
- frozen Ridge predictors and candidate weight are present for every row
- current-government Assembly evidence reaches only the declared burden route
- target mega-control is `institutional_crisis / 2.0`; event-class alignment
  prevents this intensity from selecting unrelated withdrawal evidence
- strong-incumbent-veto audit rows: 17
- outcome columns used: none
- performance metrics computed: false
- model selection performed: false
- model parameters changed: false
- deterministic target prediction SHA-256:
  `1a937ae0004441a6ed79afbde288ee38b61a34d625ca60ffa3bdcf457b04cfcc`
- full regression suite: `607 passed`

## Artifacts

- `outputs/prospective_pres_2025_v25/prospective_predictions.csv`
- `outputs/prospective_pres_2025_v25/national_summary.csv`
- `outputs/prospective_pres_2025_v25/target_feature_audit.csv`
- `outputs/prospective_pres_2025_v25/prediction_stage_audit.csv`
- `outputs/prospective_pres_2025_v25/input_manifest.csv`
- `outputs/prospective_pres_2025_v25/run_manifest.json`
- `outputs/prospective_pres_2025_v25/prospective_mega_issue_intensity.csv`
- `outputs/prospective_pres_2025_v25/prospective_mega_issue_taxonomy.csv`
- `outputs/prospective_pres_2025_v25/prospective_mega_issue_taxonomy_audit.csv`
- three structural postprocess audit CSVs in the same directory
