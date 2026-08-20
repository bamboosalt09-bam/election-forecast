# Pres 2025 strict prospective run under frozen V25 — 2026-08-21

## Boundary

This is an unscored forecast-only execution with cutoff 2025-06-02 (D-1).
No 2025 realised vote, winner, error, or performance metric was read, fitted,
selected, or written. V25 was frozen before this run.

## Historical reproduction gate

The prospective harness enters the exact bounded V25 runtime before assembling
either historical or target rows. It must reproduce the canonical historical
artifact before target output is accepted.

- compared historical rows: 232
- maximum absolute `layer_pred` difference:
  `2.220446049250313e-16`
- tolerance: `1e-12`
- canonical V25 SHA-256:
  `218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b`
- result: PASS

## Accepted third-candidate boundary

The forecast keeps V24's accepted `prediction_tilted` weak-C route. It does not
enable the rejected `affinity_only` route and does not rebind the V23 automatic
third-candidate profile/pressure pair. The target audit contains 17 lineage
ceiling rows and 17 weak same-lane refusal rows, all marked
`prediction_tilted`.

## Outcome-free national composition

The national aggregation uses 2022 valid-vote regional volume as a fixed prior
regional weight source.

| Candidate | Slot | Predicted share |
|---|---|---:|
| 김문수 | A | 47.6800% |
| 이재명 | B | 46.7231% |
| 이준석 | C | 5.5969% |

These values are forecasts, not scores. They must not be compared with the
realised 2025 result inside model selection or documentation of the repair.

## Integrity checks

- target rows: 51 (17 regions x 3 candidates)
- regional compositions sum to 1.0
- frozen Ridge predictors and candidate weight are present for every row
- current-government Assembly evidence reaches only the declared burden route
- outcome columns used: none
- performance metrics computed: false
- model selection performed: false
- model parameters changed: false
- deterministic target prediction SHA-256:
  `309c6b3487c1a15e445ae22aafdc65e967885e19c24ae71287bbc5164da37b7e`

## Artifacts

- `outputs/prospective_pres_2025_v25/prospective_predictions.csv`
- `outputs/prospective_pres_2025_v25/national_summary.csv`
- `outputs/prospective_pres_2025_v25/target_feature_audit.csv`
- `outputs/prospective_pres_2025_v25/prediction_stage_audit.csv`
- `outputs/prospective_pres_2025_v25/input_manifest.csv`
- `outputs/prospective_pres_2025_v25/run_manifest.json`
- three structural postprocess audit CSVs in the same directory
