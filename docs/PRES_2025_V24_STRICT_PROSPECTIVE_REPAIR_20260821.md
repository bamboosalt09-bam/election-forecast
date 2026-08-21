# 2025 V24 strict prospective repair — 2026-08-21

## Status and boundary

This record covers an outcome-free `pres_2025` execution with the promoted V24
model frozen.  The cutoff is 2025-06-02 (D-1).  No 2025 realised vote, winner,
error, or performance metric was read, fitted, selected, or written.  V24
coefficients, thresholds, historical rows, and the active pointer were not
changed.

## Rejected earlier harness

The earlier prospective harness was rejected even though it emitted a plausible
final composition.  It failed to reproduce the 232 frozen V24 historical rows:
the maximum absolute `layer_pred` difference was 0.03715257684532684.

The primary execution defect was an input-policy boundary error.  The promoted
runner builds `_prepare_rows()` and `_base_layer_frame()` inside
`strict_input_policy()`, while the prospective harness had built both frames
before entering that context.  This silently re-enabled undated curated
sensitivity inputs.  The first divergence appeared in the rederived Ridge base,
not in the V24-only postprocess.

A separate Assembly integration defect also affected the prospective target.
Current-government target rows had been linked to the governing-party nominee
and then reused both as candidate attention and as government burden.  This
counted 11,477 government-target sentences as if they were the nominee's own
person/party attention.  The repair keeps person/party rows in candidate
attention and routes government rows only to the issue-character government
burden fields.

The candidate context builder also sorted the risk-issue set before its
floating-point reduction so repeated runs cannot depend on Python set order.

## Strict reproduction guard

The repaired harness now applies the same strict input boundary before any
historical or target assembly.  It then runs the shared active pipeline and all
V24 extensions in promoted order:

1. strong incumbent veto;
2. third-candidate lineage ceiling;
3. weak same-lane refusal.

Before accepting the target rows, the harness compares every historical
`(election_id, region_id, source_slot)` row against the frozen V24 artifact.

- compared rows: 232
- maximum absolute `layer_pred` difference: 2.220446049250313e-16
- frozen canonical SHA-256:
  `edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a`
- guard tolerance: 1e-12
- result: PASS

The standalone V24 runner was also executed to a separate directory and its
`nested_predictions.csv` was byte-identical to the frozen canonical artifact.

## Input coverage and Assembly routing

All six frozen Ridge predictors and `candidate_weight` are nonmissing for all
51 target rows (17 regions × 3 candidates).  The candidate-strength link has 54
person/party issue rows.  The current-government bridge contains 736 aggregate
rows and 11,477 sentences; 53 aggregate rows have directional evidence.  Its
signed weight is -14.58665525 and absolute directional weight is
24.09463025.  These 736 government rows are excluded from candidate attention
and enter only the government-burden route.

## Outcome-free national chain

National values below use 2022 valid-vote regional volume only as aggregation
weights.  They are forecasts, not scores.

| Candidate | Preliminary | Ridge/base | Pre-V24 extension | Final V24 |
|---|---:|---:|---:|---:|
| 김문수 | 43.8286% | 41.6544% | 41.7420% | 47.9155% |
| 이재명 | 32.3888% | 35.2465% | 38.3642% | 46.4876% |
| 이준석 | 23.7827% | 23.0991% | 19.8938% | 5.5969% |

The strong-incumbent-veto extension did not activate: the pre-extension model
did not project the required anti-incumbent lead.  The lineage ceiling reduced
the third candidate to 10.1938%, and weak same-lane refusal reduced it further
to 5.5969% while reallocating the released mass compositionally.

## Preliminary-model interpretation

The target feature audit confirms that the preliminary result is not caused by
zero-filled Ridge predictors.  The 2022-volume-weighted values are:

| Candidate | issue_advantage | rif | partisan_prior | landscape_bloc_alignment | landscape_centrist | landscape_inferred_prior | candidate_weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| 이재명 | 0.010068 | 0.001510 | -0.011696 | -0.013327 | -0.001499 | -0.002591 | 0.454957 |
| 김문수 | 0.008671 | 0.001301 | 0.016104 | -0.024749 | -0.002056 | -0.002435 | 0.482215 |
| 이준석 | 0.008730 | 0.001309 | -0.003837 | 0.038077 | 0.003555 | -0.002868 | 0.433504 |

`partisan_prior` is a regional-lean feature: regional bloc share minus that
bloc's national mean, with time/type weighting and shrinkage.  It is not the
2024 party-list national level.  The 2024 national bloc shares (52.8007%,
37.8918%, and 5.0254%) select and order the three candidate lineages, but are
not passed as a target-specific national-level Ridge predictor.  That is a
frozen-model limitation, not a missing-value execution defect.  Changing it
would require a new post-V24 model experiment and is outside this strict run.

## Artifacts

- `outputs/prospective_pres_2025_v24/prospective_predictions.csv`
- `outputs/prospective_pres_2025_v24/national_summary.csv`
- `outputs/prospective_pres_2025_v24/target_feature_audit.csv`
- `outputs/prospective_pres_2025_v24/prediction_stage_audit.csv`
- `outputs/prospective_pres_2025_v24/run_manifest.json`
- three V24 extension audit CSVs in the same directory
