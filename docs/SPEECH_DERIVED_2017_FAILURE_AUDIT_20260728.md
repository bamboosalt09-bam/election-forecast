# Speech-derived 2017 failure audit

Date: 2026-07-28

## Scope

This audit explains the 2017 failure of
`outputs/speech_derived_issue_context_v1` without changing the active v16
model. All counterfactuals ran in temporary directories. They are causal
diagnostics, not promoted forecasting rules.

## Baseline failure

The speech-derived v1 experiment predicts:

| Candidate | Prediction | Actual | Error |
|---|---:|---:|---:|
| Moon Jae-in | 35.268% | 47.476% | -12.208%p |
| Hong Joon-pyo | 33.994% | 27.773% | +6.220%p |
| Ahn Cheol-soo | 30.739% | 24.751% | +5.988%p |

The 2017 national candidate MAE is `8.138759%p` and the turnout-weighted
regional MAE is `8.415853%p`.

## Finding 1: preliminary rank and political role are conflated

The automatic context produces candidate weights of `0.549707` for Moon,
`0.580252` for Hong, and `0.616937` for Ahn. The preliminary estimator then
produces `28.778%`, `40.231%`, and `30.991%`, respectively, and assigns:

| Candidate | Source slot | Assigned rank slot | Assigned role |
|---|---|---|---|
| Moon Jae-in | A | C | major third |
| Hong Joon-pyo | B | A | major candidate |
| Ahn Cheol-soo | C | B | major candidate |

The hierarchy constraint operates on the assigned rank slot. It therefore
regularizes Moon as C while leaving Ahn outside the C constraint. In the
stored 2017 rows, Moon's mean pre-hierarchy prediction is about `36.11%`, but
the strict-base national prediction after hierarchy is only `28.31%`.

The rest of the issue pipeline is not affected by this slot mismatch:
direct-mega, government-burden, and contest-regime code explicitly joins on
`source_slot`. The defect is concentrated in the hierarchy/role layer.

### Single-factor counterfactual

Only the 2017 political-role assignment was restored to Moon/Hong as the two
major-party candidates and Ahn as the third candidate. No issue score, model
coefficient, or response gain was changed.

| Metric | Speech v1 | Corrected role identity |
|---|---:|---:|
| 2017 national candidate MAE | 8.139%p | 3.941%p |
| 2017 regional weighted MAE | 8.416%p | 4.742%p |
| Overall national macro MAE | 3.117%p | 2.277%p |
| Overall regional macro MAE | 4.279%p | 3.544%p |

The corrected 2017 forecast is Moon `41.564%`, Hong `30.724%`, and Ahn
`27.712%`. This confirms that wrong third-candidate identification is the
largest single cause of the 2017 failure.

## Finding 2: the experiment is not fully speech-only for third-candidate size

The issue seed itself is manual-seed-free, but the context lineage still reads
`data/raw/third_candidate_profile.csv`. Its 2017 row gives Ahn:

- viability `0.90`
- centrist appeal `0.85`
- anti-major-party appeal `0.75`
- regional overlap `0.40`
- confidence `0.75`

These values directly raise public-treatment and vote-conversion features.
They are a major reason Ahn's automatic candidate weight exceeds both
major-party candidates.

Removing only the 2017 manual third-candidate row lowers Ahn's candidate
weight from `0.616937` to `0.430271`, his preliminary share from `30.991%` to
`17.997%`, and his final forecast to `20.300%`. This goes too far below the
actual `24.751%`: the current manual prior is too strong, but a zero prior is
also too weak. The missing component is an automatically derived continuous
third-candidate viability estimate.

## Finding 3: incumbent rejection is detected but its released vote is routed
too broadly

The speech profile does detect the 2017 regime-change shock. It finds four
negative directional rows for Hong and no directional rows for Moon or Ahn.
The direct regime-change score for Hong is `-0.388856`, the mega intensity is
`2.0`, and the contest is classified as `rupture_landslide`.

With political roles corrected, the stages are:

| Stage | Moon | Hong | Ahn |
|---|---:|---:|---:|
| Structural | 35.940% | 39.127% | 24.933% |
| Mega | 37.943% | 35.685% | 26.373% |
| Shock | 39.801% | 32.486% | 27.712% |
| Regime | 41.564% | 30.724% | 27.712% |

Ahn is already almost exact at the structural stage (`24.933%` versus
`24.751%`). The mega and incumbent-shock normalization then gives him about
`+2.78%p` of the support released by Hong. The regime layer preserves the
third candidate, so that excess remains.

This is the remaining opposition-consolidation defect: under a strong
incumbent-rejection regime, released flexible support is normalized across all
remaining candidates instead of being routed mainly toward the viable
major-party opposition beneficiary. This must not be solved by a 2017-specific
Moon bonus.

## Defensible design direction

1. Separate `rank_slot` from `political_role`. Rank may change, but major-party
   eligibility and third-candidate status must come from strictly prior party
   lineage, organization, and ballot context.
2. Estimate third-candidate viability continuously from dated speech attention,
   organization, coalition cohesion, party-lineage ballot history, and
   wasted-vote resistance. Do not use a hand-entered election-specific
   viability constant.
3. Apply the hierarchy to the political third candidate, not mechanically to
   preliminary rank 3. Allow a strong third candidate to approach a major
   candidate without being reclassified as a major-party candidate.
4. Under a high-confidence incumbent-rejection regime, route released flexible
   mass with a beneficiary gate based on non-incumbent alignment, major-party
   organization, and strictly preliminary viability. Preserve third-candidate
   concrete support and do not transfer it mechanically.
5. Predeclare the universal rule and test it across every 2002-2022 strict
   nested fold. Do not select gains from the 2017 outcome alone.

## Status

No active code, config, input, or output was changed. Active v16 remains the
production reference. The two counterfactuals were temporary diagnostics only.
