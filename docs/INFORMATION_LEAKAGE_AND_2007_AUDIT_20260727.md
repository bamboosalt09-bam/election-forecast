# Information Leakage and 2007 Error Audit

Date: 2026-07-27

Active policy: `active_strict_nested_v7_postprocess_selection`

## Conclusion

No direct target-outcome leakage was found in the 2007 prediction path. A
full temporary mutation changed the 2007 realized result to an extreme
counterfactual distribution and reran preliminary assignment plus the active
strict-base path. Preliminary shares, pre-hierarchy predictions, final regional
predictions, and row identity were exactly unchanged.

The backtest is nevertheless not an untouched forecast experiment. Two
post-outcome dependencies and one indirect manual-seed lineage must be reported
separately from direct predictor leakage:

1. National prediction diagnostics are aggregated with realized target-election
   `contest_votes`. This does not alter regional predictions, but the reported
   national prediction and national MAE are post-election diagnostics.
2. The evaluated candidate denominator comes from the historical A/B/C result
   table, with an explicit realized-contest exclusion for the 2002 C candidate.
   This is not yet a fully ex-ante candidate-universe rule.
3. The active policy declares the manual issue seed disabled, but precomputed
   `candidate_party_tone_gap.csv` and `candidate_public_treatment.csv` were built
   using `candidate_issue_profile.csv` and `mega_issue_attribution.csv`. Those
   derived tables feed candidate conversion and the automatic issue seed. The
   manual seed is therefore disabled as a direct forecast input but not fully
   removed from transitive data lineage.

The fixed gains and candidate stage definitions were also developed while the
through-2022 outcomes were visible. This is model-development selection, not a
mechanical target-row leak, but it prevents interpreting the five-election
score as an untouched holdout estimate.

## Executed Checks

- `python presidential_issue_engine/audit_point_in_time.py --deep`: PASS
  - target-outcome-invariance rows: 215
  - future Assembly term inflation: 0 of 952 checked rows
  - limitation: the configured Assembly-match audit source was absent, so the
    run reported `assembly_matches_present=0`
- `python scripts/audit_slot_predictor_leakage.py`: PASS
  - active v7 uses no realized `slot_A`, `slot_B`, `slotA_prior`, or
    `slotB_prior` predictor
- `python presidential_issue_engine/audit_weight_selection_boundary.py`: PASS
  - scored scope stops at 2022 and the rolling warmup is 1997
- Active manifest date scan:
  - no parsed election-linked date exceeded its election date
- Full temporary 2007 outcome mutation:
  - `assignment_max_abs_diff = 0.0`
  - `preliminary_mean_share_max_abs_diff = 0.0`
  - `pre_hierarchy_pred_max_abs_diff = 0.0`
  - `layer_pred_max_abs_diff = 0.0`
  - row identity unchanged

## Why Lee Myung-bak Is Underpredicted

The error is produced mainly by the early-fold fallback policy, not by an
inability of the stored structural signals to identify the conservative base.

### Active strict-base path

National diagnostic shares, using realized contest-vote aggregation:

| Step | Lee Myung-bak | Chung Dong-young | Lee Hoi-chang |
|---|---:|---:|---:|
| Ridge raw | 37.215% | 35.596% | 33.257% |
| Third adjustment and normalization | 33.620% | 32.640% | 33.739% |
| Party-context adjustment | 33.285% | 33.785% | 32.931% |
| Public-treatment and generation adjustments | 35.422% | 35.727% | 28.851% |
| Third-candidate hierarchy | 38.837% | 39.258% | 21.905% |
| Actual three-candidate share | 54.140% | 29.089% | 16.771% |

The Ridge fit has only 1997 and 2002 history. Its six predictors leave the
three 2007 candidates close together. The party-context layer then rates Chung's
context support above Lee's (`0.4463` versus `0.3359`) because Lee receives a
larger fragmentation/attack score. That reverses the already-small A-B margin.

The strict predictor fit also gives identical coefficients and contributions
to `issue_advantage` and `rif` in this fold. With the strict neutral defaults,
the two variables carry duplicate information for 2007 and do not independently
recover economic-growth voting or incumbent-party rejection.

### Structural candidate stages

The prior direct-party layer itself contains the missing contrast:

- Lee Myung-bak direct-party recent base: `48.03%`
- Chung Dong-young direct-party recent base: `35.09%`

Candidate-stage diagnostics are:

| Stage | Lee prediction | National candidate MAE |
|---|---:|---:|
| `strict_base` | 38.837% | 10.202%p |
| `structural` | 43.740% | 7.570%p |
| `structural_mega_shock` | 44.016% | 7.311%p |
| `structural_mega_shock_regime` | 47.636% | 4.898%p |

V7 does not select among those stages for 2007. It has only one prior scored
election, below the required minimum of two, so it falls back to `strict_base`.
The all-or-nothing fallback consequently disables forecast-time terrain and
regime information that is already present in the input frame.

## Recommended Correction Boundary

Do not choose the 2007 stage by its 2007 error. Instead, separate layers by
epistemic status:

- Make a predeclared, outcome-blind structural terrain baseline available from
  the first fold whenever dated prior party-ballot history exists.
- Keep optional mega-shock and regime-response gains behind prior-fold selection
  or derive them from non-presidential proportional/local elections.
- Replace realized A/B/C candidate scope with a forecast-date candidate-universe
  rule based on ballot status, withdrawal status, and preliminary expected
  viability.
- Produce deployable national predictions with prior turnout/population weights;
  retain realized-contest-vote aggregation under an explicitly diagnostic name.
- Rebuild party tone, public treatment, candidate conversion, and automatic
  issue seeds from a dependency graph that excludes manual seed files whenever
  `manual_issue_seed_enabled=false`.

No active model change was made during this audit.

## Follow-up promotion

After this audit, v8 promoted the full evidence-gated pipeline as a universal
historical and future policy. The original v7 audit numbers remain preserved
above as the pre-change diagnosis. See
`docs/UNIVERSAL_EVIDENCE_PIPELINE_V8_20260727.md` for the promoted result.
