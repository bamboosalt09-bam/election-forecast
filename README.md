# Election Forecast: Through-2022 Engine

This is the active open-source presidential forecast workspace. Model selection and scored
comparisons are restricted to the 2002, 2007, 2012, 2017, and 2022 presidential elections.
The 1997 presidential election is rolling warmup only. No 2025 presidential outcome is used.

## Workspace

- active engine: `C:\english_folder\poll_project`
- locked through-2022 baseline: `C:\english_folder\poll_project_through2022_baseline_locked_20260714`
- post-2025 outcome-aware archive: `C:\english_folder\poll_project_post2025_outcome_aware_20260714`

The active engine must not import weights or performance claims from the post-2025 archive.
Long Assembly transcript reprocessing is not run unless explicitly requested.

## Active model

- current version: `v23` (frozen through-2022 baseline)
- canonical runner: `python scripts/run_current_presidential_model.py`
- active pointer: `data/config/current_presidential_model.json`
- final model specification: `docs/FINAL_MODEL_V23_20260802.md`
- immediate rollback reference: `backups/model_checkpoints/20260802_pre_v23_unified_withdrawal_generation`

- active policy: `slot_free_hierarchy_no_neutral` under strict chronological nested evaluation
- six slot-free Ridge predictors; realized `slot_A`, `slot_B`, `slotA_prior`, and `slotB_prior` are forbidden
- candidate order comes from strict rolling preliminary expected shares
- weak national third-candidate hierarchy with regional shape preserved
- direct neutral-context vote adjustment disabled
- withdrawn-candidate transfer addition retained after Ridge
- one canonical automatic candidate profile for final and withdrawn candidates
- one active withdrawal-transfer registry; legacy transfer CSVs are isolated
- latest strictly prior official generation composition is active
- one exact-party lineage ledger for prior presidential, Assembly, and local elections
- broad blocs are projected from that ledger only at the Ridge feature boundary
- dated party rename/merger paths route inherited regional identity without vote fitting
- the same regional-lineage estimator and routing formula applies to all regions
- automatic Assembly issue seed; manual issue seed off
- third-candidate competitiveness and character controls
- candidate regionalism and within-bloc regional transfer
- electorate layers: core, critical support, swing support
- active electorate preference gain `0.04` (capped strict nested selection)
- strictly prior direct-party terrain gain, reliability capped at `0.25`
- regional composition axes: conservative, liberal, progressive, centrist,
  regionalist, and reform; applied only to non-core mobility
- conservative concrete-support floor; critical and swing regime responses use
  explicit elasticities `0.75` and `1.25`
- concrete support is restricted to exact People Power and Democratic Party
  lineages; all other stable support is reclassified as critical support
- nonmajor stable support remains a lane reservoir; bounded wasted-vote
  pressure can move it only to an ideologically aligned major-party candidate
- historical fixed `0.04` post-hoc experiment retained as a separate diagnostic
- turnout and nonvoter gains `0`

The electorate layer now estimates direct-party ballots separately from candidate ballots.
Candidate-tone magnitude and confidence are preserved; the former per-election maximum
normalization has been removed. See `docs/ELECTORATE_LAYER_MODEL.md`.

## Shadow stance classifier

A precision-first Assembly sentence classifier is under evaluation. It separates
directionality from polarity, abstains on uncertain or attribution-risk sentences, and
retains neutral informational content as unsigned metadata. It is not connected to the
active forecast because its current directional coverage and statistical error bound fail
the adoption gate. A manually reviewed 122-sentence direct-target expansion, fixed Korean
sentence-embedding child, and hard-risk consensus ensemble improve shadow OOF coverage from
`16.22%` to `23.42%` (engineering holdout: `20.59%`), but remain inactive. See
`docs/STANCE_PRECISION_FIRST_20260717.md`.

Two additional fixed context encoders have now been implemented and audited in shadow mode:
`klue/roberta-small` and `jhgan/ko-sroberta-nli`. The National Assembly Library is registered
as an optional, date-gated domain-corpus source rather than misrepresented as a stance model.
The NLI candidate reached only `67.5%` directional precision on 40 newly locked emissions;
the three-model majority had only three independent emissions and two were harmful. Both are
rejected for active use. See `docs/STANCE_CONTEXT_MODELS_20260717.md`.

The consolidated experiment decision is to use Assembly transcripts primarily as an issue-
salience source, not as a direct public-support stance source. Sentence-level stance remains
shadow, while a bounded v14 aggregate issue-character overlay is active at character gain
`0.04` and link gain `0.01`; it never adjusts candidate vote share directly. Neutral sentences
continue to contribute unsigned issue information. See
`docs/ASSEMBLY_SALIENCE_VS_STANCE_EXPERIMENT_20260717.md`.

## Current performance

- active v23 regional contest-vote weighted equal-election macro MAE: `3.367899%p`
- active v23 national candidate equal-election macro MAE: `1.597845%p`
- active winner accuracy: `80%`
- target election excluded from Ridge fitting; no target-specific stage selection
- undated curated issue/region weights: disabled in strict mode
- 2025 outcome use: none

V23 retains the V22 automatic-control bundle and unifies candidate profiles,
withdrawal profiles, transfer events, and generation composition under one
canonical registry. The five scored elections are a development sample rather
than an untouched historical holdout. See `docs/FINAL_MODEL_V23_20260802.md`.

The primary regional metric weights candidate-region errors by observed contest votes within
each election and then averages elections equally. Because observed votes supply the weights,
it is a post-election diagnostic. Full definitions and current tables are in
`docs/CURRENT_MODEL_PERFORMANCE_20260728.md` and
`docs/REGIONAL_ACCENT_AND_REGIME_DIAGNOSIS_20260728.md`.

## Validation

```powershell
python presidential_issue_engine\audit_weight_selection_boundary.py
python presidential_issue_engine\audit_point_in_time.py --deep
python scripts\audit_active_presidential_model_v23.py
python scripts\audit_slot_predictor_leakage.py
python presidential_issue_engine\robustness_check.py
python scripts\evaluate_electorate_layers.py
python scripts\evaluate_nested_electorate_learning.py
python scripts\run_current_presidential_model.py
python -m pytest -q
```

Latest complete verification:

- through-2022 selection-boundary audit: PASS
- strict deep PIT audit: PASS, outcome invariance `215/215`
- active slot audit: PASS
- active V23 audit: PASS, `199` prediction rows
- tests: `538 passed` (2026-08-10)

## GitHub workflow

The Git repository tracks source, tests, configuration, small canonical data,
and the frozen artifacts needed to audit active V23. Bulk outputs, backups,
virtual environments, API caches, and shadow corpora remain outside Git. Run
`python scripts/audit_github_baseline.py` before pushing. Branch, review, CI,
data-boundary, and model-promotion rules are documented in
`docs/GITHUB_WORKFLOW.md`.

## Key artifacts

- `docs/FINAL_MODEL_V23_20260802.md`
- `data/config/current_presidential_model.json`
- `data/config/active_presidential_model_v23.json`
- `data/raw/party_lineage_transitions.csv`
- `outputs/unified_exact_lineage_v21/exact_lineage_events.csv`
- `outputs/automatic_controls_v23/lineage_manifest.json`
- `outputs/automatic_controls_v23_ablation_v3/decision.json`
- `outputs/active_presidential_nested_v23/summary.json`
- `outputs/active_presidential_nested_v23/nested_predictions.csv`
- `outputs/active_presidential_nested_v23/stage_selection_audit.csv`
- `outputs/active_presidential_nested_v23/input_manifest.csv`
- `outputs/active_presidential_nested_v23/finalization_manifest.json`
- `docs/ACTIVE_V23_UNIFIED_WITHDRAWAL_GENERATION_20260802.md`
- `outputs/unified_exact_lineage_v21_ablation/decision.json`
- `outputs/regional_identity_v16_camp_donor_experiment/decision.json`
- `docs/REGIONAL_IDENTITY_V16_20260728.md`
- `archives/experiments/regional_identity_v16_20260728/archive_manifest.csv`
- `outputs/chungcheong_identity_v15_experiment/decision.json`
- `archives/experiments/chungcheong_identity_v15_20260728/archive_manifest.csv`
- `outputs/all_fold_regional_offset_v14_experiment/decision.json`
- `outputs/strategic_lane_transfer_v12_experiment/decision.json`
- `outputs/orientation_affinity_fix_v13_experiment/decision.json`
- `outputs/major_party_core_v11_experiment/decision.json`
- `outputs/regional_accent_regime_v10_ablation/summary.json`
- `docs/CURRENT_MODEL_PERFORMANCE_20260728.md`
- `docs/CHUNGCHEONG_ERROR_DIAGNOSIS_20260728.md`
- `docs/REGIONAL_ACCENT_AND_REGIME_DIAGNOSIS_20260728.md`
- `docs/MAJOR_PARTY_CORE_V11_20260728.md`
- `docs/STRATEGIC_LANE_TRANSFER_V12_20260728.md`
- `docs/SAME_LANE_AFFINITY_V13_20260728.md`
- `archives/experiments/regional_accent_regime_v10_20260728/`
- `archives/experiments/major_party_core_v11_20260728/`
- `archives/experiments/strategic_lane_transfer_v12_20260728/`
- `archives/experiments/same_lane_affinity_v13_20260728/`
- `docs/UNIVERSAL_EVIDENCE_PIPELINE_V8_20260727.md`
- `docs/PARTY_CONTEXT_COHESION_V9_20260727.md`
- `docs/ACTIVE_NESTED_MODEL_PROMOTION_20260718.md`
- `data/config/electorate_layers.json`
- `data/config/electorate_layers_fixed_experiment.json`
- `outputs/electorate_layer_experiment/summary.json`
- `outputs/electorate_layer_experiment/fixed_structural_predictions.csv`
- `outputs/electorate_layer_profile_experiment/summary.json`
- `outputs/electorate_mass_profile_experiment/summary.json`
- `outputs/electorate_layer_experiment/history_source_audit.csv`
- `outputs/electorate_layer_experiment/pres_2022_region_diagnostics.csv`
- `outputs/electorate_nested_learning/summary.json`
- `outputs/electorate_nested_learning/outer_selected_gains.csv`
- `outputs/electorate_nested_learning/nested_comparison.csv`
- `outputs/electorate_nested_learning_uncapped/summary.json`
- `presidential_issue_engine/report/tables/issue_vote_engine_rolling_predictions.csv`
- `presidential_issue_engine/report/tables/issue_vote_engine_rolling_national_summary.csv`
