# Strict Nested Stage Selection v7 (2026-07-27)

## Purpose

This change removes target-election outcomes from postprocess-stage selection.
The previous v6 Ridge outer fits were target-excluded, but the structural,
mega-issue, incumbent-shock, and contest-regime stack had been promoted after
reviewing aggregate 2002-2022 performance.

The v6 code, policy, inputs, and canonical outputs are frozen at:

`archives/experiments/pre_full_nested_v7_20260727/`

`SHA256SUMS.csv` contains the restoration hashes.

The completed v7 implementation and outputs are frozen separately at:

`archives/experiments/full_nested_v7_20260727/`

## Selection rule

Five ordered candidates are evaluated as PIT-safe outer predictions:

1. `strict_base`
2. `structural`
3. `structural_mega`
4. `structural_mega_shock`
5. `structural_mega_shock_regime`

For target election `t`, the selector receives only regional vote-weighted MAE
from scored elections strictly earlier than `t`. At least two prior scored
elections are required. Otherwise the selector falls back to `strict_base`.
Ties prefer the less complex stage.

| Target | Selection history | Selected stage |
|---|---|---|
| 2002 | none | strict_base |
| 2007 | 2002 | strict_base |
| 2012 | 2002, 2007 | structural_mega_shock_regime |
| 2017 | 2002, 2007, 2012 | structural_mega_shock_regime |
| 2022 | 2002, 2007, 2012, 2017 | structural_mega_shock_regime |

The target-loss mutation test leaves every selected stage unchanged.

## Undated curated inputs

Strict v7 does not read these undated manually curated weights:

- `fixed_dataset/issue_importance.csv`
- `fixed_dataset/region_issue_sensitivity_curated.csv`

The engine uses its neutral fallbacks instead: issue importance `0.5` and
regional sensitivity `0.3`. The run-time input manifest confirms neither file
is read by v7.

## Performance

| Election | Regional weighted MAE | National candidate MAE |
|---|---:|---:|
| 2002 | 3.6791%p | 2.4846%p |
| 2007 | 11.7398%p | 10.2024%p |
| 2012 | 2.7897%p | 1.4952%p |
| 2017 | 4.5077%p | 3.0114%p |
| 2022 | 1.5693%p | 0.3359%p |
| **Macro** | **4.8571%p** | **3.5059%p** |

- winner accuracy: `3/5` (`60%`)
- regional prediction sum maximum error: `1.11e-16`
- scored prediction rows: `199`

The old v6 development-selected metrics were regional `4.0522%p`, national
`2.6785%p`, and winner accuracy `80%`. The deterioration is expected: 2007 can
no longer use corrections promoted after seeing the 2007 result.

## Interpretation boundary

This is fully nested **execution and stage selection** for the fixed candidate
family. It is not a retroactively untouched historical holdout. The stage
definitions and their numeric gain values were created during earlier analysis
of the through-2022 sample. No procedure can erase that historical researcher
feedback from the same five elections. The candidate family is frozen now; a
future election can provide a genuinely untouched evaluation.

## Verification

- full test suite: `361 passed`
- strict deep PIT audit: PASS
- through-2022/2025 selection-boundary audit: PASS
- active slot audit: PASS; realized slot predictors absent
- input manifest: `42` files, all hashed, no 2025 path
- undated curated importance/sensitivity paths absent from manifest

## Artifacts

- `data/config/active_presidential_model.json`
- `outputs/active_presidential_nested_v7/summary.json`
- `outputs/active_presidential_nested_v7/stage_selection_audit.csv`
- `outputs/active_presidential_nested_v7/candidate_stage_summary.csv`
- `outputs/active_presidential_nested_v7/input_manifest.csv`
- `outputs/active_presidential_nested_v7/nested_predictions.csv`
- `outputs/active_presidential_nested_v7/national_predictions.csv`
- `archives/experiments/full_nested_v7_20260727/SHA256SUMS.csv`

The frozen legacy-prediction reproduction difference is still diagnostic, not
a guard. Neutralizing the undated inputs intentionally changes the replacement
feature frame, so it cannot reproduce the old frozen v6 predictions. The
frozen prediction column is not used as the v7 final prediction.
