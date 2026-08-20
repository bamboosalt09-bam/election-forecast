# Final Presidential Model V24

## Status

V24 is the active through-2022 presidential model selected by
`data/config/current_presidential_model.json`. It is frozen before any 2025
evaluation. V23 remains the immutable rollback model.

V24 is a versioned structural extension of the V23 base configuration, not an
independently retuned Ridge specification. The active runner is
`scripts/run_active_presidential_model_v24.py`; the shared numerical base is
`data/config/active_presidential_model_v23.json`.

## Point-model lineage

The base estimator is strict chronological Ridge with the same six slot-free
predictors and coefficients used by V23. Every target fold is excluded from its
outer fit. V24 then applies the fixed structural cascade recorded in
`outputs/active_presidential_nested_v24/summary.json`.

Relative to V23, V24 formalizes these versioned structural changes:

1. restore the ballot-faithful third candidate for 2002 and keep 1992 and 1997
   as genuine historical three-way elections;
2. score all candidates at or above a uniform 1% national share rather than
   using election-specific exclusions;
3. prevent withdrawn candidates from overwriting the slot of the candidate who
   actually remained on the ballot;
4. represent party-backed non-major organization strength continuously from
   strictly prior party-list evidence;
5. cap weak third-party vehicles by their own direct party evidence unless a
   documented major-party split lineage supplies carry-in;
6. apply the recorded strong-incumbent-veto and weak-same-lane refusal layers,
   including the theoretical 1% floor.

The complete derivation, rejected alternatives, and sensitivity grids remain
in `docs/EXPERIMENT_V24_LINEAGE_20260819.md`. The final point predictions are
not modified by the interval layer.

## Historical score panel

| Election | candidates x regions | Regional weighted MAE | National candidate MAE | Winner |
|---|---:|---:|---:|---|
| 2002 | 48 | 2.6579%p | 2.2399%p | no |
| 2007 | 48 | 4.0405%p | 1.1212%p | yes |
| 2012 | 34 | 2.5592%p | 1.0323%p | yes |
| 2017 | 51 | 3.3391%p | 0.5294%p | yes |
| 2022 | 51 | 1.2523%p | 0.4556%p | yes |
| **Equal-election macro** | **232** | **2.7698%p** | **1.0757%p** | **4/5** |

V23 reported 3.3679%p regional and 1.5978%p national MAE on 199 rows. V24's
headline is not a clean same-panel ablation because the restored weak-third
rows expand the panel to 232 rows. Both versions use the through-2022
development sample; neither is an untouched historical holdout.

## Vote-share predictive intervals

The active interval artifact uses an empirical candidate-common, regional, and
local log-share residual hierarchy plus uncertainty in forecast-time regional
vote weights. For a target fold, both sources are estimated only from strictly
earlier elections. The residual scale is fixed at 1.0 and is not selected for
coverage.

| Nominal level | Historical equal-election coverage | Mean national width |
|---:|---:|---:|
| 50% | 91.67% | 4.42%p |
| 80% | 91.67% | 8.39%p |
| 90% | 91.67% | 10.73%p |
| 95% | 91.67% | 12.73%p |

The calibration has only four evaluable elections and eleven candidate
outcomes. These are predictive intervals, not Ridge coefficient confidence
intervals, and the observed coverage must not be presented as a future
guarantee. 2002 is the warm-up residual election and therefore has no
chronological interval of its own.

## Overfitting and manual-rule disclosure

The outer Ridge fit is target-excluded, the interval bounds are target-blind,
and no 2025 result is used. This does not erase development-sample selection:
several postprocess stages and numerical gains were historically proposed and
compared on the elections through 2022. The V24 summary explicitly records
`strict_nested_postprocess_selection=false` and
`candidate_numeric_parameters_historically_development_selected=true`.

The 1% weak-candidate floor is a declared theoretical support floor. The 10%p
incumbent-veto trigger is a documented structural hypothesis. Neither should be
described as learned automatically from an untouched holdout. Future changes
must be a new version and must retain rejected variants and sensitivity records.

## Reproduction and canonical artifacts

```bash
python scripts/run_active_presidential_model_v24.py --output-dir outputs/reproduction_v24
python scripts/build_active_v24_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v24.py
python -m pytest -q
```

Canonical records:

- `outputs/active_presidential_nested_v24/nested_predictions.csv`
- `outputs/active_presidential_nested_v24/input_manifest.csv`
- `outputs/active_presidential_nested_v24/summary.json`
- `outputs/active_presidential_nested_v24/national_predictive_intervals.csv`
- `outputs/active_presidential_nested_v24/predictive_interval_manifest.json`
- `outputs/active_presidential_nested_v24/promotion_manifest.json`
- `outputs/active_presidential_nested_v24/finalization_manifest.json`

No 2025 prospective output is part of this promotion.
