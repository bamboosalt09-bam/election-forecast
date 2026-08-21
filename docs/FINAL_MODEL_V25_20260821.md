# Final Presidential Model V25

## Status

V25 is the active through-2022 presidential model selected by
`data/config/current_presidential_model.json`. It was frozen before any 2025
evaluation. V24 and V23 remain immutable rollback artifacts.

V25 corrects the V24 runner's accidental bypass of promoted V23 runtime
bindings. It does not retune the six-predictor Ridge model or alter V24's
ballot panel, 1% scored floor, lineage ceiling, veto, or accepted weak-C
formula.

## Bounded runtime repair

The exact repair boundary is recorded in
`docs/V24_RUNTIME_LINEAGE_DEFECT_20260821.md`. V25 restores the policy loader,
automatic contest response, unified prior and regional lineage, duplicate
general-identity suppression, and the applicable V23 automatic inputs.

The following V24 runtime paths remain unchanged:

- upper strategic-transfer conversion context;
- third-candidate profile;
- third-candidate pressure;
- ballot-faithful empty slot-keyed withdrawal registry.

The V23 automatic third-candidate pair was tested but rejected because stacking
it on V24's accepted weak-C response flipped the 2022 winner gate. The active
weak-C route remains `prediction_tilted`; `affinity_only` is not used.

## Historical score panel

| Election | candidates x regions | Regional weighted MAE | National candidate MAE | Winner |
|---|---:|---:|---:|---|
| 2002 | 48 | 2.7519%p | 2.3416%p | no |
| 2007 | 48 | 4.2771%p | 1.3053%p | yes |
| 2012 | 34 | 2.5955%p | 0.6408%p | yes |
| 2017 | 51 | 3.0253%p | 0.2014%p | yes |
| 2022 | 51 | 1.2200%p | 0.4591%p | yes |
| **Equal-election macro** | **232** | **2.7739%p** | **0.9896%p** | **4/5** |

V24 reported 2.7698%p regional and 1.0757%p national MAE on the same 232
rows. V25's regional headline is effectively unchanged while national
candidate allocation improves. These five elections remain a development
sample, not an untouched holdout.

## Predictive intervals

The downstream chronological calibration reports national candidate
vote-share predictive intervals, not Ridge coefficient confidence intervals.

| Nominal level | Historical equal-election coverage | Mean national width |
|---:|---:|---:|
| 50% | 83.33% | 4.50%p |
| 80% | 91.67% | 8.51%p |
| 90% | 91.67% | 10.90%p |
| 95% | 91.67% | 12.95%p |

Only four elections and eleven candidate outcomes are evaluable. Coverage is a
small historical calibration record, not a future guarantee.

## 2025 boundary

No 2025 realised outcome was used in the V25 repair, ablation, intervals, or
promotion. The separate V25 prospective run uses the 2025-06-02 D-1 cutoff,
reproduces all 232 frozen historical rows before emitting target rows, and
computes no performance metric.

## Reproduction

```bash
python scripts/run_active_presidential_model_v25.py --output-dir outputs/reproduction_v25
python scripts/build_active_v25_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v25.py
python scripts/run_prospective_forecast.py --version v25
python -m pytest -q
```

Canonical records are under `outputs/active_presidential_nested_v25/`. The
forecast-only record is under `outputs/prospective_pres_2025_v25/` and is not
part of historical scoring. The promotion-time full regression suite reports
`601 passed`.
