# Reproducibility and Frozen V25 Boundary

## Frozen scope

Active V25 is a through-2022 development model. Its runner, V23 base
configuration, V24 versioned panel, predictions, interval records, and
promotion manifests are frozen. V24 and V23 remain immutable rollback
boundaries. New experiments must use a new versioned path.

Canonical prediction artifacts:

```text
V25 active:
outputs/active_presidential_nested_v25/nested_predictions.csv
SHA-256: 218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b

V24 rollback:
outputs/active_presidential_nested_v24/nested_predictions.csv
SHA-256: edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a

V23 rollback:
outputs/active_presidential_nested_v23/nested_predictions.csv
SHA-256: dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b
```

`data/config/current_presidential_model.json` selects V25. V25 uses the V23
JSON configuration and V24 ballot panel, then restores the bounded runtime
bindings listed in `docs/V24_RUNTIME_LINEAGE_DEFECT_20260821.md`.

## Reproduce the checks

```bash
python scripts/run_active_presidential_model_v25.py --output-dir outputs/reproduction_v25
python scripts/build_active_v25_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v25.py
python scripts/audit_github_baseline.py
python -m pytest -q
```

The public audit checks pointer fields, V23/V24 rollback hashes, V25 input
hashes, compositional rows, chronological intervals, the accepted
`prediction_tilted` weak-C route, and finalized artifact hashes.

## Predictive intervals

`national_predictive_intervals.csv` contains national candidate vote-share
predictive intervals at 50%, 80%, 90%, and 95%. They are not coefficient
confidence intervals. Every target fold uses only earlier-election residuals
and regional vote-volume transitions to construct its bounds; the target result
is read afterward only for historical coverage.

Four target elections and eleven candidate outcomes are evaluable. This is a
small historical calibration record, not a future coverage guarantee or an
untouched holdout.

## 2025 prospective run

```bash
python scripts/run_prospective_forecast.py --version v25
```

The run uses the 2025-06-02 D-1 cutoff. Before emitting 51 target rows it must
reproduce all 232 frozen V25 historical rows within `1e-12`. The manifest
asserts that no 2025 outcome field or performance metric was used.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election, then averages elections equally. The
national metric also uses realised regional contest votes and is therefore a
post-election aggregation diagnostic. V25 and V24 share the 232-row panel;
V23's older headline uses a different 199-row panel.
