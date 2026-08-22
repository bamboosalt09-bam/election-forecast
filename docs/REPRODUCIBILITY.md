# Reproducibility and Frozen V27 Boundary

## Frozen scope

Active V27 is a through-2022 development model. Its runner, V23 base
configuration, V24 versioned panel, predictions, interval records, and
promotion manifests are frozen. V26 through V23 remain immutable rollback
boundaries. New experiments must use a new versioned path.

Canonical prediction artifacts:

```text
V27 active:
outputs/active_presidential_nested_v27/nested_predictions.csv
SHA-256: f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b

V26 rollback:
outputs/active_presidential_nested_v26/nested_predictions.csv
SHA-256: 9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3

V24 rollback:
outputs/active_presidential_nested_v24/nested_predictions.csv
SHA-256: edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a

V23 rollback:
outputs/active_presidential_nested_v23/nested_predictions.csv
SHA-256: dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b
```

`data/config/current_presidential_model.json` selects V27. V27 uses the V26
runtime and adds the fixed gain-1 core-weighted regional dispersion layer.

## Reproduce the checks

```bash
python scripts/run_active_presidential_model_v27.py --output-dir outputs/reproduction_v27
python scripts/verify_v27_clean_reproduction.py
python scripts/build_active_v27_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v27.py
python scripts/audit_github_baseline.py
python -m pytest -q
```

The public audit checks pointer fields, V23~V26 rollback hashes, V27 prediction
and artifact hashes, compositional rows, gain 1, and chronological intervals.
The clean-reproduction command rebuilds predictions in a temporary directory
and requires their raw-byte SHA-256 to equal the frozen V27 artifact.

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
python scripts/run_prospective_forecast_v27.py
```

The run uses the 2025-06-02 D-1 cutoff and prior-2022 regional vote volumes for
V27's conservation weights. The manifest asserts that no 2025 outcome field or
performance metric was used.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election, then averages elections equally. The
national metric also uses realised regional contest votes and is therefore a
post-election aggregation diagnostic. V27 through V24 share the 232-row panel;
V23's older headline uses a different 199-row panel.
