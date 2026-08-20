# Reproducibility and Frozen V24 Boundary

## Frozen scope

Active V24 is a through-2022 development model. Its version wrapper, V23 base
configuration, versioned V24 inputs, predictions, interval records, and
promotion manifests are frozen. V23 remains an immutable rollback boundary.
Experiments must use a new versioned path and must not edit either frozen output
directory in place.

The canonical V24 prediction artifact is:

```text
outputs/active_presidential_nested_v24/nested_predictions.csv
SHA-256: edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a
```

The preserved V23 rollback artifact is:

```text
outputs/active_presidential_nested_v23/nested_predictions.csv
SHA-256: dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b
```

`data/config/current_presidential_model.json` selects V24. V24 deliberately
uses the frozen V23 JSON configuration as its base and applies its additional
ballot, scored-scope, lineage, veto, and same-lane rules in the versioned V24
runner. The pointer records both facts explicitly.

## Reproduce the checks

Install the test extras and run:

```bash
python scripts/build_active_v24_predictive_intervals.py
python scripts/audit_github_baseline.py
python scripts/audit_public_active_presidential_model_v24.py
python -m pytest -q
```

To reproduce the point model without overwriting the frozen output, pass a new
directory:

```bash
python scripts/run_active_presidential_model_v24.py --output-dir outputs/reproduction_v24
```

The public audit verifies local bytes when they are present. For intentionally
unredistributed bulk inputs, it verifies the path, byte count, SHA-256 record,
Git exclusion, and ignore rule in
`data/raw/official_sources/external_active_inputs.json`.

## Predictive intervals

`national_predictive_intervals.csv` contains national candidate vote-share
predictive intervals at 50%, 80%, 90%, and 95%. They are not coefficient
confidence intervals. For each target election from 2007 onward, the bounds use
only point errors and regional vote-volume transitions from earlier elections.
The target result is consulted only after the bounds are fixed to calculate
historical coverage. The unscaled residual multiplier is fixed at 1.0 rather
than selected against coverage.

The interval evaluation contains four target elections and eleven candidate
outcomes. It is therefore a small historical calibration record, not a promise
of nominal future coverage. The point-model rules were themselves developed on
the through-2022 sample, so the interval record is not an untouched holdout.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election and then averages elections equally. The
national metric also uses realized regional contest votes and is therefore a
post-election aggregation diagnostic. V24 restores weak third-candidate rows,
so its 232-row score panel is not identical to V23's 199-row panel.

No 2025 result, row, or post-cutoff artifact is used in the V24 point model,
interval fitting, interval calibration, promotion comparison, or finalization.
