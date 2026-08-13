# Reproducibility and Frozen V23 Boundary

## Frozen scope

Active V23 is a through-2022 development model. Its runner, configuration,
canonical inputs, predictions, and promotion records are frozen. Experiments
must use a new versioned path and must not edit V23 in place.

The canonical prediction artifact is:

```text
outputs/active_presidential_nested_v23/nested_predictions.csv
SHA-256: dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b
```

The active pointer remains `data/config/current_presidential_model.json` and
continues to select V23. A V24 experiment is not active unless a person reviews
it and performs a separate promotion change.

## Reproduce the checks

Install the test and visualization extras first with
`python -m pip install -e ".[dev,viz]"`. A public clone can then run these three
verification commands:

```bash
python scripts/audit_github_baseline.py
python scripts/audit_public_active_presidential_model_v23.py
python -m pytest -q
```

The canonical local audit command is
`python scripts/audit_active_presidential_model_v23.py`. It additionally checks
the bytes of a frozen external input that is intentionally not redistributed
in Git. The public wrapper checks its signed metadata record when those bytes
are absent and otherwise runs the same V23 invariance audits.

## Why two manifest hashes differ

`outputs/active_presidential_nested_v23/finalization_manifest.json` records the
state at the time V23 was frozen. Its `artifacts` array contains two living
documents, `README.md` and `docs/HANDOFF_CURRENT_STATE.md`. Those documents were
updated after model finalization, so their current hashes differ from the
historical values in the manifest.

The other 14 artifact records match their files byte for byte. They cover the
model specification, active pointer, configuration, runners and audit, exact
input records, predictions, summary, and promotion record. These 14 artifacts
are the substantive frozen model boundary.

The two document hashes must not be rewritten in the finalization manifest.
Preserving the original record is what makes later documentation changes
visible and auditable; changing it after the fact would erase that evidence.

## Recorded workspace field

The manifest's `workspace` field contains the local development path that was
active at finalization. It is provenance, not a runtime requirement. It is left
unchanged for the same reason as the historical hashes: the finalization record
must describe the event as it occurred. All maintained code resolves paths from
the repository root and does not require that recorded path.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election and then averages elections equally. The
national metric also uses realized regional contest votes and is therefore a
post-election aggregation diagnostic. Neither metric is an untouched holdout:
the five scored elections are the through-2022 development sample.
