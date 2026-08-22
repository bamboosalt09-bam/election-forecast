# Reproducibility and Frozen V28 Boundary

## Frozen scope

Active V28 is a through-2022 development model. Its runner, V23 base
configuration, V24 versioned panel, predictions, interval records, and
promotion manifests are frozen. V27 through V23 remain immutable rollback
boundaries. New experiments must use a new versioned path.

Canonical prediction artifacts:

```text
V28 active:
outputs/active_presidential_nested_v28/nested_predictions.csv
SHA-256: f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b

V27 rollback (prediction-equivalent predecessor):
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

`data/config/current_presidential_model.json` selects V28, and
`data/config/active_presidential_model.json` is an identical public
compatibility alias. The explicit `active_presidential_model_v16.json` file is
only the frozen internal base used by the versioned runner lineage. V28 keeps
V27's statistical chain but disables neural inference and excludes the direct
sentence-level stance overlay. The disclosed frozen historical
`candidate_issue_profile.csv` remains because full removal materially changes
the postprocess. Historical predictions remain byte-identical.

## Reproduce the checks

### Installed wheel (source checkout not required)

```bash
python -m pip install election_forecast-0.28.0.dev0-py3-none-any.whl
election-forecast audit-current-presidential
election-forecast verify-current-presidential
election-forecast run-current-presidential --output-dir outputs/reproduction_v28
```

The wheel embeds the Git-tracked public V28 runtime as a deterministic archive.
On first use it rejects path traversal, extracts into a versioned user cache and
verifies every file's size and SHA-256 before executing the V28
runner. Local caches, credentials, full-text corpora and the uncertain-rights
KOSPI export are excluded.

### Source checkout

```bash
python -m pip install -e ".[dev,reproduce-v28]"
python scripts/run_active_presidential_model_v28.py --output-dir outputs/reproduction_v28
python scripts/verify_v28_clean_reproduction.py
python scripts/build_active_v28_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v28.py
python scripts/audit_github_baseline.py
python scripts/audit_public_data_rights.py
python scripts/audit_publication_security.py
python -m pytest -q
```

The public audit checks pointer fields, V23~V27 rollback hashes, V28 prediction
and artifact hashes, compositional rows, gain 1, and chronological intervals.
The clean-reproduction command rebuilds predictions in a temporary directory.
It always pins the stored frozen artifact's raw-byte SHA-256. Rebuilt tables
must have identical shape, column order and categorical values. On the original
development machine the rebuilt file is byte-identical. Cross-hardware CI
requires the final `layer_pred` within `0.001` share (`0.10%p`) and every
numeric diagnostic within `0.0012` share (`0.12%p`). The rebuilt byte hash and
observed maxima are always reported; this tolerance is not used to redefine
the frozen artifact.

Exact V28 regeneration uses Windows, Python 3.13, the `reproduce-v28`
optional dependency set and single-threaded BLAS (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` set to `1`). The ordinary package remains usable and tested on
Linux with Python 3.11+, but those broader combinations are supported for use
rather than claimed as the frozen numerical build environment. GitHub-hosted
Windows and Linux rebuilds with the same high-level dependency versions exposed
CPU/BLAS drift up to
`0.001174501524589` in an intermediate share-scale field, so the project does
not claim hardware-independent byte identity.
The recorded independent-clone run is in
`docs/V27_CLEAN_CLONE_VERIFICATION_20260822.md`.

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
python scripts/run_prospective_forecast_v28.py
```

The run uses the 2025-06-02 D-1 cutoff and prior-2022 regional vote volumes for
V28's conservation weights. The manifest asserts that no 2025 outcome field or
performance metric was used.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election, then averages elections equally. The
national metric also uses realised regional contest votes and is therefore a
post-election aggregation diagnostic. V28 through V24 share the 232-row panel;
V23's older headline uses a different 199-row panel.
