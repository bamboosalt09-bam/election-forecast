<!-- active-model-version: v29 -->
# Reproducibility and the Frozen V29 Boundary

## What is and is not reproducible from this repository

| artifact | reproducible from the public tree | check |
| --- | --- | --- |
| V29 scored historical model | **yes**, byte for byte | `clean-reproduction` |
| V29 packaged runtime | **yes**, from the built wheel | `wheel-reproduction` |
| 2025 D-1 demonstration | **yes, with one boundary** | `prospective-reproduction` |

The 2025 demonstration is built from official Assembly proceedings. Its
collected form carries verbatim excerpts and is not redistributed, and the
historical speaker-issue matches it needs lived only under `archives/`, which
the repository boundary forbids tracking. Three derived files ship in their
place - the 2025 rows with each excerpt replaced by its character count, the
output of the keyword matching that does read the words, and the historical
matches gzipped. Together they are 3.7 MB against 154 MB of untracked inputs,
and none carries a source sentence.

So the boundary is this: from the public tree the forecast is rebuilt and
compared **downstream of the keyword matching**, and the matching itself is
taken as given. Confirming that these proceedings produce these issue weights
needs the proceedings; confirming that these issue weights produce this forecast
does not. `verify_v29_prospective_reproduction.py` reports which of the two it
did rather than letting a weaker reproduction look like a stronger one.

[PRES_2025_INPUT_GUIDE.md](PRES_2025_INPUT_GUIDE.md) gives the full procedure for
collecting the proceedings and recomputing every derived file from them.

None of this was known until the 2025 path was given a CI job; no job had run
it before.

## Frozen scope

Active V29 is a through-2022 development model. Its runner, V23 base
configuration, V24 versioned panel, predictions, interval records, and
promotion manifests are frozen. V28 through V23 remain immutable rollback
boundaries. New experiments must use a new versioned path.

Canonical prediction artifacts:

```text
V29 active:
outputs/active_presidential_nested_v29/nested_predictions.csv
SHA-256: fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904

V28 rollback (pre-expansion predecessor):
outputs/active_presidential_nested_v28/nested_predictions.csv
SHA-256: 23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba

V27 rollback:
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

`data/config/current_presidential_model.json` selects V29, and
`data/config/active_presidential_model.json` is an identical public
compatibility alias. The explicit `active_presidential_model_v16.json` file is
only the frozen internal base used by the versioned runner lineage. V29 keeps
V28's statistical chain and its external-model boundary — no neural inference,
no direct sentence-level stance overlay, and the two excluded automatic mega
seeds blocked even when a legacy engine loads after the wrapper enters its
runtime guard — and adds a third-share-indexed expansion of the regional
dispersion. The disclosed frozen historical `candidate_issue_profile.csv`
remains because full removal materially changes the postprocess.

## Reproduce the checks

### Installed wheel (source checkout not required)

```bash
python -m pip install election_forecast-0.29.0.dev0-py3-none-any.whl
election-forecast audit-current-presidential
election-forecast verify-current-presidential
election-forecast run-current-presidential --output-dir outputs/reproduction_v29
```

The wheel embeds the Git-tracked public V29 runtime as a deterministic archive.
On first use it rejects path traversal, extracts into a versioned user cache and
verifies every file's size and SHA-256 before executing the V29
runner. Local caches, credentials, full-text corpora and the uncertain-rights
daily KOSPI export are excluded. The wheel includes only the 15 attributed D-1
election×slot aggregates actually consumed by V29.

### Source checkout

```bash
python -m pip install -e ".[dev,reproduce-v29]"
python scripts/run_active_presidential_model_v29.py --output-dir outputs/reproduction_v29
python scripts/verify_v29_clean_reproduction.py
python scripts/build_active_v29_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v29.py
python scripts/audit_github_baseline.py
python scripts/audit_public_data_rights.py
python scripts/audit_publication_security.py
python -m pytest -q
```

The public audit checks pointer fields, V23~V28 rollback hashes, V29 prediction
and artifact hashes, compositional rows, gain 1, and chronological intervals.
The clean-reproduction command rebuilds predictions in a temporary directory.
It always pins the stored frozen artifact's raw-byte SHA-256. Rebuilt tables
must have identical shape, column order and categorical values. On the original
development machine the rebuilt file is byte-identical. Cross-hardware CI
requires the final `layer_pred` within `0.001` share (`0.10%p`) and every
numeric diagnostic within `0.0012` share (`0.12%p`). The rebuilt byte hash and
observed maxima are always reported; this tolerance is not used to redefine
the frozen artifact.

Exact V29 regeneration uses Windows, Python 3.13, the `reproduce-v29`
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
python scripts/run_prospective_forecast_v29.py
```

The run uses the 2025-06-02 D-1 cutoff and prior-2022 regional vote volumes for
V29's conservation weights. The manifest asserts that no 2025 outcome field or
performance metric was used.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election, then averages elections equally. The
national metric also uses realised regional contest votes and is therefore a
post-election aggregation diagnostic. V29 through V24 share the 232-row panel;
V23's older headline uses a different 199-row panel.
