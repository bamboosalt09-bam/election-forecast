<!-- active-model-version: v32 -->
# Reproducibility and the Frozen V32 Boundary

## What is and is not reproducible from this repository

| artifact | reproducible from the public tree | check |
| --- | --- | --- |
| V32 scored historical model | **yes**, within the tolerance below | `clean-reproduction` |
| V32 packaged runtime | **yes**, from the built wheel | `wheel-reproduction` |
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
does not. `verify_v32_prospective_reproduction.py` reports which of the two it
did rather than letting a weaker reproduction look like a stronger one.

[PRES_2025_INPUT_GUIDE.md](PRES_2025_INPUT_GUIDE.md) gives the full procedure for
collecting the proceedings and recomputing every derived file from them.

None of this was known until the 2025 path was given a CI job; no job had run
it before.

## Frozen scope

Active V32 is a through-2022 development model. Its runner, V23 base
configuration, V24 versioned panel, predictions, interval records, and
promotion manifests are frozen. V31 through V23 remain immutable rollback
boundaries. New experiments must use a new versioned path.

Canonical prediction artifacts:

```text
V32 active:
outputs/active_presidential_nested_v32/nested_predictions.csv
SHA-256: 969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069

V31 rollback (multiplicative expansion; byte-identical to V32):
outputs/active_presidential_nested_v31/nested_predictions.csv
SHA-256: 969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069

V30 rollback (additive expansion with a zero cap):
outputs/active_presidential_nested_v30/nested_predictions.csv
SHA-256: afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e

V29 rollback (target-turnout-weighted predecessor):
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

Note that the active and the newest rollback hash to the same value. That is
V32's central claim rather than an error: V32 changes only the assembly that
builds the target election's features, and the five scored elections do not take
that path. It also means the hash-consistency check in
`audit_version_consistency` cannot distinguish the two here, so the *label* on
the block above is not defended by it — which is how this document came to say
"V31 active" under a V32 pointer while every hash check passed.

`data/config/current_presidential_model.json` selects V32, and
`data/config/active_presidential_model.json` is an identical public
compatibility alias. The explicit `active_presidential_model_v16.json` file is
only the frozen internal base used by the versioned runner lineage. V32 keeps
V31's statistical chain, its terminal transforms and its external-model boundary
— no neural inference, no direct sentence-level stance overlay, and the
automatic mega seeds refused rather than read. What it changes is the
prospective assembly: every column the target frame lacks is classified and
built, declared zero with a reason, or fatal, instead of silently filled with a
zero. The disclosed frozen historical `candidate_issue_profile.csv` remains
because full removal materially changes the postprocess.

## Reproduce the checks

### Installed wheel (source checkout not required)

```bash
python -m pip install election_forecast-0.32.0.dev0-py3-none-any.whl
election-forecast audit-current-presidential
election-forecast verify-current-presidential
election-forecast run-current-presidential --output-dir outputs/reproduction_v32
```

The wheel embeds the Git-tracked public V32 runtime as a deterministic archive.
On first use it rejects path traversal, extracts into a versioned user cache and
verifies every file's size and SHA-256 before executing the V32
runner. Local caches, credentials, full-text corpora and the uncertain-rights
daily KOSPI export are excluded. The wheel includes only the 15 attributed D-1
election×slot aggregates actually consumed by V32.

### Source checkout

```bash
python -m pip install -e ".[dev,reproduce-v32]"
python scripts/run_active_presidential_model_v32.py --output-dir outputs/reproduction_v32
python scripts/verify_v32_clean_reproduction.py
python scripts/build_active_v32_predictive_intervals.py
python scripts/audit_public_active_presidential_model_v32.py
python scripts/audit_github_baseline.py
python scripts/audit_public_data_rights.py
python scripts/audit_publication_security.py
python -m pytest -q
```

The public audit checks pointer fields, V23~V31 rollback hashes, V32 prediction
and artifact hashes, compositional rows, gain 1, and chronological intervals.

### The reproduction tolerance, stated once

The clean-reproduction command rebuilds predictions in a temporary directory.
Rebuilt tables must have identical shape, column order and categorical values —
a text column has no tolerance to spend — and every numeric column must agree
with the frozen artifact within:

```text
atol = 1e-12   rtol = 0
```

That is what `verify_v32_clean_reproduction.py`,
`verify_v32_prospective_reproduction.py` and the V32 runner all enforce, and it
is the only tolerance this project asks a third party to meet. An earlier
edition of this document described a far looser cross-hardware contract —
`0.001` share on `layer_pred` and `0.0012` on diagnostics — which no check has
enforced for several versions. It is withdrawn rather than carried forward: a
reader asking what the tolerance is must get one answer, and the enforced one is
above.

The rebuilt byte hash is always reported alongside. On the original development
machine the rebuilt file is byte-identical to the frozen artifact. It is **not**
byte-identical everywhere: the GitHub-hosted Windows runner produces one of two
stable results, and the alternative differs by `1.388e-13` — inside the
tolerance, and recorded rather than tolerated silently. Byte identity is
therefore asserted of the *committed* artifact, in the audit and the
finalization manifest, and never demanded of a rebuild.

A historical measurement, kept as history rather than as a contract: GitHub
Windows and Linux rebuilds under V27's dependency set showed CPU/BLAS drift up
to `0.001174501524589` in an intermediate share-scale field, recorded in
`docs/V27_CLEAN_CLONE_VERIFICATION_20260822.md`. Were that to recur today the
checks above would fail rather than accept it. The project does not claim
hardware-independent byte identity.

### The frozen numerical build environment

Exact V32 regeneration uses Windows, Python 3.13, the `reproduce-v32`
optional dependency set and single-threaded BLAS (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` set to `1`). The ordinary package remains usable and tested on
Linux with Python 3.11+, but those broader combinations are supported for use
rather than claimed as the frozen numerical build environment.

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
python scripts/run_prospective_forecast_v32.py
```

The run uses the 2025-06-02 D-1 cutoff and prior-2022 regional vote volumes for
V32's conservation weights, which are V31's unchanged. The manifest asserts that no 2025 outcome field or
performance metric was used.

## Metric scope

The primary regional metric weights candidate-region absolute errors by
`contest_votes` within each election, then averages elections equally. The
national metric also uses realised regional contest votes and is therefore a
post-election aggregation diagnostic. This is the *metric's* weighting only:
since V30 no transform reads the target election's turnout, so the diagnostic
weighting enters no prediction. V32 through V24 share the 232-row panel;
V23's older headline uses a different 199-row panel.
