<!-- active-model-version: v32 -->
# Repository and distribution boundaries

The public repository keeps executable releases and research history together
for transparency, but they are not the same publication surface.

| Boundary | Canonical location | Install package | Meaning |
| --- | --- | --- | --- |
| Active engine | `src/election_forecast/`, `current_presidential_model.json`, reachable V31 engine modules and declared public/derived inputs | Yes | Locally runnable V31 code and inputs; neural runtime, weights, sentence corpus and direct overlay are excluded, while one disclosed frozen candidate-issue aggregate is retained |
| Frozen active evidence | `outputs/active_presidential_nested_v31/` | Yes | Immutable V31 predictions, intervals and promotion audits |
| Rollback evidence | V23-V30 frozen prediction CSVs | Prediction CSV only | Hash-verifiable rollback boundary, not active runtime logic |
| Corrected demonstration | `outputs/prospective_pres_2025_v31/` | Yes | D-1 demonstration repaired after the result was known; not prospective validation |
| Boundary history reference | `outputs/external_model_free_v25_baseline/` | Yes | V25 pipeline under the external-model boundary; verification reference only, never promoted or scored |
| Document record | `docs/` dated experiments, diagnoses and superseded final models | Yes | Indexed in `docs/INDEX.md`; accurate for the state each describes, never current |
| Historical research | `research/`, `docs/EXPERIMENT_*`, non-promoted experiment outputs | No | Methodological record; never presented as active V32 output |
| Local/private acquisition | ignored caches, checkpoints, full text, credentials and uncertain-rights exports | No | Must be reacquired under the provider's terms or remain local |

The wheel contains a hash-indexed runtime archive. The source distribution
contains the same runnable V31 source plus current publication, security and
rights documents. It deliberately excludes superseded visualization archives,
old corrected demonstrations and non-promoted output grids. The GitHub
repository remains the complete public research record.

## Promotion rule

A research change becomes active only through a new version. Promotion must:

1. preserve every earlier frozen prediction hash;
2. document development and outcome-awareness boundaries;
3. pass the complete regression suite, active-model audit, public-data audit,
   publication-security audit, distribution audit and clean reproduction;
4. update the current pointer, architecture/reproduction documents and current
   `vNN_` visualizations together.

No experiment may overwrite a frozen active directory or silently replace a
file inside an already released package.

## Frozen GitHub baselines

`scripts/audit_github_baseline.py` reads only the current one. The rest are the
frozen records of earlier promotions, and the four earliest predate the
versioned filename convention, so their version is in the file rather than the
name:

| file | active version |
| --- | --- |
| `docs/GITHUB_BASELINE_20260810.json` | v23 |
| `docs/GITHUB_BASELINE_20260820.json` | v24 |
| `docs/GITHUB_BASELINE_20260821.json` | v25 |
| `docs/GITHUB_BASELINE_20260822.json` | v26 |
| `docs/GITHUB_BASELINE_V27_20260822.json` | v27 |
| `docs/GITHUB_BASELINE_V28_20260823.json` | v28 |
| `docs/GITHUB_BASELINE_V29_20260823.json` | v29 |
| `docs/GITHUB_BASELINE_V30_20260824.json` | v30 |
| `docs/GITHUB_BASELINE_V31_20260825.json` | v31 |
| `docs/GITHUB_BASELINE_V32_20260826.json` | **v32, current** |

They are not renamed because frozen finalizer scripts reference them by path.
Every one carries its own `active_version` field, which is authoritative.
