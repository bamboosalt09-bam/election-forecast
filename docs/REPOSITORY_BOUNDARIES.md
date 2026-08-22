# Repository and distribution boundaries

The public repository keeps executable releases and research history together
for transparency, but they are not the same publication surface.

| Boundary | Canonical location | Install package | Meaning |
| --- | --- | --- | --- |
| Active engine | `src/election_forecast/`, `current_presidential_model.json`, reachable V28 engine modules and declared public/derived inputs | Yes | Locally runnable V28 code and inputs; external-model-derived overlays are excluded |
| Frozen active evidence | `outputs/active_presidential_nested_v28/` | Yes | Immutable V28 predictions, intervals and promotion audits |
| Rollback evidence | V23-V27 frozen prediction CSVs | Prediction CSV only | Hash-verifiable rollback boundary, not active runtime logic |
| Corrected demonstration | `outputs/prospective_pres_2025_v28/` | Yes | D-1 demonstration repaired after the result was known; not prospective validation |
| Historical research | `research/`, `docs/EXPERIMENT_*`, non-promoted experiment outputs | No | Methodological record; never presented as active V28 output |
| Local/private acquisition | ignored caches, checkpoints, full text, credentials and uncertain-rights exports | No | Must be reacquired under the provider's terms or remain local |

The wheel contains a hash-indexed runtime archive. The source distribution
contains the same runnable V28 source plus current publication, security and
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
