# Data provenance and redistribution register

This register distinguishes project-authored derived tables from externally
sourced records. Apache-2.0 covers project code and documentation only. It does
not relicense source data.

| Data family | Repository paths | Primary source | Redistribution status |
| --- | --- | --- | --- |
| Election and candidate records | `data/raw/official_sources/nec_*`, candidate registry/history and derived ballot tables | National Election Commission and Public Data Portal APIs | Source facts are publicly accessible, but no blanket data-license grant is recorded for every snapshot. Keep attribution and source URLs; verify portal terms before redistributing a newly downloaded bulk file. |
| National Assembly members and proceedings | `data/raw/official_sources/assembly_*`, `assembly_pres_2025_minutes`, reconstructed stance/context tables | National Assembly Open Data Portal and minutes viewers | Public records; repository copies and derived excerpts must retain source manifests. Bulk republication rights have not been independently certified, so do not treat them as Apache-2.0 data. |
| Macroeconomic series | `presidential_issue_engine/fixed_dataset/economic_indicators.csv`, `interest_rate_indicators.csv` | Bank of Korea ECOS / National Income | Derived numeric extracts with source notes. Attribution required; consult ECOS terms before republishing a refreshed bulk extract. |
| Housing series | `presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv` | Korea Real Estate Board / KOSIS | Derived regional aggregates. Source and table identifier must remain; refreshed source data remain under provider terms. |
| Market series | `presidential_issue_engine/fixed_dataset/kospi_daily.csv` | User-provided historical KOSPI source noted in the file | Provenance is recorded but redistribution permission is not established. This is the highest-priority unresolved data-rights item; do not redistribute it separately as an open dataset. |
| Administrative boundaries | downloaded at visualization build time; generated map PNGs under `presidential_issue_engine/poster_figures` | `vuski/admdongkor`, SGIS-derived 2025-04-01 snapshot | CC BY 4.0 and Korea Open Government License Type 1 attribution. Pinned source, hash and terms are in `VISUALIZATION_DATA.md`. |
| Project-authored annotations and derived controls | `data/raw/auto_issue_seed`, most `presidential_issue_engine/fixed_dataset` policy/lineage tables, generated audits | Derived by this project from the source families above | Transformation code and authored schema/notes are Apache-2.0. Underlying facts and excerpts retain source restrictions; redistribution is safe only when no restricted source content is reproduced. |
| News-source configuration | `data/config/source_list.csv`, `feeds.csv` | Publisher URLs and RSS endpoints | Configuration metadata is project-authored. Article text is not relicensed and must not be redistributed without permission. |
| Empty input templates | `presidential_issue_engine/fixed_dataset/templates` and zero-row CSV schemas | Project-authored | Redistributable under Apache-2.0. |
| Model outputs and audits | `outputs/active_presidential_nested_v23` through `v27`, prospective manifests, evaluation extracts | Project calculations over the above inputs | Project-authored calculations may be reused with attribution, but embedded names, factual records and any source-derived excerpts remain subject to their source terms. Frozen versions must not be modified in place. |

## Practical reuse rule

Green-light reuse applies to source code, documentation, empty templates and
purely project-authored formulas. Attribution-required reuse applies to the map
geometry and generated map. For official-record extracts, economic series,
KOSPI history and text excerpts, the conservative default is **source terms
must be checked before separate bulk redistribution**.

This register is a provenance and project-policy record, not legal advice. A
future data refresh must add its source URL, access date, available date,
license/terms URL, content hash and redistribution decision before promotion.

## Machine-readable source evidence

Detailed row counts and PIT coverage are available with:

```bash
python scripts/describe_inputs.py inventory
python scripts/describe_inputs.py sources pres_2025
```

Official-source manifests under `data/raw/official_sources` retain source URLs,
hashes and collection metadata where available. Missing licensing metadata is
an unresolved restriction, not permission to redistribute.

