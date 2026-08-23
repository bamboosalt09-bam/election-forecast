# Data provenance and redistribution register

This register distinguishes project-authored derived tables from externally
sourced records. Apache-2.0 covers project code and documentation only. It does
not relicense source data.

| Data family | Repository paths | Primary source | Redistribution status |
| --- | --- | --- | --- |
| Election and candidate records | `data/raw/official_sources/nec_*`, candidate registry/history and derived ballot tables | National Election Commission Open API / Public Data Portal | Included as factual public-data extracts and project-derived audits with NEC attribution. This decision does not extend to unrelated NEC publications without an explicit KOGL/open-data indication. |
| Historical National Assembly members and proceedings | derived salience/link tables and historical source hashes | National Assembly Library, National Academic Information Cloud legislative datasets | The source XLSX/ZIP files were downloaded from the official legislative-dataset catalogue. V29 distributes only compact project-derived aggregates with attribution; source workbooks, full transcripts and sentence corpora are excluded. The provider terms require attribution and prohibit selling or sharing the supplied information unchanged without a service layer. |
| 2025 official proceedings | `data/raw/official_sources/assembly_pres_2025_*` manifests and aggregates | National Assembly minutes service (`record.assembly.go.kr`) | Official XML/PDF URLs, availability dates and hashes are recorded. Source XML/PDF and full extracted text are excluded; compact manifests and derived aggregates are included with attribution. |
| Macroeconomic series | `presidential_issue_engine/fixed_dataset/economic_indicators.csv`, `interest_rate_indicators.csv` | Bank of Korea ECOS | Included derived numeric extracts. The ECOS Open API is listed by the Public Data Portal with no use restriction; retain BOK attribution. |
| Housing series | `presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv`, `housing_price_index_sgg.csv` | Korea Real Estate Board / KOSIS | Included derived regional series with attribution. KOSIS permits commercial and noncommercial reuse; unmodified downloaded tables must not be resold as such. |
| Market series | `data/raw/official_sources/bok_ecos_kospi_manifest.json`, `presidential_issue_engine/fixed_dataset/kospi_election_context.csv` | Bank of Korea ECOS / Korea Exchange | The local row-level KOSPI history and ECOS cache are Git-ignored and excluded from the wheel. V29 publishes only 15 D-1 election×slot context rows needed by the model, with source attribution and no daily source table. |
| Administrative boundaries | downloaded at visualization build time; generated map PNGs under `presidential_issue_engine/poster_figures` | `vuski/admdongkor`, SGIS-derived 2025-04-01 snapshot | CC BY 4.0 and Korea Open Government License Type 1 attribution. Pinned source, hash and terms are in `VISUALIZATION_DATA.md`. |
| Project-authored annotations and derived controls | `data/raw/auto_issue_seed`, most `presidential_issue_engine/fixed_dataset` policy/lineage tables, generated audits | Derived by this project from the source families above | Transformation code and authored schema/notes are Apache-2.0. Underlying facts and excerpts retain source restrictions; redistribution is safe only when no restricted source content is reproduced. |
| News-source configuration | `data/config/source_list.csv`, `feeds.csv` | Publisher URLs and RSS endpoints | Configuration metadata is project-authored. Article text is not relicensed and must not be redistributed without permission. |
| Empty input templates | `presidential_issue_engine/fixed_dataset/templates` and zero-row CSV schemas | Project-authored | Redistributable under Apache-2.0. |
| Model outputs and audits | `outputs/active_presidential_nested_v23` through `v29`, prospective manifests, evaluation extracts | Project calculations over the above inputs | Project-authored calculations may be reused with attribution, but embedded names, factual records and any source-derived excerpts remain subject to their source terms. Frozen versions must not be modified in place. V29 retains one disclosed compact external-model-derived candidate-issue aggregate; model weights, source sentences and the direct overlay are excluded. |

## Practical reuse rule

Green-light reuse applies to source code, documentation, empty templates and
purely project-authored formulas. Attribution-required reuse applies to the map
and included official-data-derived tables. Raw KOSPI rows, full parliamentary
text, source HTML/PDF, publisher text and collection caches are not part of the
public package and must not be added without a new rights decision.

This register is a provenance and project-policy record, not legal advice. A
future data refresh must add its source URL, access date, available date,
license/terms URL, content hash and redistribution decision before promotion.

## Machine-readable source evidence

Detailed row counts and PIT coverage are available with:

```bash
python scripts/describe_inputs.py inventory
python scripts/describe_inputs.py sources pres_2025
```

`PUBLIC_DATA_SOURCES.json` is the machine-readable publication allowlist.
`python scripts/audit_public_data_rights.py` fails if an excluded source is
tracked or if a public data path has no registered source/terms coverage.
