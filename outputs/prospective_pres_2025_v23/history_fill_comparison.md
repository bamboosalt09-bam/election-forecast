# pres_2025 Speech-context History Fill Comparison

This record compares two prospective runs only. It does not contain, read, or
compare against the 2025 election result.

| Item | Adapter (before) | Speech-derived context (after) |
|---|---:|---:|
| National prediction | Kim Moon-soo 39.1202% / Lee Jae-myung 37.3354% / Lee Jun-seok 23.5444% | Kim Moon-soo 38.5672% / Lee Jae-myung 37.2815% / Lee Jun-seok 24.1513% |
| Lee Jun-seok regional sample standard deviation | 2.582%p | 2.678%p |
| Lee Jae-myung regional sample standard deviation | 18.559%p | 18.195%p |
| Kim Moon-soo regional sample standard deviation | 19.879%p | 19.556%p |
| Candidate weight (Lee Jae-myung / Kim Moon-soo / Lee Jun-seok) | 0.5386 / 0.5376 / 0.3672 | 0.5777 / 0.4984 / 0.4561 |
| Candidate-strength method | 12-row ridge projection | Direct speech-derived candidate context |
| pres_2025 speaker-issue match rows | 0 | 2,446 |

## Source boundary

The supplied `trash_dataset.zip` ends on 2024-12-31, so it cannot produce a
2025 match row. The after run supplements that historical corpus with the
repository's official National Assembly minutes collection. The supplement is
restricted to meetings from 2025-03-05 through 2025-06-02 and to documents
whose conservative `available_date` is no later than 2025-06-02. The latest
eligible meeting represented is 2025-05-08.

The historical 2002-2022 match region remains cell-for-cell identical after
normalizing the outer ZIP wrapper path. The four candidate-context CSVs retain
their original 13 rows as an exact byte prefix and append three `pres_2025`
rows.

The transformed 2,446-row slice is published at
`data/raw/official_sources/assembly_pres_2025_context/pres_2025_speaker_issue_matches.csv`;
its source and selection boundary are recorded in the adjacent
`speaker_issue_match_manifest.json`.

## Outcome boundary

- `outcome_columns_used: []`
- `performance_metrics_computed: false`
- `pres_2025_outcome_present: false`
- Forecast cutoff: `2025-06-02`
