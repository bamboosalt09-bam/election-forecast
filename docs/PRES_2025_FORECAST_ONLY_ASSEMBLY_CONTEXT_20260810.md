# 2025 Forecast-only Assembly Context

## Decision

The 2025 presidential election is a forecast-only demonstration target.

- scored and development elections remain 2002, 2007, 2012, 2017, and 2022;
- 1997 remains rolling warmup only;
- the 2025 result is forbidden from fitting, tuning, ablation, stage selection,
  comparison, and error calculation;
- the forecast cutoff is 2025-06-02, one day before the 2025-06-03 election.

The election date is supported by the [NEC election schedule](https://www.nec.go.kr/site/nec/ex/bbs/View.do?bcIdx=269345&cbIdx=1147).

## Official minutes collection

The local 15th-22nd Assembly corpus ended on 2024-12-31. It is now
supplemented from the [official National Assembly minutes system](https://record.assembly.go.kr/assembly/mnts/total/22.do).
The collector walks all six published classes: plenary, standing committee,
special committee, budget committee, inspection, and investigation.

Collection result:

| Item | Result |
|---|---:|
| Meeting window | 2025-01-01 to 2025-06-02 |
| Meetings discovered | 239 |
| Meetings completed and hashed | 239 |
| Duplicate meeting IDs | 0 |
| First actual meeting | 2025-01-06 |
| Last actual meeting | 2025-05-14 |
| XML attributed-speech sources | 238 |
| Official PDF text fallback | 1 |
| Derived issue rows | 48,588 |

Raw HTML, PDFs, per-meeting parts, and the atomic checkpoint are stored in
Git-ignored official-source cache directories. The small discovery, meeting,
collection, and model manifests remain auditable project inputs.

## Point-in-time rule

Meeting date is not treated as document availability. The official site does
not expose an exact first-publication timestamp, and some current PDFs were
created or regenerated well after their meeting dates. The collector therefore
uses the embedded official PDF `CreationDate` plus one full day as a
conservative day-level availability proxy. Missing metadata fails closed to
the collection date.

This rule produced:

| Item | Result |
|---|---:|
| Meetings eligible by 2025-06-02 | 91 |
| Meetings retained but excluded | 148 |
| Eligible derived rows | 14,985 |
| Post-cutoff derived rows excluded | 33,603 |

This policy is deliberately conservative. It prevents the current version of
a transcript generated after the election from entering the demo. It cannot
prove the exact first publication of an earlier version, because the official
site does not expose that field. The limitation is recorded rather than hidden.

## Model context build

The model context streams two compatible sources without copying the 5.8 GB
base file:

1. the verified 2022-03-22 to 2024-12-31 Assembly issue corpus;
2. the official 2025 H1 supplement collected above.

The combined build scanned 4,776,442 rows. It saw 570,665 `pres_2025` rows,
included 537,062 rows, excluded 33,603 post-cutoff rows, and found zero
duplicates. The source date range now reaches 2025-05-14; the latest eligible
meeting represented in model inputs is 2025-05-08.

Candidate linkage uses the complete official seven-candidate registry rather
than selecting the eventual top three. Candidate identity linkage becomes
available only when the dated registry is available. Candidate status,
withdrawal, votes, rank, winner, and vote share are not read.

The forecast-only loader is
`presidential_issue_engine.forecast_only_inputs.load_forecast_only_assembly_inputs`.
It verifies output hashes, rejects outcome fields, and reapplies the central
point-in-time filter before returning salience and candidate-link frames.

## Reproduction

```powershell
python scripts\collect_pres_2025_official_minutes.py

python scripts\build_pres_2025_assembly_context.py `
  --source "<EXTERNAL_CORPUS>\assembly_stance_rows_15_22.csv" `
  --supplement-source "data\raw\official_sources\assembly_pres_2025_minutes\assembly_stance_rows_2025_h1.csv"

python scripts\audit_pres_2025_demo_boundary.py
python scripts\audit_active_presidential_model_v23.py
```

Current audits:

- 2025 demo boundary: PASS;
- active V23 historical invariance: PASS, 215/215 rows;
- 2025 outcome fields used: none;
- 2025 performance metrics computed: false.

The 33 MB row-level 2025 supplement is intentionally Git-ignored. Its hash,
meeting-level manifests, collector, sufficient-statistic context files, and
reproduction command are tracked; raw HTML, PDFs, and the 5.8 GB base corpus
remain external data.

## Do not claim

- Do not report a 2025 MAE or compare a 2025 prediction with the realized vote.
- Do not call all 239 meetings pre-election inputs; only 91 pass the strict
  availability proxy.
- Do not treat shadow stance classifiers as active candidate vote signals.
- Do not use the 2025 result to choose issue weights, postprocessing gains, or
  the model version shown in the demonstration.
