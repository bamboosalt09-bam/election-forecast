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

## Candidate conversion history fill

The 16th-22nd Assembly workbook archive ends on 2024-12-31. Its historical
speaker-issue extraction was rerun and reproduced all 195,758 rows for the five
scored elections after normalizing the outer ZIP wrapper path. The 2025 segment
therefore comes from the official minutes supplement rather than from the
workbook archive.

To keep the candidate conversion layer methodologically aligned with the
historical campaign extracts, the supplement is restricted further to the
2025 D-90 meeting window and the conservative D-1 availability rule. This
produces 2,446 speaker-issue rows from 2025-03-05 through 2025-05-08. The
text-free match slice and its selection manifest are tracked at:

- `data/raw/official_sources/assembly_pres_2025_context/pres_2025_speaker_issue_matches.csv`
- `data/raw/official_sources/assembly_pres_2025_context/speaker_issue_match_manifest.json`

The four candidate-context tables preserve the frozen through-2022 files as an
exact byte prefix and append three `pres_2025` rows. They live under
`data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/`
instead of replacing `data/raw/`, because the latter is part of the active V23
historical audit boundary. The prospective runner patches these four files
only while assembling the forecast target. Its candidate-strength method is
therefore `direct_speech_derived_candidate_context`; the 12-row ridge adapter
remains a fallback for missing direct context.

## Reproduction

```powershell
python scripts\collect_pres_2025_official_minutes.py

python scripts\build_pres_2025_assembly_context.py `
  --source "<EXTERNAL_CORPUS>\assembly_stance_rows_15_22.csv" `
  --supplement-source "data\raw\official_sources\assembly_pres_2025_minutes\assembly_stance_rows_2025_h1.csv"

python scripts\extract_assembly_speaker_issue_matches.py `
  --source "<EXTERNAL_CORPUS>\trash_dataset.zip" `
  --matches-15 "<EXTERNAL_CORPUS>\15th_assembly_issue_phrase_matches.csv" `
  --out "outputs\pres_2025_speech_reextract\assembly_speaker_issue_matches_16_22.csv" `
  --pres-2025-supplement "data\raw\official_sources\assembly_pres_2025_minutes\assembly_stance_rows_2025_h1.csv" `
  --pres-2025-out "outputs\pres_2025_speech_reextract\assembly_speaker_issue_matches_pres_2025.csv" `
  --combined-out "outputs\pres_2025_speech_reextract\assembly_speaker_issue_matches_15_22.csv" `
  --member-history-source "<EXTERNAL_CORPUS>\historical_assembly_members.csv" `
  --member-history-out "outputs\pres_2025_speech_reextract\assembly_member_history.csv"

python -m presidential_issue_engine.build_assembly_speaker_influence `
  --matches "outputs\pres_2025_speech_reextract\assembly_speaker_issue_matches_15_22.csv" `
  --member-history "outputs\pres_2025_speech_reextract\assembly_member_history.csv" `
  --speaker-out "outputs\pres_2025_speech_reextract\assembly_speaker_influence.csv" `
  --issue-out "outputs\pres_2025_speech_reextract\assembly_issue_speaker_weighted.csv" `
  --scope-out "outputs\pres_2025_speech_reextract\issue_scope_weights_speaker.csv" `
  --conversion-out "outputs\pres_2025_speech_reextract\issue_vote_conversion_speaker.csv" `
  --diagnostics-out "outputs\pres_2025_speech_reextract\assembly_speaker_influence_diagnostics.csv"

python scripts\build_speech_derived_candidate_context_v2.py `
  --output-dir "outputs\pres_2025_speech_derived_candidate_context" `
  --assembly-matches "outputs\pres_2025_speech_reextract\assembly_speaker_issue_matches_15_22.csv" `
  --candidates "data\raw\official_sources\pres_2025_candidate_registry.csv" `
  --speaker-profile "outputs\pres_2025_speech_reextract\assembly_speaker_influence.csv" `
  --preserve-history-dir "data\raw"

python scripts\run_prospective_forecast.py --version v23

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
