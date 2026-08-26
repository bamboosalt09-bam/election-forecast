<!-- active-model-version: v32 -->
# The 2025 forecast's Assembly inputs: what ships, what does not, and how to rebuild any of it

The 2025 D-1 demonstration is built from official National Assembly proceedings.
Neither the collected 2025 proceedings nor the archived historical matches are
redistributed; three derived files are. This document says exactly what each
contains, what the difference costs, and how to regenerate every one of them.

## The five files

| file | ships | size | carries source text |
| --- | :---: | ---: | :---: |
| `assembly_stance_rows_2025_h1.csv` | **no** | 35.0 MB | yes — 48,588 excerpts |
| `assembly_stance_rows_2025_h1_public.csv.gz` | yes | 0.52 MB | no |
| `assembly_issue_matches_2025_h1_public.csv.gz` | yes | 0.12 MB | no |
| `assembly_speaker_issue_matches_15_22.csv` (archived) | **no** | 118.9 MB | no |
| `assembly_speaker_issue_matches_15_22.csv.gz` | yes | 3.07 MB | no |

The first three live in `data/raw/official_sources/assembly_pres_2025_minutes/`.
The published historical matches live in `data/raw/official_sources/`; the
archived original sits under `archives/experiments/`.

Together the three published files are 3.7 MB, against 154 MB of inputs the
repository does not track.

### `assembly_stance_rows_2025_h1.csv` — collected, not redistributed

One row per (speech, sentence, issue) with `text_excerpt` holding the verbatim
sentence and `speaker` / `member_id` naming who said it. This is the sentence
corpus the project excludes: it is listed under `excluded_paths` in
[PUBLIC_DATA_SOURCES.json](PUBLIC_DATA_SOURCES.json), ignored by Git, and
excluded from the wheel. The provider's terms require attribution and prohibit
sharing the supplied information unchanged.

### `assembly_stance_rows_2025_h1_public.csv.gz` — the rows without the words

The same rows, with `text_excerpt` replaced by `text_length` (its character
count) and the `stance_*` / `target_*` columns dropped. Those columns are
outputs of the external stance models that V28 removed from the runtime, so the
active model never reads them.

`source_sha256` is retained. Anyone holding the official minutes can therefore
verify any row against its source without the project redistributing the text.

### `assembly_speaker_issue_matches_15_22.csv.gz` — the historical matches

The forecast is built against the five scored elections, and their speaker-level
issue matches are a 195,758-row table. It used to be read straight out of
`archives/experiments/manual_seed_lineage_v17_rejected_20260728/`, which was a
poor dependency twice over: the repository boundary forbids tracking
`archives/`, so a clean checkout could not run this path, and the directory name
says the experiment was *rejected* even though the active forecast read it.

Neither was a rights problem — the table already carries `text_length` rather
than `text_excerpt`. It is published gzipped instead: the same rows, 3.07 MB
against 118.9 MB, because `committee`, `agenda` and `source_file` repeat
heavily. **The published copy is now the only one the active path reads**; the
two are identical in content, and the archive is a historical record that
nothing active consults.

Rebuild it with:

```bash
python scripts/build_redistributable_assembly_issue_matches.py
```

### `assembly_issue_matches_2025_h1_public.csv.gz` — the keyword rematch result

`election_id, period, speaker, issue_name, issue_weight, matched_term_count`,
34,253 rows.

This one exists because a length cannot substitute for the words. Three places
consume the collected file, and two of them use only `text_excerpt`'s length —
but `_historical_compatible_target_matches` in
`scripts/run_prospective_forecast.py` joins the excerpts of a speech and runs
`match_issue_weights` over the joined text. That step needs the words, so its
**output** is published instead of its input.

## What that costs, precisely

With the three published files, a clean checkout reproduces the 2025 forecast
**downstream of the keyword matching**. The matching itself is taken as given.

That is a real boundary and it is worth being blunt about: someone auditing from
the public tree alone can confirm that these issue weights produce this
forecast, and cannot confirm that these proceedings produce these issue weights.
The second half needs the proceedings.

It is not, however, a new kind of compromise. `candidate_issue_profile.csv`
already ships on the same terms — a disclosed derived aggregate whose inputs are
not redistributed — and the same reasoning covers it.

## Regenerating each file

### From the official source, end to end

1. Collect the proceedings. `scripts/collect_pres_2025_official_minutes.py`
   downloads the 22nd Assembly minutes for the 2025 window and writes
   `assembly_stance_rows_2025_h1.csv` together with `manifest.json`, which
   records a SHA-256 for every source document.

   ```bash
   python scripts/collect_pres_2025_official_minutes.py
   ```

   This requires access to the official minutes service and takes a long time.
   Everything it fetches is a public record; what the project declines to do is
   *redistribute* it.

2. Verify what you collected against the shipped manifest:

   ```bash
   python scripts/audit_pres_2025_demo_boundary.py
   ```

   This checks the collected file against the recorded hashes and against the
   point-in-time boundary — that nothing dated after the D-1 cutoff,
   **2025-06-02**, entered the inputs.

3. Rebuild the published files:

   ```bash
   python scripts/build_redistributable_pres_2025_stance_rows.py
   python scripts/build_redistributable_assembly_issue_matches.py
   ```

   The output is deterministic: the gzip header carries no timestamp, so the
   artifacts hash identically across rebuilds and machines. If your hashes
   differ from the committed ones, your collected source differs — that is the
   check working, not a nuisance.

### From the public files only

Nothing to regenerate; they are in the repository. To confirm the forecast they
produce matches the frozen artifact:

```bash
python scripts/verify_v32_prospective_reproduction.py
```

With the collected file present this rebuilds everything including the keyword
matching. Without it, the matching is loaded from the published result and the
rest is recomputed. Either way the script says which of the two it did — it
reports `matches_recomputed` in the run diagnostics — so a reproduction is never
silently weaker than it appears.

### Rebuilding the forecast itself

```bash
python scripts/run_prospective_forecast_v32.py
```

Writes `outputs/prospective_pres_2025_v32/`. The runner resolves the collected
file if present and the public form otherwise; no flag selects between them, so
neither path can be taken by accident.

## The point-in-time rule

Every input is filtered to what was available on **2025-06-02**, the day before
the election. `available_date` carries that eligibility per row and
`availability_basis` records why. The 2025 outcome is not used anywhere: not in
fitting, not in weighting, not in stage selection, and not in any comparison.
`scripts/audit_pres_2025_demo_boundary.py` enforces this, and the forecast run
refuses to proceed if the cutoff drifts.

## Where this is recorded elsewhere

- [PUBLIC_DATA_SOURCES.json](PUBLIC_DATA_SOURCES.json) — machine-readable
  provenance and the `excluded_paths` list
- [DATA_PROVENANCE_AND_REDISTRIBUTION.md](DATA_PROVENANCE_AND_REDISTRIBUTION.md)
  — the per-family redistribution decisions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — what can and cannot be rebuilt from
  the public tree
- [DIAGNOSIS_PROSPECTIVE_2025_PATH_20260823.md](DIAGNOSIS_PROSPECTIVE_2025_PATH_20260823.md)
  — why this path was unrunnable, and how it was repaired
