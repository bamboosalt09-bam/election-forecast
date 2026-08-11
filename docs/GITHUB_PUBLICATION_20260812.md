# GitHub Initial Publication Record

## Destination and history

- repository: `https://github.com/bamboosalt09-bam/election-forecast`
- baseline branch: `main`
- baseline commit: `5757a5d` (`chore: establish reproducible V23 baseline`)
- publication branch: `codex/initial-publication`
- delivery method: pull request after GitHub Actions CI

The root commit preserves the through-2022 V23 state that was staged before
the 2025 forecast-only work. The publication branch adds the 2025 input
boundary, official-source collectors, tests, and documentation without
changing the frozen V23 predictions or active-model hashes.

## Public data boundary

Tracked:

- source code, tests, configuration, and small canonical inputs;
- frozen V23 audit artifacts;
- 2025 official meeting and context manifests;
- 2025 sufficient-statistic context files used by the forecast-only loader;
- official National Assembly and Bank of Korea ECOS collection code.

Not tracked:

- the 5.8 GB Assembly sentence corpus;
- the 33 MB row-level 2025 Assembly supplement;
- raw Assembly HTML and PDFs, API caches, and checkpoints;
- the commercial/user-provided KOSPI export and the downloaded ECOS row file;
- virtual environments, backups, shadow corpora, and noncanonical outputs.

The excluded data are represented by source metadata, row counts, hashes, and
reproduction commands. No real credentials are stored in tracked files.

## Forecast and information boundary

- scored elections: 2002, 2007, 2012, 2017, and 2022;
- rolling warmup only: 1997;
- forecast-only demonstration: 2025;
- 2025 result use: prohibited for fitting, tuning, ablation, stage selection,
  comparison, and error calculation;
- 2025 Assembly cutoff: 2025-06-02;
- official-minute availability proxy: PDF `CreationDate` plus one day, with
  missing metadata failing closed.

## Publication verification

- repository-boundary audit: PASS;
- active V23 audit: PASS, 199 prediction rows;
- target-outcome invariance: PASS, 215/215 rows;
- 2025 demonstration boundary audit: PASS;
- 2025 official minutes: 239 discovered and completed, 91 eligible at cutoff;
- regression suite: 547 passed;
- tracked secret-pattern scan: no matches;
- largest intended publication file: below the 25 MiB repository policy.

## KOSPI status

The official ECOS fetcher and provenance manifest are published, but the
source migration is not active in V23. The commercial local export remains
ignored. Activating ECOS data requires a new versioned run and the promotion
checks in `docs/KOSPI_OFFICIAL_SOURCE_MIGRATION_20260810.md`.
