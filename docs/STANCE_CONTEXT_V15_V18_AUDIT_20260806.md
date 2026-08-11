# Stance Context V15-V18 Audit (2026-08-06)

## Decision

V18 improves the precision-first parliamentary stance classifier, but it is
not safe enough for forecast integration or a full-corpus production run.
Active forecast V23 remains unchanged.

- V18 independent directional precision: `91.5254%`
- harmful errors: `5 / 59`
- harmful-error 95% upper bound: `16.9963%`
- required gate: zero observed harmful errors and upper bound at most `5%`
- rolling non-degradation: not tested because the classifier gate failed first
- 2025 outcomes used: no

The remaining errors are target/ownership errors, not positive-versus-negative
reversals. They are still harmful because neutral evidence would become a
directional forecast signal.

## Frozen forecast boundary

The active presidential model is still:

- version: `v23`
- regional weighted macro MAE: `3.367899%p`
- national candidate macro MAE: `1.597845%p`
- winner accuracy: `4/5`

No classifier output in this experiment was connected to the active forecast.
No full 291k target-bearing corpus run was launched.

## Data provenance

Fresh audit corpora were selected from:

`C:\english_folder\poll_project_post2025_outcome_aware_20260714\outputs\assembly_stance\full_15_22\assembly_stance_rows_15_22.csv`

The directory name contains `post2025`, but the source artifact itself is the
Assembly 15th-22nd corpus. Every fresh selector and application validator
restricted elections to `pres_2002`, `pres_2007`, `pres_2012`, `pres_2017`,
and `pres_2022`. The generated states record:

- `post_2022_rows_present: false`
- `vote_outcomes_used: false`

Each fresh corpus contains 5,000 unique rows, exactly 1,000 per scored election.
Previously selected and audited text hashes were excluded before selection.

## Audit lineage

| Version | Evaluation status | Emissions | Harmful errors | Precision | 95% upper bound | Decision |
|---|---|---:|---:|---:|---:|---|
| V15 | independent V7 | 59 | 7 | 88.14% | 21.13% | reject |
| V16 | development on V7 | 52 | 0 | 100.00% | 5.60% | not independent |
| V16 | independent V8 | 59 | 9 | 84.75% | 25.11% | reject |
| V17 | development on V8 | 50 | 0 | 100.00% | 5.82% | not independent |
| V17 | independent V9 | 59 | 8 | 86.44% | 23.14% | reject |
| V18 | development on V9 | 51 | 0 | 100.00% | 5.70% | not independent |
| V18 | independent V10 | 59 | 5 | 91.53% | 17.00% | reject |

Development rows were used to design the next rule version and therefore are
never cited as independent evidence.

## V18 changes

V18 adds only general, outcome-free abstention rules:

1. Named-government mentions are checked against the meeting date. A generic
   government target is rejected when the named administration was not in
   office on that date.
2. Joint governing-party/government policy commitments are treated as target
   self-position, not external approval.
3. Unresolved responsibility questions are neutralized.
4. Conditional approval-loss projections and conditional evaluations are
   neutralized.
5. A narrow reported-politician frame is neutralized.

V18 uses no candidate vote share, election error, or 2025 outcome.

## Independent V10 errors

The five V10 harmful errors were:

1. A conditional local-government finance risk assigned to the national
   government target.
2. Firms' distrust reported as the speaker's own negative stance.
3. A prior administration's housing project assigned to the current generic
   government target because the prior administration was not named locally.
4. A Gallup survey result treated as the speaker's own stance.
5. India's support for Korean policy treated as the speaker/government's own
   positive evaluation.

These categories show why a full-corpus run is premature. Remaining work is
primarily target resolution and stance ownership, not polarity strength.

## Reproducible artifacts

Core rules and tests:

- `src/election_forecast/stance_context_v15.py`
- `src/election_forecast/stance_context_v16.py`
- `src/election_forecast/stance_context_v17.py`
- `src/election_forecast/stance_context_v18.py`
- `tests/test_stance_precision.py`

Independent audit locks:

- V9 audit hash: `55a41da5f6cb7e4b66953aa832470426aec1d408bcc2a0afcdfad3e9018bd728`
- V10-A hash: `19310d9c7ac1b4bf3282824c388270866c1d3d00c8a2a7fc0b6ef5a4e87a1f32`
- V10-B hash: `542db8615fe4c14e337018faa6b31c34f7c91a72a5b4f5c6963606b2889b9eb9`
- V10 labels hash: `913fb1a75b20b3ec9985b6c882aa7940ce536a2b1c94951f4b9a3cf4c200b7c7`

Authoritative metric files:

- `outputs/assembly_stance/stance_context_speaker_scope_v17/locked_audit_v9_metrics.json`
- `outputs/assembly_stance/stance_context_speaker_scope_v18/locked_audit_v10_metrics.json`

## Next technically justified step

Do not add more election-specific regexes. A future V19 should separate three
tasks explicitly:

1. entity-time target resolution for named and unnamed administrations;
2. quotation/attribution ownership, including surveys, firms, and foreign
   actors;
3. stance polarity only after the first two stages pass.

The next independent evaluation must use new locked text hashes. V10 cannot be
reused as confirmatory evidence for V19.

## Verification

- targeted stance tests: `49 passed`
- full repository suite: `494 passed in 101.76s`
- active forecast integration: unchanged and intentionally not executed
