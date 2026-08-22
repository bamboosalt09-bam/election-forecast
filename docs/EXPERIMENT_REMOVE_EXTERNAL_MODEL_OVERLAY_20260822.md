# External-model overlay removal experiment — 2026-08-22

## Decision

Remove `data/raw/assembly_issue_character_overlay.csv` and the three automatic
issue-seed tables derived from it from the active model lineage. Keep those
files only as frozen V23–V27 rollback/research inputs and keep all historical
classifier records outside the installable runtime.

The retained parliamentary evidence is produced from official National
Assembly Library / National Assembly records using deterministic phrase,
context and point-in-time rules.  V28 runs no neural encoder and consumes no
feature derived from one. The excluded seed tables are
`candidate_issue_profile.csv`, `mega_issue_axis.csv`, and
`mega_issue_attribution.csv` under `data/raw/auto_issue_seed/`.

## Controlled comparison

The V27 chain was rerun first with the direct overlay disabled and then with
both the direct overlay and its automatic seed descendants disabled before
creating V28.

| Diagnostic | V27 | no-overlay candidate | Change |
| --- | ---: | ---: | ---: |
| historical prediction rows | 232 | 232 | 0 |
| regional macro MAE | 2.613902987%p | 2.613902987%p | 0 |
| national macro MAE | 0.720993881%p | 0.720993881%p | 0 |
| winner accuracy | 0.8 | 0.8 | 0 |
| maximum historical numeric difference | — | 0 | 0 |

The stricter descendant-removal run was also exactly identical on all 232
historical rows (`layer_pred` maximum difference `0`). Thus the zero result is
not an artifact of disabling only the final overlay loader.

Disabling only the direct overlay changed the 2025 D-1 demonstration by at
most `0.003918%p` in a regional candidate share and `0.001081%p` nationally.
The final stricter run, which also removed all three automatic seed
descendants, changed it by at most `0.018289%p` regionally and `0.006620%p`
nationally. No 2025 outcome was read, scored or used to make this decision.

The removal is adopted because it reduces licensing, disclosure and supply-
chain surface without sacrificing historical performance.  It is not claimed
as a predictive improvement.
