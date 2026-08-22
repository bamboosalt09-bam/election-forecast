# External-model overlay removal experiment — 2026-08-22

## Corrected decision

Remove neural inference, model weights, source sentence corpora,
`data/raw/assembly_issue_character_overlay.csv`, and the unused
`mega_issue_axis.csv` / `mega_issue_attribution.csv` descendants from the
installable runtime. Retain the compact frozen historical
`candidate_issue_profile.csv` because the active postprocess still consumes it
and its actual removal is not negligible. Its ancestry is disclosed rather
than incorrectly describing V28 as free of every external-model-derived input.

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

The original stricter-run claim was invalid: the feature assembler was
disabled, but `scripts/run_active_presidential_model.py` independently loaded
the same candidate profile for direct mega-issue and government-burden
postprocessing. The generated input manifest then removed that path after the
fact, hiding the live read.

Disabling only the direct overlay changed the 2025 D-1 demonstration by at
most `0.003918%p` in a regional candidate share and `0.001081%p` nationally.
An actual schema-only profile injection changed the 2002–2022 development
panel as follows: regional macro MAE `2.613902987%p` to `4.935929128%p`,
national macro MAE `0.720993881%p` to `4.128408475%p`, winner accuracy `0.8`
to `0.6`, and maximum row-level `layer_pred` difference `0.2074599613`.
No 2025 outcome was read, scored or used to make this decision.

The runtime and direct-overlay removal is adopted because it reduces licensing
and supply-chain surface. Full descendant removal is rejected because it is a
material model change, not a cleanup. The retained aggregate is packaged and
audited explicitly so a clean installation cannot silently depend on an
unpublished local file.
