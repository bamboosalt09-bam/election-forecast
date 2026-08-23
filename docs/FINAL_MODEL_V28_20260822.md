# Final presidential model V28 — external-model-runtime-free boundary

V28 is V27 without neural inference, downloaded model weights, source sentence
corpora or the direct Assembly stance overlay in its installable runtime. Two
unused automatic issue-seed descendants are also excluded. The frozen
historical `data/raw/auto_issue_seed/candidate_issue_profile.csv` remains an
active postprocess input and is included with explicit provenance disclosure.
Official parliamentary records and deterministic issue matching remain active.
The two excluded mega seeds are blocked by a process-wide guard so legacy
modules loaded after the V28 wrapper cannot silently re-enable them.

## Frozen development metrics

- regional equal-election macro MAE: `2.638410502170951%p`
- national equal-election macro MAE: `0.726249711635409%p`
- winner accuracy: `0.8`
- scored rows: `232`
- post-2022 outcomes used: `false`

The historical prediction table differs slightly from V27 because the final
runtime audit found that a late-loaded legacy module could re-enable the two
excluded mega seeds. Enforcing the boundary changed regional macro MAE by
`+0.024507515%p`, national macro MAE by `+0.005255831%p`, and did not change
winner accuracy. V27 remains frozen as the pre-removal rollback.

## AI boundary

- hosted inference API: none
- downloaded model weights at runtime: none
- external neural encoder at runtime: none
- external-model-derived active input: one compact frozen candidate-issue aggregate
- fitted component: scikit-learn Ridge plus deterministic project transforms

Historical stance experiment code, model weights, sentence corpora and direct
overlay are not distributed in the V28 wheel. The retained compact aggregate
contains no source sentence or model weight. A full-removal diagnostic changed
the development-panel regional MAE from `2.613902987%p` to `4.935929128%p` and
winner accuracy from `0.8` to `0.6`; therefore it cannot truthfully be described
as a negligible inactive dependency.

## Evidence boundary

2002–2022 remain a development panel, not an untouched holdout.  The 2025
artifact remains a corrected D-1 demonstration, not genuine prospective
validation.  Future-election prospective validation is still absent.
