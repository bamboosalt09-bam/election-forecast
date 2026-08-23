# AI and statistical-model specification

This document addresses Attachment 2 and Article 9 of the 2026 Open Source
Developer Competition rules.

## Submitted V29 model

V29 is a self-developed, locally executable statistical forecasting pipeline.
Its fitted component is scikit-learn Ridge regression (`alpha=0.30`) over six
documented, slot-free predictors, followed by deterministic electorate,
contest, lineage, transfer and regional-dispersion transforms. There is no
hosted inference API, neural foundation model, downloaded model weight or
approval-gated model in the submitted V29 execution path. The direct
sentence-level stance overlay is rejected. One compact historical
candidate-by-issue aggregate derived during an earlier encoder experiment is
retained and disclosed because removing it materially changes the validated
postprocess; no model weight or source sentence is bundled.

All model code, input schemas, public/derived inputs, frozen outputs and audit
manifests needed by V29 are present in the public repository and in the built
wheel's verified runtime bundle. The model can run locally after installation.
The fitted Ridge coefficients are regenerated from the chronological
development panel; there is no separately distributed opaque weight file.

## Data and validation boundary

- Development/scored panel: 2002, 2007, 2012, 2017 and 2022 presidential
  elections under chronological nested folds.
- Warm-up history: earlier records as declared by the fold audit.
- 2025: corrected D-1 demonstration, not untouched out-of-sample validation.
- Future/prospective validation: absent as of this release.
- Post-2022 outcomes in the frozen V29 development artifact: none.

Public source families and redistribution decisions are listed in
`PUBLIC_DATA_SOURCES.json` and `DATA_PROVENANCE_AND_REDISTRIBUTION.md`.

## Archived external-model experiments

The repository history documents stance-classifier experiments that referenced
the following public model pages. No model executes in active V29 and their
weights are not redistributed. The V29 wheel excludes experiment runners,
optional dependency extras, inactive stance modules, source sentences and the
direct overlay. A disclosed frozen aggregate at
`data/raw/auto_issue_seed/candidate_issue_profile.csv` remains an active
historical postprocess input. The frozen V23-V27 rollback record is not itself
an active V29 input.

| Model | Use in repository | Source/license status | Submission status |
| --- | --- | --- | --- |
| [`pongjin/roberta_with_kornli`](https://huggingface.co/pongjin/roberta_with_kornli) | Shadow NLI comparison | Hugging Face model card declares Apache-2.0 | Not active; not bundled |
| [`jhgan/ko-sroberta-nli`](https://huggingface.co/jhgan/ko-sroberta-nli) | Historical embedding and bounded-overlay experiment | Public weights; model card did not expose an explicit license tag during this audit | Runtime, weights, source sentences and direct overlay removed; one compact derived aggregate retained and disclosed |
| [`jhgan/ko-sroberta-multitask`](https://huggingface.co/jhgan/ko-sroberta-multitask) | Historical precision experiment | Public weights; model card did not expose an explicit license tag during this audit | Not active; not bundled; do not use for the submitted demo without separate permission verification |
| [`klue/roberta-small`](https://huggingface.co/klue/roberta-small) | Historical encoder experiment | Public weights; downstream dataset/model terms require separate review | Not active; not bundled; do not use for the submitted demo without completing that review |

This fail-closed treatment is intentional: public download access is not
treated as a redistribution license.

## Development-assistant disclosure

Commercial AI coding assistants were used for code drafting, debugging,
documentation and audit support. Human review, local execution, regression
tests, frozen-hash checks and source-rights decisions remain the participant's
responsibility. Assistant use is not presented as the project's technical
innovation and no assistant API is required at runtime.
