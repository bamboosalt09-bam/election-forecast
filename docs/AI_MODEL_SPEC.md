# AI and statistical-model specification

This document addresses Attachment 2 and Article 9 of the 2026 Open Source
Developer Competition rules.

## Submitted V27 model

V27 is a self-developed, locally executable statistical forecasting pipeline.
Its fitted component is scikit-learn Ridge regression (`alpha=0.30`) over six
documented, slot-free predictors, followed by deterministic electorate,
contest, lineage, transfer and regional-dispersion transforms. There is no
hosted inference API, neural foundation model, downloaded model weight or
approval-gated model in the submitted V27 execution path.

All model code, input schemas, public/derived inputs, frozen outputs and audit
manifests needed by V27 are present in the public repository and in the built
wheel's verified runtime bundle. The model can run locally after installation.
The fitted Ridge coefficients are regenerated from the chronological
development panel; there is no separately distributed opaque weight file.

## Data and validation boundary

- Development/scored panel: 2002, 2007, 2012, 2017 and 2022 presidential
  elections under chronological nested folds.
- Warm-up history: earlier records as declared by the fold audit.
- 2025: corrected D-1 demonstration, not untouched out-of-sample validation.
- Future/prospective validation: absent as of this release.
- Post-2022 outcomes in the frozen V27 development artifact: none.

Public source families and redistribution decisions are listed in
`PUBLIC_DATA_SOURCES.json` and `DATA_PROVENANCE_AND_REDISTRIBUTION.md`.

## External models outside the submitted runtime

The repository preserves historical, optional stance-classifier experiments
that referenced the following public model pages. They do not feed active V27,
their weights are not redistributed, and the standalone V27 wheel excludes
their experiment runners.

| Model | Use in repository | Source/license status | Submission status |
| --- | --- | --- | --- |
| [`pongjin/roberta_with_kornli`](https://huggingface.co/pongjin/roberta_with_kornli) | Shadow NLI comparison | Hugging Face model card declares Apache-2.0 | Not active; not bundled |
| [`jhgan/ko-sroberta-nli`](https://huggingface.co/jhgan/ko-sroberta-nli) | Shadow embedding comparison | Public weights; model card did not expose an explicit license tag during this audit | Not active; not bundled; do not use for the submitted demo without separate permission verification |
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
