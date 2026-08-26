<!-- active-model-version: v32 -->
# AI and statistical-model specification

This document addresses Attachment 2 and Article 9 of the 2026 Open Source
Developer Competition rules.

## Submitted V31 model

V31 is a self-developed, locally executable statistical forecasting pipeline.
Its fitted component is scikit-learn Ridge regression over six
documented, slot-free predictors, followed by deterministic electorate,
contest, lineage, transfer and regional-dispersion transforms. There is no
hosted inference API, neural foundation model, downloaded model weight or
approval-gated model in the submitted V31 execution path. The direct
sentence-level stance overlay is rejected. One compact historical
candidate-by-issue aggregate derived during an earlier encoder experiment is
retained and disclosed because removing it materially changes the validated
postprocess; no model weight or source sentence is bundled.

All model code, input schemas, public/derived inputs, frozen outputs and audit
manifests needed by V31 are present in the public repository and in the built
wheel's verified runtime bundle. The model can run locally after installation.
### Ridge regularisation

The specification previously named a single `alpha=0.30`. The artifact carries
five. Retrospective outer folds use chronologically selected and then frozen
fold-specific values, and the through-2022 deployment configuration uses its
own separately frozen deployment alpha, so the two must not be quoted as one
number:

| outer fold | frozen alpha |
| --- | ---: |
| pres_2002 | 0.3 |
| pres_2007 | 0.3 |
| pres_2012 | 0.8 |
| pres_2017 | 1.2 |
| pres_2022 | 1.2 |

Each is selected inside its own fold from strictly earlier elections, which is
why they differ; reading them as one hyperparameter would describe a model the
panel never ran. The values are reproduced in
`outputs/active_presidential_nested_v31/fold_audit.csv`.

The fitted Ridge coefficients are regenerated from the chronological
development panel; there is no separately distributed opaque weight file.

## Data and validation boundary

- Development/scored panel: 2002, 2007, 2012, 2017 and 2022 presidential
  elections under chronological nested folds.
- Warm-up history: earlier records as declared by the fold audit.
- 2025: corrected D-1 demonstration, not untouched out-of-sample validation.
- Future/prospective validation: absent as of this release.
- Post-2022 outcomes in the frozen V31 development artifact: none.
- Target-election outcome quantities reaching any transform: none. The two
  terminal transforms weight by the *previous* election's regional valid
  votes; 2002 uses a 1997 warmup table. The panel's membership rule and the
  diagnostic metric weighting remain outcome-defined and are disclosed as
  such in `DIAGNOSIS_SCORING_SCOPE_20260824.md` and the README.

Public source families and redistribution decisions are listed in
`PUBLIC_DATA_SOURCES.json` and `DATA_PROVENANCE_AND_REDISTRIBUTION.md`.

## Archived external-model experiments

The repository history documents stance-classifier experiments that referenced
the following public model pages. No model executes in active V32 and their
weights are not redistributed. The V31 wheel excludes experiment runners,
optional dependency extras, inactive stance modules, source sentences and the
direct overlay. A disclosed frozen aggregate at
`data/raw/auto_issue_seed/candidate_issue_profile.csv` remains an active
historical postprocess input. The frozen V23-V27 rollback record is not itself
an active V32 input.

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
