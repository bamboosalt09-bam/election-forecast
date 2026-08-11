# External Stance Models V26-S to V29-S Audit

Date: 2026-08-10

## Boundary

- Workspace: `C:\english_folder\poll_project`
- Active presidential forecast: V23, unchanged
- 2025 outcomes: not read or used
- Assembly PDF/XLSX extraction: not rerun
- Full-corpus stance integration: not run
- Rolling forecast integration: not run
- All models in this record are shadow-only

## External Basis

The experiment replaced cumulative sentence-specific regex patches with three
ideas from external work:

1. Target-Stance Extraction separates target identification from stance
   prediction instead of treating sentiment as stance.
   - https://aclanthology.org/2023.acl-long.560/
2. KLUE supplies Korean NLI and relation-understanding evidence and pretrained
   Korean language models.
   - https://arxiv.org/abs/2105.09680
   - https://github.com/KLUE-benchmark/KLUE
3. Selective classification evaluates a risk-coverage trade-off and permits
   abstention rather than forcing a label.
   - https://proceedings.mlr.press/v97/geifman19a.html

The direct NLI model was `pongjin/roberta_with_kornli`, pinned at revision
`138378c1fb502754eb27a699a8ad71955c4d9668` under Apache-2.0. It was trained on
KorNLI and reports 0.811 validation accuracy in its model card. The sentence
encoder was `jhgan/ko-sroberta-nli`, pinned at revision
`c4e15f24df2aceadfc931e2a57094726b2409861`.

## Implemented Methods

### V26-S: direct NLI cascade

Six Korean hypotheses were evaluated for every row: direct target, speaker
ownership, external report, negative stance, positive stance, and factual
neutrality. Logistic heads predicted target validity, ownership, and stance.
A direction was emitted only when every gate passed.

On audit versions V16-V17 treated as a pseudo-holdout, the model emitted only
9/179 rows, made one harmful error, and reached 88.89% precision. The approach
was too conservative and no more accurate than the existing rule line.

### V26-S role veto and V27-S task-specific veto

The NLI model was then prevented from creating or reversing direction. It
could only veto an existing prediction. V27-S also prohibited polarity
features from entering target and owner heads.

V26-S retained 8/179 rows with one harmful error. V27-S retained 2/179 rows
with zero observed errors. Neither was useful because coverage collapsed.

The pseudo-holdout ROC-AUC was 0.7104 for target validity but only 0.5892 for
speaker ownership. The NLI representation recognized negative content but did
not reliably distinguish an adopted claim from a report, victim role, or
historical reference.

### V28-S: Korean NLI embedding veto

All human audit labels through V15 were used with grouped cross-validation.
The fixed Korean NLI sentence embedding was compared alone and concatenated
with direct NLI probabilities. The development coverage floor was corrected
to scale the 59-emission independent-audit requirement from 179 rows to 909
development rows.

| V28-S feature set | V16-V17 emissions | Harmful | Precision | Upper 95% |
|---|---:|---:|---:|---:|
| Embedding only | 68 | 2 | 97.06% | 8.97% |
| Embedding + direct NLI | 59 | 3 | 94.92% | 12.62% |
| Dual target consensus | 67 | 2 | 97.01% | 9.10% |

This looked promising but did not pass the 5% upper-bound rule. The two
embedding-only errors were then inspected, so V16-V17 ceased to be a valid
independent test for any later consensus design.

### V29-S: fresh locked audit

V16-V17 labels were incorporated into development. The V29-S policy was frozen
before reading any new labels:

- logistic `C=2.0`;
- target threshold `0.70`;
- owner threshold `0.55`;
- minimum grouped-OOF development emissions `514/1088`;
- grouped-OOF precision `93.39%` with 34 harmful errors in 514 emissions.

The first untouched pool contained 125 base directional rows and produced 30
locked emissions (V18). Without opening V18, the same frozen artifact and
policy were applied to 344 additional untouched base emissions, producing 83
more locked rows (V19). The combined 113 rows were then adjudicated.

## Independent Result

| Audit | Emissions | Correct | Harmful | Precision | Upper 95% |
|---|---:|---:|---:|---:|---:|
| V18 | 30 | 24 | 6 | 80.00% | 35.70% |
| V19 | 83 | 72 | 11 | 86.75% | 20.98% |
| Combined | 113 | 96 | 17 | 84.96% | 21.71% |

The combined harmful errors comprise 15 neutral-to-direction errors and two
wrong-sign errors. The main residual classes are:

1. quoted or externally reported criticism;
2. prior-government criticism assigned to the current government scope;
3. a named person acting as the source of a quotation rather than its target;
4. a policy prescription mistaken for criticism;
5. praise or defense misread as negative because the surrounding topic is
   adversarial.

The independent pool was harder than V16-V17 because it consisted of
previously unaudited base-model emissions. That distribution shift is the
relevant production test. V28-S's pseudo-holdout improvement did not
generalize.

## Decision

- Do not promote V26-S through V29-S.
- Do not tune another threshold against V18-V19.
- Do not connect these artifacts to vote-share estimation.
- Keep the active V23 forecast and its issue pipeline unchanged.
- Stop treating generic NLI entailment probabilities or fixed sentence
  embeddings as a sufficient owner-target parser.

The next defensible route is a genuinely target-aware supervised model trained
on balanced random neutral rows plus hard negatives, with separate target,
owner, and polarity losses. It should be fine-tuned end to end on Korean text,
validated by source and election groups, calibrated for selective risk, and
tested on a new locked random-plus-hard audit. A stronger external generative
model may be used as an annotation assistant, but never as unreviewed truth.

## Artifacts

- V26-S cascade metrics:
  `outputs/assembly_stance/stance_external_kornli_role_cascade_v26s/metrics.json`
- V26-S veto metrics:
  `outputs/assembly_stance/stance_external_kornli_role_veto_v26s/metrics.json`
- V27-S task-specific metrics:
  `outputs/assembly_stance/stance_external_kornli_task_specific_veto_v27s/metrics.json`
- V28-S feature ablation:
  `outputs/assembly_stance/stance_external_embedding_role_veto_v28s/metrics.json`
- V29-S independent metrics:
  `outputs/assembly_stance/stance_external_embedding_role_veto_v29s/locked_audit_v18_v19_metrics.json`
- Locked audits and labels:
  `data/shadow/stance_locked_audit_v18*`,
  `data/shadow/stance_locked_audit_v19*`

## Final Verification

- Full regression suite: `538 passed in 184.00s`
- Active pointer SHA256:
  `d01b584b126ae850696f3b280872267783e7d336bd59ae1dc2a04a6bacf824d4`
- V23 finalization manifest SHA256:
  `7e9b7c23a5b38f438c1cb7fadaff2735909ef75eb7254ae3236791d7d637c3a4`
- Backup:
  `backups/model_checkpoints/20260810_external_stance_v26_v29_shadow`
