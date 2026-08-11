# Stance Context V21-V25-S Independent Audit

Run period: 2026-08-07 to 2026-08-10

## Boundary

- Workspace: `C:\english_folder\poll_project`
- Active presidential forecast: V23, unchanged
- 2025 outcomes: not read or used
- Assembly PDF/XLSX extraction: not rerun
- Frozen source read for sampling only:
  `assembly_stance_rows_15_22.csv`
- All stance versions in this record are shadow classifiers.

The adoption rule requires all of the following: an independent audit, audited
target attribution and point-in-time validity, zero observed harmful errors, a
one-sided 95% harmful-error upper bound at or below 5%, at least 59 independent
directional emissions, and a later rolling non-degradation check. No version in
this record reached the classifier-quality gate, so forecast integration was
not attempted.

## Independent Results

| Version | Audit | Emissions | Correct | Harmful | Precision | Harmful upper 95% | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| V21 | V13 corrected | 95 | 88 | 7 | 92.63% | 13.39% | fail |
| V22 | V14 | 73 | 68 | 5 | 93.15% | 13.86% | fail |
| V23-S | V15 | 65 | 56 | 9 | 86.15% | 22.92% | fail |
| V24-S | V16 | 92 | 85 | 7 | 92.39% | 13.82% | fail |
| V25-S | V17 | 87 | 73 | 14 | 83.91% | 24.01% | fail |

V13 has a separate adjudication-correction file because one current sentence
was initially reviewed from truncated console text. The locked audit was not
overwritten. The corrected metrics file is authoritative.

## What Changed

V21 added explicit discourse ownership, assertion status, government scope,
referent source, and target-match fields. V22 added grammatical-role and
central-government gates. V23-S added public/committee ownership and historical
named-government scope. V24-S added positive-effect conflict, compound-boundary
handling such as `천정부지`, neutral policy response, deictic history, and
generic crisis mechanisms. V25-S added victim-role, factual-mention,
hypothetical-agent, reported-consensus, and event-description gates.

Each new version was developed only after the previous independent audit was
locked and reviewed. The next confirmatory corpus excluded every prior shadow
text hash. Code hashes were frozen into corpus state files before inference.

## Large Confirmatory Runs

V24-S used a fresh 40,000-row corpus with 8,000 rows per election. It contained
31,765 cue-rich and 8,235 general rows, produced 159 base directional rows, and
retained 92 after V24-S gates.

V25-S used another fresh 40,000-row corpus with 8,000 rows per election. Prior
sampling had depleted the expanded cue pool, so this corpus contained only
3,053 cue-rich and 36,947 general rows. It produced 145 base directional rows
and retained 87 after V25-S gates.

The V25-S precision decline is therefore important rather than incidental: the
rule stack fitted prior error types but generalized poorly when the independent
sample shifted toward ordinary parliamentary language.

## Residual Error Structure

The recurring errors are semantic-role errors, not missing sentiment words:

1. A government or candidate is the victim or burdened actor, not the object of criticism.
2. A speaker reports criticism by the public, media, opposition, or a former official without adopting it.
3. A prior or generic government is mapped to the election's assigned government.
4. `정부` appears inside a public-enterprise name, debt category, policy concept, or Korean compound.
5. A factual premise, prescription, incomplete fragment, or hypothetical executor is treated as a negative stance.
6. Adjacent context supplies a negative topic even when the current target relation is neutral.

These errors require joint owner-target-event interpretation. Adding another
small regex for every audited sentence is now more likely to overfit than to
improve independent precision.

## Decision

- Do not promote V21-V25-S.
- Do not run the full target-bearing corpus through these versions.
- Do not run rolling forecast integration or tune vote-share gains.
- Keep V20 and all later versions as reproducible shadow artifacts only.
- Stop cumulative regex patching after V25-S.

The next defensible experiment is a separately trained, abstention-capable
role classifier with three explicit tasks: stance ownership, target validity,
and polarity. V13-V17 audit labels can supply hard negatives, but evaluation
must use source-group and election-group splits plus a new locked corpus. The
model should emit direction only when all three task probabilities clear their
own conservative thresholds.

## Final Verification

- Full regression suite: `535 passed in 179.48s`
- Active pointer SHA256:
  `d01b584b126ae850696f3b280872267783e7d336bd59ae1dc2a04a6bacf824d4`
- V23 finalization manifest SHA256:
  `7e9b7c23a5b38f438c1cb7fadaff2735909ef75eb7254ae3236791d7d637c3a4`
- Long V24-S and V25-S inference resumed from atomic 1,000-row checkpoints;
  no partial chunk was treated as complete.

## Authoritative Artifacts

- V21 corrected metrics:
  `outputs/assembly_stance/stance_context_discourse_target_v21/locked_audit_v13_metrics_corrected.json`
- V22 metrics:
  `outputs/assembly_stance/stance_context_grammatical_target_v22/locked_audit_v14_metrics.json`
- V23-S metrics:
  `outputs/assembly_stance/stance_context_pragmatic_role_v23s/locked_audit_v15_metrics.json`
- V24-S metrics:
  `outputs/assembly_stance/stance_context_lexical_role_v24s/locked_audit_v16_metrics.json`
- V25-S metrics:
  `outputs/assembly_stance/stance_context_semantic_role_v25s/locked_audit_v17_metrics.json`
- Locked audits and adjudications:
  `data/shadow/stance_locked_audit_v13*` through
  `data/shadow/stance_locked_audit_v17*`
