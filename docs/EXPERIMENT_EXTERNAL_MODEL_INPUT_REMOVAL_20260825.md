<!-- active-model-version: v30 -->
# Removing the external-model-derived input, re-measured on V30

## Status

- Date: 2026-08-25
- Status: **diagnostic**; nothing removed, nothing promoted
- Post-2022 outcomes used: none
- Reproduce: `python scripts/evaluate_external_model_input_removal.py`

## Why it was re-measured

`data/raw/auto_issue_seed/candidate_issue_profile.csv` is the one active input
descended from an external encoder. It is retained and disclosed rather than
dropped, and the reason given is that removal is not cost-free.

The figure behind that claim was measured on **V27** in
`EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md` and never repeated.
By V30 the compliance document and the rights registry were quoting a cost
against a baseline that no longer existed: `2.613902987%p` was V27's regional
macro, three promotions behind.

## Method

A *schema-only injection*: the file is replaced by one with the same header and
no rows. That is stricter than deleting it, because a missing file changes
control flow while an empty one runs the same code path with no evidence in it.

The file is swapped **on disk** rather than by patching the paths that read it.
The original V27 experiment first reported no change at all, because several
modules loaded the same file independently and patching one left the others
live. Swapping the file cannot miss a reader. The script restores the original
afterwards and verifies its SHA-256, so a crashed run cannot leave a tracked
input modified.

## Result

Scored on the frozen `contest_votes` axis, so the two measurements are
comparable:

| diagnostic | retained | removed | change |
| --- | ---: | ---: | ---: |
| regional equal-election macro MAE | `2.566444753%p` | `4.948234183%p` | `+2.381789431%p` |
| national equal-election macro MAE | `0.720437417%p` | `4.144568904%p` | `+3.424131486%p` |
| winner accuracy | `0.8` | `0.6` | `−0.2` |
| prediction rows | 232 | 232 | 0 |

Against the V27-era measurement (`2.613902987` → `4.935929128`, winner `0.8` →
`0.6`), the conclusion is unchanged and the magnitude is nearly identical. The
claim that removal is not cost-free now rests on a current number rather than a
superseded one.

## What this does and does not establish

It establishes that the input is load-bearing: the postprocess that consumes it
degrades to roughly twice the regional error without it, and the file is
therefore disclosed rather than described as an inert leftover.

It does not establish that the input is *good*. A large degradation on removal
and a well-founded input are different claims; this measures only the first.

It also says nothing about rights. If the reading of the encoder's absent
licence tag ever changes, the file is dropped and this is the cost that would
be paid — which is why it is measured rather than estimated.

## Related

- `EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md` — the V27 original
- `COMPETITION_COMPLIANCE_2026.md` — the rights record this figure supports
- `AI_MODEL_SPEC.md` — the external-model boundary
