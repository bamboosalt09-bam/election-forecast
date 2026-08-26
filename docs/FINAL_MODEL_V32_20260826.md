<!-- active-model-version: v32 -->
# Final presidential model V32 — the version whose score cannot move

V32 is V31 with one change: the prospective feature assembly obeys the same
contract as the historical one. Everything else — the Ridge stack, the
predictors, the shock structure, the V28 external-model boundary, V27's regional
transform, the forecast-time weighting, the gain and V31's multiplicative
dispersion expansion — is V31's, unchanged.

Its scored artifact is byte-identical to V31's. That is the point, not a
caveat: what V32 changes is the path that builds the target election's features,
and the five scored elections do not take that path.

## Why

The prospective assembly closed the gap between the historical frame and the
target frame like this:

```python
# The assembled target already contains the same electorate and issue
# feature contract.  Missing diagnostic-only columns are inert.
for column in historical_base.columns:
    if column not in out.columns:
        out[column] = np.nan if column == "actual" else 0.0
```

The comment states the assumption that made it safe. A sweep of the shipped
2025 artifact found **40 columns identically zero across all 51 rows** while
populated for every scored election, and **five families among them were
model-active**:

| family | columns | what it fed |
| --- | ---: | --- |
| `regional_accent_*` | 27 | the accent gain, which gates the log shift, which moves the prediction |
| `major_party_core_eligible` | 1 | every 2025 candidate was marked ineligible, so their durable core was zeroed and folded into critical support |
| `lineage_identity_*` | 5 | the routing resolved `pres_2025` to no date and skipped it |
| `wasted_vote_resistance` | 1 | the consumer read a history-only table while the target's own context carried it |
| `strategic_transfer_confidence` | 1 | same consumer, same cause |

Zero is a legal value everywhere they landed, so none of it surfaced in the
output. The scored elections carry accent gains of 0.10 to 0.20; the published
2025 forecast ran with that layer contributing exactly nothing.

The last three never appeared in the missing list at all — upstream stages
create them and default them to zero, so a contract inspecting only *absent*
columns passed straight over them. Catching those needed a second check
(`audit_required_derived`) and a fix in the consumer, not in the contract.

## What replaces it

Every column the target lacks is classified, and a column belonging to no class
stops the run:

| class | what happens |
| --- | --- |
| `REQUIRED_DERIVED` | a registered builder produces it; **no builder is a hard failure**, never a zero |
| `EXPLICIT_ZERO` | zero by design, named individually with a reason |
| `OUTCOME_ONLY` | `NaN`, never zero — a zero here is a fabricated result downstream code cannot tell from a real one |
| `DIAGNOSTIC_ONLY` | `0.0`, and the claim is enforced by test rather than asserted |

The contract cannot be satisfied by adding a zero. It has to be satisfied by
deciding what the column is. The column-by-column classification is in
`PROSPECTIVE_FEATURE_CONTRACT_20260826.md`.

## Frozen development metrics

- regional equal-election macro MAE: `2.5007010072077227%p`
- national equal-election macro MAE: `0.7242913678028117%p`
- winner accuracy: `0.8`
- scored rows: `232`
- post-2022 outcomes used: `false`

Both macros change by exactly `0.000000%p` against V31, and
`nested_predictions.csv` hashes to the same
`969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069`. The audit
checks that as a byte comparison rather than a tolerance, because a tolerance is
the form of this claim that could later be quietly loosened.

Byte identity is asserted of the **committed** artifact — in the audit, in the
finalization manifest and in `tests/test_v32_promotion.py`, all of which compare
files in the tree. A *rebuild* is held to the repository's published
reproduction contract instead, `1e-12`, and records whether the bytes also
matched. The difference matters: demanding a byte match of a rebuild made a
correct third-party reproduction fail, and it did so *intermittently* — the same
job on the same runner later reproduced the panel byte for byte. The record is in
`EXPERIMENT_V32_PROSPECTIVE_FEATURE_CONTRACT_20260826.md`.

**So the version was decided on the correctness of the feature contract, not on
a score.** There was no score to decide it on. The 2025 outcome was not
consulted at any point, and the size of the forecast's movement was measured
after the change was decided, not used to decide it.

## What the 2025 forecast does

The demonstration was regenerated. Against the published V31 artifact the
regional shares move by at most `1.2381%p` (mean `0.2960%p`) and the national
levels by `0.4152%p`; the winner and the ranking are unchanged. Those figures
are measured from the two artifacts by the finalizer rather than transcribed.

## Two other things V32 closes

**External-model-derived inputs are refused by file name, not directory path.**
The old rule required `data/raw/auto_issue_seed/mega_issue_axis.csv` and so
stopped matching once the prospective runner had copied the same table into a
temporary directory, where the trace showed it opened three times. All three
tables — the sentence-level stance overlay and the two mega-issue seeds — are
refused, the refusal is recorded, and the run still completes, so none of them
was needed. Each was separately demonstrated inert: V31 re-runs byte-identical
with the files moved out of the tree.

**The dispersion calibration has an acceptance tolerance.** It used to stop
early under `1e-11` and otherwise return its last iterate with nothing checked
and nothing recorded, which makes a near-miss indistinguishable from
convergence. One tolerance now replaces the split between a convergence
criterion and a failure criterion: `1e-8` in share units, which is `1e-6`
percentage points on a published figure. Exhausting the iteration budget is not
a success condition. Details, including what remains unresolved about the
residual plateau, are in `DIAGNOSIS_INPUT_BOUNDARY_AND_CALIBRATION_20260826.md`.

## AI boundary

Unchanged from V28, re-audited, and now traced where the read happens:

- hosted inference API: none
- downloaded model weights at runtime: none
- external neural encoder at runtime: none
- external-model-derived active input: one compact frozen candidate-issue aggregate
- fitted component: scikit-learn Ridge plus deterministic project transforms

The Ridge regularisation strength is disclosed per fold: `0.3, 0.3, 0.8, 1.2,
1.2`.

## Evidence boundaries

2002–2022 remain a development panel, not an untouched holdout. The 2025
artifact remains a corrected D-1 demonstration at cutoff `2025-06-02`.

The scored panel is still defined by which candidates cleared roughly 1% of the
actual vote — a declared modelling scope recorded in
`DIAGNOSIS_SCORING_SCOPE_20260824.md`. The headline metric still weights by
`contest_votes`; that is the definition of national vote share rather than a
leak, and the reasoning is in `METRIC_WEIGHTING_20260825.md`.

One defect in V31's frozen record — its audit script's docstring opening "Audit
active V30", carried across the promotion while everything under it was updated
— is corrected in V32's own copy rather than rewritten in place, because that
file pins its own hash in the V31 manifest.

## Known open

| item | grade |
| --- | --- |
| the calibration residual plateau's root cause | unresolved observation, not a defect |
| the `1e-11` condition inside `_calibrate` is unchanged, so three plateau calls still run all 200 rounds | P3 performance; no effect on any published figure |
| the scored-path manifest check still inspects a file the line above it rewrote | P2, carried forward |
