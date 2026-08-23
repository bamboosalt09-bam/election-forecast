# The 2025 forecast path has been unrunnable since the V28 boundary was enforced

## Status

- Date: 2026-08-23
- Status: **diagnosed, not fixed**; the fix changes the published 2025 forecast
- Found while promoting V29; unrelated to V29
- The V29 historical promotion is unaffected and is verified

## Symptom

Every run of the 2025 demonstration is rejected by the prospective harness's own
guard:

```
prospective harness does not reproduce frozen V25 history:
maximum absolute difference=0.003619628460480362
```

The affected rows are pres_2022 in `sido_44`, `sido_36`, `sido_30` and
`sido_31`. `scripts/run_prospective_forecast_v28.py` fails with a
bit-identical difference, so this is not introduced by V29.

## Cause

`8d7851b "Enforce the external-model-free V28 boundary"` made
`external_model_free_runtime()` set `POLL_PROJECT_BLOCK_EXTERNAL_MODEL_SEEDS=1`
and `POLL_PROJECT_ENHANCED_ISSUES=0` process-wide, so the engine no longer reads
`mega_issue_axis.csv` or `mega_issue_attribution.csv`.

The prospective harness builds the 2025 target on top of `base.run("v25")`, and
that call asserts it reproduces the **frozen V25** history byte for byte. V25
was frozen *with* those seed inputs. Under the enforced boundary the assertion
cannot hold, and it is not supposed to: the harness is being asked to match a
baseline built under a different runtime.

Confirmed by elimination:

| test | result |
| --- | --- |
| V28 prospective runner | fails, 0.003619628460480362 |
| V29 prospective runner | fails, same value |
| KOSPI fixed-dataset path replaced with the pre-`15f29f9` computation | fails, same value |
| **V27 prospective runner (no boundary guard)** | **runs** |

The KOSPI fixed dataset was the first hypothesis and is wrong; it is recorded
here so it is not re-investigated.

## The part that is not just reproducibility

`outputs/prospective_pres_2025_v28/` was committed at `f925637`, before
`8d7851b`. It was therefore produced **with** the external-model-derived seed
inputs that V28 documents as removed. The published 2025 forecast does not
satisfy the boundary it is published under, and regenerating it under that
boundary will change the numbers.

That is why this is not fixed here. Choosing between "regenerate the forecast
under the boundary" and "restate the boundary the forecast was made under"
changes what the project publishes, and that is not a call to make silently
inside an unrelated promotion.

## Why nothing caught it

No CI job runs the 2025 path. `v29-clean-reproduction` rebuilds the *historical*
model only. Eight jobs were green across the whole window in which the published
forecast became unreproducible.

`scripts/verify_v29_prospective_reproduction.py` is added by this change as the
missing check. It is **not** wired into CI yet, because it would fail on the
known-broken path; it is the acceptance test for the fix, and it should be added
to the workflow in the same change that repairs the harness.

## Fix sketch

The guard's premise - the harness must not alter history - is right; its
baseline is wrong once the runtime changes. The repair is to compare against a
baseline built under the same runtime, which means freezing a
"V25 pipeline under the external-model-free runtime" reference and having
`base.run` use it when the boundary is active. Then regenerate the 2025
artifact, which will differ from the published one.
