<!-- active-model-version: v32 -->
# The stance overlay and the dispersion calibration, closed

## Status

- Date: 2026-08-26
- Both items are **closed** and are not carried as open defects
- Every claim below is tagged **observed**, **demonstrated**, **inferred** or
  **unresolved**, and nothing is stated more strongly than its evidence

---

## 1. `assembly_issue_character_overlay.csv`

### Closure

> V31 — the direct overlay makes no contribution to any prediction and is
> excluded from the packaged runtime, but source execution retains an inert
> legacy read. V32 — the read itself is removed. The manifest audit structure
> is a separate remediation target, tracked below rather than here.

### What was established

**Observed.** `issue_vote_engine.py:3215` calls
`_read_csv_if_exists(ASSEMBLY_ISSUE_CHARACTER_OVERLAY)` through a bare module
constant. The file exists, is 106,114 bytes, and self-identifies as
`source_model=stance_nli_ambiguity_v14`.

**Observed.** `external_model_free_runtime()` sets an environment variable,
`overlay_gain = 0.0` and a registry flag. It does not alter that constant and
does not intercept that call. The module's own docstring says the guard "does
not read the sentence-level stance overlay", which for this code path is not
true as written.

**Observed.** `setup.py`'s `PUBLIC_EXCLUDED_FILES` names the overlay, and the
packaged runtime archive does not contain it.

**Demonstrated.** V31 was re-run end to end with the file moved out of the tree.
`nested_predictions.csv` came back at `969e63fe…`, identical to the shipped
artifact; `layer_pred` differed on 0 of 232 rows; the regional macro was
`2.500701` either way.

**Demonstrated.** Every `issue_pref_*` and `issue_attention_*` column has been
identically zero since V28 — 17 of 17 populated in V26 and V27, 0 of 17 from
V28 onward. The overlay's downstream signal is switched off by the guard
through the consumer that *does* honour it.

**Inferred.** The read is therefore inert: it opens a file whose product is
already zeroed before it can reach a prediction. This is an inference from the
two demonstrations above rather than a separate measurement.

### What V32 changes

The V32 prospective runner refuses the path instead of reading and discarding
it, and records the refusal so the trace still shows that the engine asks:

```
data/raw/assembly_issue_character_overlay.csv    refused_by_v32    1
```

For V32 the word "removed" is literally accurate for the source path as well.

### Separate remediation target: the manifest audit

**Observed.** V28 establishes its claim in this order:

```python
with external_model_free_runtime():
    v27.run(...)
strip_external_model_inputs(manifest_path)          # deletes the rows
assert_external_model_free_manifest(manifest_path)  # checks what is left
```

The assertion inspects a file the line above it edited. A passing manifest
therefore evidences only that the row was removed before anyone looked.

`raw_input_read_trace.py` records reads where they happen and is never edited;
the V32 prospective run writes `raw_input_read_trace.csv` beside its artifact
and the check reads that. Extending the same treatment to the scored path, and
separating "what was read" from "what may be redistributed", is **open** and
tracked as P2.

### What that P2 is and is not

It is a defect in **one method of proof**, not in the model, and not in the
claim the method was meant to support. The claim — that no external-model
product reaches a prediction — rests on four independent legs, and the manifest
check is the weakest of them. Removing it entirely would not weaken the
conclusion:

| evidence | what it shows | grade |
| --- | --- | --- |
| the files moved out of the tree | V31 re-runs to `969e63fe…`, byte-identical; `layer_pred` differs on 0 of 232 rows | **demonstrated** |
| downstream disappearance | every `issue_pref_*` and `issue_attention_*` column identically zero since V28; 17 of 17 populated in V26 and V27, 0 of 17 after | **demonstrated** |
| the read trace | written where the read happens and never edited afterwards: 0 calls on the scored path, and all three tables refused on the prospective one | **observed** |
| the distribution | the three files are absent from the wheel and the sdist, checked on every build | **observed** |

So the exposure is not that an external model might be influencing the forecast.
It is that a reader auditing the chain would find a check that inspects a file
the line above it rewrote — a check that cannot fail — and would be right to
discount it. The cost is credibility of the audit, not correctness of the model.

**Why it was not fixed in V32.** The strip-then-assert pair lives in
`external_model_free_runtime.py`, which is pinned by hash in the V30, V31 **and**
V32 finalization manifests. Editing it in place breaks three versions'
reproduction at once, which the change policy forbids. It is fixable — by a
runtime seam, or by a version that can repin the module and invert the order —
and it is carried forward rather than left unfixable.

---

## 2. `party_regionalism_dispersion._calibrate`

### Closure

> The tolerance-contract defect is **fixed**. The root cause of the residual
> plateau is an **unresolved numerical observation** and is not carried as a
> defect.

### What was established

**Observed.** The loop stops early when the candidate residual falls below
`1e-11` and otherwise returns its last iterate with no check and no record.

**Observed.** On the scored panel, three of five calibration calls never meet
`1e-11`. They settle at `1.862645e-09`, `3.725290e-09` and `3.725290e-09`.

**Observed.** Raising the iteration budget from 200 to 20,000 does not reduce
any of them by a single digit; the same values return.

**Inferred.** The `1e-11` termination condition was therefore stricter than the
numerical fixed point this implementation reaches, for those three calls. The
other two meet it and stop early, so **the loop was not always exhausting its
budget** — an earlier draft said it was, which the five-call breakdown directly
above contradicts.

**Unresolved.** Whether that plateau is a floating-point floor or a property of
the alternation — or a small incompatibility between how the two constraints
are applied — is not determined. An attempt to separate them by recomputing in
`np.longdouble` answered nothing: on this platform `np.longdouble` has the same
`2.22e-16` epsilon as `float64`, so the precision was never raised. The claim
"float64 makes `1e-11` unreachable" was made earlier in this work and is
**withdrawn**; machine epsilon is `2.22e-16` and summing 51 rows does not by
itself produce a `1e-9` floor. Resolving this needs arbitrary precision and is
not required by the fix.

### The fix

One acceptance tolerance replaces the split between a convergence criterion and
a failure criterion:

```
CALIBRATION_ABS_TOL = 1e-8       # share units = 1e-6 percentage points
```

It is an accuracy contract rather than a figure fitted to the plateau: a
numerical reconciliation is accepted when it deforms a prediction by no more
than a millionth of a percentage point. The observed plateau sits inside that
bound; the bound would be the same had the plateau been elsewhere.

Four conditions are checked together, because conserving candidate levels while
breaking the regional composition is not success either:

| condition | tolerance |
| --- | --- |
| worst candidate-level residual | `<= 1e-8` |
| worst region-sum residual | `<= 1e-8` |
| all values finite | — |
| all values valid shares | — |

**Exhausting the iteration budget is not a success condition.** Well-posed
input meets the tolerance; input that does not, after the budget, fails.

The audit records `iteration_budget`, `max_candidate_residual`,
`max_region_sum_residual`, `tolerance`, `converged`, and the impact bound, so
the choice does not have to be reconstructed later.

**Demonstrated.** Under the contract the scored panel passes and V32's scored
predictions are byte-identical to V31's (`969e63fe…`).

### Why a wrapper rather than an edit

`party_regionalism_dispersion` is pinned by hash in the V30 and V31
finalization manifests. Wrapping leaves those reproducing. Moving the tolerance
into the loop — which would also let it stop early — belongs to whichever
version can edit that module under its own manifest.

---

## Still open

| item | grade | note |
| --- | --- | --- |
| calibration plateau root cause | — | unresolved observation, not a defect |
| `audit_public_active_presidential_model_v31.py` docstring says "Audit active V30" | P3 | the file pins its own hash, so correcting it needs a version that can repin — deferred to the V32 seal |

The scored-path trace item is closed: the scored run is now traced, its trace
comes back empty, and the reason is recorded beside it — that path never calls
`issue_vote_engine._read_csv_if_exists`, measured at 0 calls, because it reads
the frozen through-2022 rederived artifacts instead of assembling from source
tables.

Two items that were on this list are now closed.

### `mega_issue_*` seeds — closed

**Demonstrated.** V31 was re-run with `mega_issue_axis.csv` and
`mega_issue_attribution.csv` moved out of the tree. `nested_predictions.csv`
came back at `969e63fe…`, identical; `layer_pred` differed on 0 of 232 rows;
the regional macro was `2.500701` either way. `direct_mega_score`,
`direct_mega_log_shift` and `mega_issue_intensity_response` kept exactly the
same non-zero counts with and without the files.

Same conclusion as the overlay, by the same method: read, and inert.

### Blocking rules keyed on path strings — closed

**Observed.** The rule required `data/raw/auto_issue_seed/mega_issue_axis.csv`
and therefore missed the same table once the prospective runner had copied it
into a temporary directory, where the trace showed it opened three times. A
rule keyed on where a file sits checks the path, not the input.

Matching is now on the file name. **Demonstrated:** the temp-directory copy is
matched, all three external-model-derived files are refused in the V32
prospective run, and the run still completes — so the combined seed is not
needed either. Refusing it moves the 2025 forecast by `1.2381%p` at most
against V31 (from `1.2394%p` when only the overlay was refused), and the scored
panel stays byte-identical at `969e63fe…`.
