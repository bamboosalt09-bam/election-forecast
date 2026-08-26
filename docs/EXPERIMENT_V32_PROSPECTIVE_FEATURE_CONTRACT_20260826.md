<!-- active-model-version: v32 -->
# V32 — prospective/historical feature-contract parity

## Status

- Date: 2026-08-26
- Promoted and frozen as V32
- Every claim below is graded **observed**, **demonstrated**, **inferred** or
  **unresolved**, and nothing is stated more strongly than its evidence

## The experiment, stated so it can fail

A version is normally accepted or rejected on what it does to the scored panel.
V32 cannot be, and saying so plainly is part of the record: **its scored
artifact is byte-identical to its predecessor's**, so no metric can speak for or
against it. The claim being tested is therefore about the contract, not the
score:

> The prospective assembly must not be able to satisfy a missing feature by
> writing a zero. Every column the target lacks must be classified, a required
> column with no builder must fail the run, and a column belonging to no class
> must stop it.

That claim is falsifiable in the only way it can be: add an unclassified column
and the run raises; take a builder away and the run raises; make the target's
own path produce a zero where history has a value and the audit names the
family.

## How the defect was found, including four wrong turns

**Observed.** A sweep of the shipped 2025 artifact against the scored panel
found 40 columns identically zero across all 51 rows while populated for every
scored election.

**Observed.** The assembly filled every missing column with `0.0`, under a
comment asserting the missing ones were diagnostic-only.

Before the zero-fill was located, three explanations were pursued and
abandoned — the accent gain map, then the temporary-directory seed copies, then
a suspected fold-selection difference. Each was consistent with the symptom and
none was the cause. `DIAGNOSIS_2025_ACCENT_ZEROING_20260825.md` records them,
because a diagnosis that only records the correct turn overstates how legible
the defect was.

A fourth wrong turn is recorded here rather than there, because it came after
the contract existed. The first `audit_required_derived` treated any
required-derived family that was identically zero as unbuilt, and it
false-positived immediately: `regional_accent_reform_{share,trend,volatility}`
come back zero, and that is the canonical answer — they are zero for all five
scored elections too. Families a builder actually produced are now exempt from
that check. A rule reading "all zero means unbuilt" is wrong wherever zero is
the right answer, which is precisely where this whole class of defect lives.

## The five model-active families

**Observed.** `regional_accent_*` (27 columns) feeds the accent gain, which
gates the log shift, which moves the prediction. The scored elections carry
gains of 0.10 to 0.20; the published 2025 forecast ran with the layer
contributing exactly nothing.

**Demonstrated.** Building them by attaching the accent directly produced
values that looked right and differed from the canonical ones by up to `0.036`
in reliability. Running the target rows through `estimate_electorate_layers` —
the same estimator the scored rows get — is equal to it by construction rather
than by resemblance, verified at a maximum difference of `5.6e-17`.

**Observed.** `major_party_core_eligible` was zero for every 2025 candidate,
including both major-party nominees, so their durable core was zeroed and folded
into critical support. Eligibility is decidable from the ballot bloc.

**Observed.** `lineage_identity_*` (5 columns) never entered the missing list.
The routing resolved `pres_2025` through a date map that stops at 2022, got
`None`, and skipped the target with a bare `continue`, leaving the columns at
their initialised zero. **Measured:** 35 of 232 scored rows carry a non-zero
lineage score and 87 of 232 a non-zero log shift; after the fix, 27 of 51 target
rows carry a lineage shift.

An earlier statement that the score was zero in 230 of 232 rows was a
`pres_2022`-only measurement — 2 rows of 51 — generalised to the panel. It is
withdrawn and the panel-wide figures above replace it.

**Observed.** `wasted_vote_resistance` and `strategic_transfer_confidence` were
read by the consumer from a history-only table while the target's own context
carried them.

## What was *not* done

**Rejected: any use of the 2025 outcome.** Not for fitting, tuning, ablation,
stage selection, thresholds or parameters. The forecast's movement was measured
after the change was decided.

**Rejected: tuning the accent gain map.** It was one of the wrong turns above,
and it stayed rejected after it stopped being a wrong turn, because a gain
chosen to move the 2025 figure is a parameter fitted to the target.

**Rejected: editing any file pinned in the V30 or V31 finalization manifests.**
Twice a fix was written directly into `issue_vote_engine.py` and
`audit_public_active_presidential_model_v31.py`, which broke V31's audit; both
were reverted. Every V32 behaviour change that touches a pinned module is a
module-level flag defaulting to off, switched on by the V32 runner — the pattern
V30 used for `v27.WEIGHT_COLUMN`.

## The failure this nearly repeated

**Observed.** The first version of the lineage routing fix changed the shared
module unconditionally. The scored panel could not move — all five scored
elections resolve through the original date map — and that reasoning was
correct. But the *prospective* target is precisely the case that took the `None`
branch, so V31's own 2025 forecast moved, and CI caught it:

```
prospective_predictions.csv: rebuilt forecast differs from the frozen
artifact by 0.003350856250666223
```

Only the scored verifier had been run. An earlier session had recorded the
identical mistake through `region_bloc_prior` at `0.0032`, and that record had
been read before this was done. The fix is the default-off seam
(`ROUTING_REQUIRES_DATABLE_TARGET`): V31 reproduces, V32 keeps the fix.

## The guard that was right about the wrong thing

**Observed.** The first version of this runner required the rebuilt scored
artifact to be **byte identical** to V31's on every run. On the Windows CI
runner it raised:

```
V32's scored panel differs from V31's ... Expected 969e63fe..., got 1c2a5fee...
```

**Observed.** The same wheel, with the same pinned dependencies (`numpy 2.4.6`,
`pandas 3.0.5`, `scipy 1.18.0`, `scikit-learn 1.9.0`), reproduces byte-for-byte
on the authoring machine, and the Linux reproduction job passes.

**Observed.** V31's verifier — and every verifier before it — compares values at
`atol=1e-12`, never bytes. That is the reproduction contract this repository
publishes.

**Inferred.** The Windows runner's floating-point path formats a few final
digits differently while agreeing well inside `1e-12`. V31's contract absorbed
that; V32's byte condition, being strictly stronger, turned a correct
reproduction into a failure.

**Unresolved.** The exact magnitude of the difference on that runner. The guard
raised before anything measured it, which is itself part of the defect: a
fail-closed check should say how far off it was.

The claim was not weakened, it was moved to where it is true. Byte identity is a
property of the committed artifact and is still asserted in three places that
compare files in the tree — the V32 audit, the finalization manifest, and
`tests/test_v32_promotion.py`. What a *rebuild* must satisfy is the published
`1e-12`, and whether its bytes also matched is recorded in `summary.json`
(`scored_panel_identical_to_v31`, `scored_panel_max_abs_difference_vs_v31`)
rather than assumed. Those fields are measured, not written as literals, for the
same reason V31's two hardcoded manifest claims had to be replaced.

A text column has no tolerance to spend and still fails exactly; so does a
changed row count or a changed schema.

## Result

**Demonstrated.** `outputs/active_presidential_nested_v32/nested_predictions.csv`
hashes to `969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069`,
identical to V31's. Both macros change by exactly `0.000000%p`. The three
predictive-interval CSVs are byte-identical too; only the manifest differs, by
its version label and its warning text.

**Demonstrated.** The 2025 forecast moves: regional shares by at most
`1.2381%p` (mean `0.2960%p`), national levels by `0.4152%p`, winner and ranking
unchanged. The finalizer measures this from the two artifacts rather than
recording a transcribed number — the V31 manifest hardcoded three conclusions
about its own transform and two of them had gone false by the time anyone read
them.

## Related

- `PROSPECTIVE_FEATURE_CONTRACT_20260826.md` — the generated classification
- `DIAGNOSIS_INPUT_BOUNDARY_AND_CALIBRATION_20260826.md` — the overlay and seed
  reads, and the calibration acceptance contract
- `DIAGNOSIS_2025_ACCENT_ZEROING_20260825.md` — how the accent family was
  localised
- `DIAGNOSIS_REGIONALISM_DEAD_ENDS_20260825.md` — two abandoned approaches to
  regionalism for political newcomers
