<!-- active-model-version: v31 -->
# V32 in progress — handoff (superseded 2026-08-26)

> **Superseded.** V32 was sealed on 2026-08-26. Both P1 failures below were
> fixed — the prospective reproduction by the default-off
> `ROUTING_REQUIRES_DATABLE_TARGET` seam, and the repository boundary by
> admitting the V32 output directories to the baseline. The current record is
> `FINAL_MODEL_V32_20260826.md` and
> `EXPERIMENT_V32_PROSPECTIVE_FEATURE_CONTRACT_20260826.md`; the open items
> that survived the seal are listed in the V32 finalization manifest under
> `known_open`. This file is kept as written, because a handoff that is edited
> after the fact stops being evidence of what was known at the time.

## Status at the time of writing

- Branch `codex/v32-target-accent`, 9 commits ahead of `main`, pushed
- Draft PR: `https://github.com/bamboosalt09-bam/election-forecast/pull/38`
- **Not sealed.** No V32 pointer, finalization manifest or promotion. The active
  version is V31.
- **Two CI jobs are failing** and are the immediate next work — see below.

## What V32 is for

Not "fix the accent layer". The goal is **historical/prospective feature-contract
parity**: the prospective assembly used to fill any column the target lacked
with zero, and the point is that the next column to go quietly dead is caught by
a rule rather than by someone reading an artifact.

## The two failures to fix first

### 1. `prospective-reproduction` — V31's 2025 forecast moved

```
prospective_predictions.csv: rebuilt forecast differs from the frozen
artifact by 0.003350856250666223
```

**Cause.** The runtime lineage cutoff fix in
`presidential_issue_engine/unified_lineage_identity.py` — `_routing_cutoff()`
plus turning the bare `continue` into a `raise`. That module is shared, so
**V31's prospective run** now dates `pres_2025` and applies lineage shifts,
which the frozen V31 2025 artifact does not have.

**Why it was missed.** The argument that the scored panel cannot move is
correct — all five scored elections resolve through the original date map. But
the prospective target is precisely the case that took the `None` branch, so
changing that branch necessarily changes V31's prospective output. Only
`verify_v31_clean_reproduction.py` (scored) was run; the prospective verifier
was not. An earlier session recorded the identical mistake through
`region_bloc_prior` at 0.0032 and the record was read before this was done.

**Fix direction.** Move the lineage cutoff behaviour behind a module-level hook
defaulting to off, and have the V32 runner switch it on — the pattern V30 used
for `v27.WEIGHT_COLUMN` and V32 already uses for `_calibrate` and
`_read_csv_if_exists`. Then V31 reproduces and V32 keeps the fix.

### 2. `repository-boundary` — not yet diagnosed

Fails on CI, passes locally after the V31 baseline was given
`outputs/prospective_pres_2025_v32/` in `allowed_output_prefixes`. Likely a
second check, or the scored V32 directory. **Read the job log before changing
anything.**

## Work already done on this branch

| area | state |
| --- | --- |
| `prospective_feature_contract.py` | four classes, unclassified column raises. 28 `REQUIRED_DERIVED` + 3 `OUTCOME_ONLY` at `_target_base` |
| `audit_required_derived()` | closes the present-but-zero bypass; families a builder produced are exempt |
| lineage routing | dates through the central registry, raises instead of skipping — **needs the seam above** |
| `raw_input_read_trace.py` | records reads, never edited; refusals kept and labelled |
| overlay + `mega_issue_*` | measured inert (V31 re-runs return `969e63fe` unchanged); refused in V32 |
| `calibration_guard.py` | one acceptance tolerance `1e-8` share = `1e-6`%p; four invariants; audit CSV |
| scored invariance | V31 → V32 scored predictions byte identical, `969e63fe` |
| disclosure | Ridge α five fold values; superseded scope file annotated; SHA invariant across six surfaces |

## Standing constraints

- **Do not seal.** No pointer, manifest or promotion until told.
- **Never edit a file pinned in the V30/V31 finalization manifests** —
  `issue_vote_engine.py`, `party_regionalism_dispersion.py`,
  `audit_public_active_presidential_model_v31.py` and others. Use a runtime seam
  in the V32 runner instead. Editing one and reverting has already happened
  twice.
- 2025 outcome must not enter fitting, tuning, selection, thresholds or
  parameters. A read-only post-election evaluation is the disclosed exception.
- Report cause, impact and grade before fixing; grade claims as **observed**,
  **demonstrated**, **inferred** or **unresolved**.

## Known open

| item | grade |
| --- | --- |
| `prospective-reproduction` failing (above) | **P1** |
| `repository-boundary` failing (above) | **P1** |
| V31 audit docstring says "Audit active V30"; the file pins its own hash | P3, deferred to the seal |
| calibration plateau root cause | unresolved observation, not a defect |
| PR #38 body claims every V31 audit passes | needs correcting once the above is fixed |

## Where the evidence lives

- `docs/PROSPECTIVE_FEATURE_CONTRACT_20260826.md` — the generated classification
- `docs/DIAGNOSIS_INPUT_BOUNDARY_AND_CALIBRATION_20260826.md` — overlay, seeds,
  calibration, with evidence graded
- `docs/DIAGNOSIS_2025_ACCENT_ZEROING_20260825.md` — how the accent family was
  localised, including four wrong turns
- `docs/DIAGNOSIS_REGIONALISM_DEAD_ENDS_20260825.md` — two abandoned approaches
