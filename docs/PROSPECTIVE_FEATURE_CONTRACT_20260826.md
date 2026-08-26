<!-- active-model-version: v32 -->
# What the prospective target owes the model, column by column

## Status

- Date: 2026-08-26
- Applies to the **V32** prospective assembly; V31 and earlier used the
  blanket zero-fill this replaces
- The classification below is produced by the contract itself, not transcribed

## What it replaces

The prospective assembly closed the gap between the historical frame and the
target frame like this:

```python
# The assembled target already contains the same electorate and issue
# feature contract.  Missing diagnostic-only columns are inert.
for column in historical_base.columns:
    if column not in out.columns:
        out[column] = np.nan if column == "actual" else 0.0
```

The comment states the assumption that made it safe. A sweep of the shipped 2025 artifact found **53 columns identically zero across
all 51 rows** while non-zero somewhere on the scored panel. The criterion is
stated because an earlier edition published **40** with no criterion and the
figure does not reproduce; 53 is what
`outputs/prospective_pres_2025_v31/prediction_stage_audit.csv` gives against
`outputs/active_presidential_nested_v32/nested_predictions.csv`. Of the 53:

| family | columns | classified |
| --- | ---: | --- |
| `regional_accent_*` | 28 | REQUIRED_DERIVED |
| `lineage_identity_*` | 5 | REQUIRED_DERIVED |
| strategic-lane group | 3 | REQUIRED_DERIVED |
| `major_party_core_eligible` | 1 | REQUIRED_DERIVED |
| **unclassified** | **9** | **none — see below** |
| outcome-only / explicit-zero / diagnostic-only | 7 | by declaration |

The accent family is **27 input columns** at `_target_base`, which is what the
contract fills. 28 appear dead in the artifact because 24 of those 27 were dead
(the other three, `regional_accent_reform_{share,trend,volatility}`, are zero on
the scored panel too and that is the canonical answer) and four downstream accent
columns went dead as a consequence. The two numbers count different things at
different sites; earlier editions gave both without saying which.

**Nine columns are dead and belong to no class.**
`rejection_beneficiary_{rate,transfer_in,transfer_net,transfer_out}` and
`strategic_lane_{pressure,reservoir,transfer_in,transfer_net,transfer_out}` are
present-and-zero, so the contract never sees them, and `audit_required_derived`
inspects only REQUIRED_DERIVED families. Whether any of them is model-active is
**unresolved** — which means "five model-active families" is what was found, not
what was proven to be all of them.

Zero is a legal value everywhere they landed, so none of it surfaced in the
output.

## The four classes

Every column the target lacks belongs to exactly one, and a column belonging to
none stops the run.

| class | meaning | what happens |
| --- | --- | --- |
| `REQUIRED_DERIVED` | computable from point-in-time evidence | a registered builder produces it; **no builder is a hard failure**, never a zero |
| `EXPLICIT_ZERO` | zero by design, named individually with a reason | set to `0.0` |
| `OUTCOME_ONLY` | exists only after the election | set to `NaN`, never zero — a zero here is a fabricated result downstream code cannot tell from a real one |
| `DIAGNOSTIC_ONLY` | not read by any prediction stage | set to `0.0`, and the claim is enforced by test rather than asserted here |

## What the contract actually handled

Captured at the contract site during a full V32 prospective run:

| site | class | columns |
| --- | --- | ---: |
| `_target_base` | `REQUIRED_DERIVED` | **28** |
| `_target_base` | `OUTCOME_ONLY` | **3** |

### `REQUIRED_DERIVED` — 28 columns, two families

**`regional_accent` (27).** The whole family: `share`, `trend`, `volatility`
and `reliability` for each of six axes — conservative, liberal, progressive,
centrist, regionalist, reform — plus `regional_accent_reliability`,
`regional_accent_signal` and `regional_accent_volatility`.

Built by running the target rows through `estimate_electorate_layers`, the same
estimator the scored rows get. An earlier builder attached the accent directly
and produced values that looked right and differed from the canonical ones by
up to `0.036` in reliability; calling the estimator is equal to it by
construction rather than by resemblance, verified at a maximum difference of
`5.6e-17`.

Three of the 27 — `regional_accent_reform_{share,trend,volatility}` — come back
zero, and that is the canonical answer: they are zero for all five scored
elections too. A rule reading "all zero means unbuilt" flags them wrongly,
which is why families a builder actually produced are exempt from that check.

**`major_party_core_eligible` (1).** Decidable from the ballot bloc. Under the
zero-fill every 2025 candidate was marked ineligible, including both
major-party nominees.

### `OUTCOME_ONLY` — 3 columns

`err_pp`, `abs_err_pp`, `reproduced_legacy_pred`. Set to `NaN`.

### Declared but not encountered at this site

`EXPLICIT_ZERO` names `frozen_reproduction_difference`,
`landscape_legacy_confidence` and `landscape_legacy_blend`;
`DIAGNOSTIC_ONLY` names `frozen_reproduction_guard_required` and
`dominant_cumulative_rejection`. They are declared because they were found in
the original sweep, and a declaration that is not exercised on a given run is
not thereby wrong.

## Two families that needed the consumer fixed, not the contract

`lineage_identity_*` (5) and the strategic-lane group —
`wasted_vote_resistance`, `strategic_transfer_confidence`,
`major_party_gravity` — never appeared in the missing list, because upstream
stages **create them and default them to zero**. The contract only ever
inspected columns the target *lacked*, so a family that is present-and-dead
passed straight through the check built to catch it.

Two things closed that:

- `audit_required_derived()` treats a required-derived family that is
  identically zero on every row as unbuilt, unless a builder produced it on
  that call;
- the lineage routing now dates a target through the central registry and
  raises rather than skipping. It previously resolved `pres_2025` to `None`
  through a date map that stops at 2022 and skipped it with a bare `continue`,
  leaving all five columns at their initialised zero. **27 of 51 rows now carry
  a lineage shift.**

## The invariant

A column with no declared class raises `ProspectiveFeatureError` naming it. The
contract cannot be satisfied by adding a zero; it has to be satisfied by
deciding what the column is.

## Related

- `DIAGNOSIS_2025_ACCENT_ZEROING_20260825.md` — how the accent family was
  localised, including three explanations that were wrong
- `DIAGNOSIS_INPUT_BOUNDARY_AND_CALIBRATION_20260826.md` — the overlay and seed
  reads, and the calibration acceptance contract
