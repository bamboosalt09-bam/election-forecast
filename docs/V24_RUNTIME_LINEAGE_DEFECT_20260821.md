# V24 runtime-lineage defect and bounded V25 repair — 2026-08-21

## Decision

The published V24 artifact remains unchanged as the predecessor and rollback
record. V25 is a separate successor. No `pres_2025` realised outcome was read,
scored, fitted, or used for this repair.

## Defect

V24 was documented as retaining the V23 numerical runtime while adding the
ballot-faithful panel, uniform 1% scored floor, and V24 structural
postprocesses. Its runner instead invoked the generic active runner directly.
That bypassed promoted V23 runtime bindings for the lineage-projected prior,
prior-selected contest response, unified regional lineage, duplicate general
regional-identity suppression, and several automatic-control inputs. It also
changed `active.CONFIG_PATH` without rebinding the default argument captured by
`load_policy()`.

## Correction boundary

V25 restores the omitted runtime bindings individually. It retains all V24
ballot and scored-scope corrections and applies the accepted V24 postprocesses
in their existing order:

1. strong incumbent veto;
2. third-candidate lineage ceiling;
3. weak same-lane refusal with `recipient_weight_mode=prediction_tilted`.

Three paths are deliberately left in their accepted V24 state:

- `data/raw/candidate_vote_conversion_context.csv` for the upper strategic
  transfer route;
- `data/raw/third_candidate_profile.csv`;
- `data/raw/third_candidate_pressure.csv`.

The first experimental V25 draft incorrectly rebound the conversion path to
the V24 versioned assembly context. That was outside the V23 inheritance repair
and was removed.

The V23 automatic third-candidate profile/pressure pair is also not rebound.
V24's weak-C mechanism was selected with the existing generic pair. Rebinding
the V23 pair on top of the V24 mechanism duplicates third-candidate pressure
and fails the already-declared winner safety gate. This is not a newly tuned
coefficient or election-specific exception; it preserves the runtime used by
the accepted V24 hypothesis.

The slot-keyed V23 withdrawal registry also remains disabled because restoring
it removes the real 2022 slot-C ballot candidate, repeating the representation
defect V24 was designed to repair.

## Controlled ablation

Every row below uses the same 232-row V24 ballot panel and the same
`prediction_tilted` weak-C postprocess. No coefficient or threshold was varied.

| Runtime repair | Regional weighted macro MAE | National candidate macro MAE | Winners |
|---|---:|---:|---:|
| V24 predecessor | 2.769788%p | 1.075668%p | 4/5 |
| policy binding only | 2.770%p | 1.076%p | 4/5 |
| automatic contest response only | 2.751%p | 1.065%p | 4/5 |
| unified prior only | 2.768%p | 1.074%p | 4/5 |
| unified identity only | 2.806%p | 1.052%p | 4/5 |
| duplicate general identity disabled only | 2.779%p | 1.085%p | 4/5 |
| V23 automatic inputs as one bundle | 2.734%p | 1.035%p | 3/5 |
| core repairs plus V23 third-candidate inputs only | 2.718%p | 1.044%p | 3/5 |
| bounded V25, preserving V24 third-candidate inputs | 2.774%p | 0.990%p | 4/5 |

The winner failure is therefore not caused by the accepted
`prediction_tilted` implementation being replaced with the rejected
`affinity_only` implementation. The audit contains only `prediction_tilted`.
It occurs when the V23 automatic third-candidate profile/pressure pair is
stacked on top of V24's accepted weak-C response.

## Version policy

- V23 remains immutable.
- V24 remains immutable as the published predecessor.
- V25 is the bounded runtime-lineage correction.
- The V25 historical artifact must reproduce deterministically and pass the
  full regression suite before the active pointer is changed.
- A V25 `pres_2025` forecast is permitted only after the historical V25
  artifact is frozen, and remains strictly outcome-free.
