<!-- active-model-version: v30 -->
# The scoring panel is defined by an actual result, and what that does

## Status

- Date: 2026-08-24
- Status: **recorded, not changed**
- The rule is a declared modelling scope; this document says what it costs and
  what it does not

## The rule

The scored panel holds the A/B/C ballot slots of each election. A slot is
dropped when its candidate took roughly under 1% of the vote. In the shipped
results table this is carried as `is_active_slot`, and it is False for exactly
one scored slot:

| election | slot C actual share | in the panel |
| --- | ---: | :---: |
| pres_2002 | 3.90 % | yes |
| pres_2007 | 15.08 % | yes |
| pres_2012 | **0.17 %** | **no** |
| pres_2017 | 21.42 % | yes |
| pres_2022 | 2.38 % | yes |

The lowest actual share anywhere in the panel is 2.409%.

## Why it is a scope decision and not a defect

Whether to model a contest as two-way or as one with a viable third candidate is
the researcher's framing, and it has to be settled somehow. Settling it with a
written threshold is better practice than deciding case by case, and the
threshold was fixed once rather than tuned.

## What is nonetheless true

The threshold reads an **actual** result, so the panel's membership is not
knowable at forecast time. On a borderline case — a candidate polling several
percent who finishes just under the line — the rule would retroactively remove a
candidate a real forecast would have had to predict.

No such case exists in this panel. The one exclusion is 2012's slot C at 0.17%,
which is not near any plausible line.

## The rule did not flatter the model

This is the part worth measuring rather than asserting. Slot C in 2012 took
0.17% of the vote. Including it would have added rows whose actual value is near
zero and whose prediction would also have been near zero — small absolute errors
that **dilute** a mean absolute error computed over the panel.

So excluding it, if anything, made the reported figure slightly worse than it
would otherwise have been. The rule is not self-serving, which is the specific
worry a reader should have about any outcome-defined scope.

## What would close it properly

Defining the same scope with a quantity available before the election — final
pre-election polling support, or registration and ballot status — would keep the
intent and remove the dependence on the result. That is a modelling change with
its own consequences and is not made here.

## Related

- `EXPERIMENT_V30_FORECAST_TIME_WEIGHTS_20260824.md` — the other place a target
  outcome reached the model, which V30 does change
- `REPRODUCIBILITY.md` — the evidence boundary the scored panel sits inside
