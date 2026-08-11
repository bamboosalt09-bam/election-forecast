# Active Outcome-Blind Nested Model Promotion

> 2026-07-19 update: point-in-time-safe structural terrain and candidate conversion layers were reactivated after an early-fold neutral-config bug was identified. Current metrics and adoption details are recorded in `docs/STRUCTURAL_LAYER_REACTIVATION_20260719.md` and supersede the performance figures below.

## Decision

The active presidential evaluation policy is `slot_free_hierarchy_no_neutral`.
It replaces the realized-rank A/B predictor path. The legacy outcome-aligned
slot model remains reproducible for historical comparison but is not active.

## Active sequence

1. Generate candidate order from strict rolling preliminary expected shares.
2. Regenerate the outcome-blind automatic issue seed from the current
   point-in-time Assembly interpretation overlay.
3. Fit Ridge with six slot-free predictors only.
4. Apply third-candidate structure adjustment.
5. Apply the final withdrawn-candidate transfer addition.
6. Apply sparse-history residual calibration when eligible.
7. Normalize each candidate-region contest to 100%.
8. Apply deterministic context and electorate layers with direct neutral
   context disabled.
9. Apply the weak national third-candidate hierarchy while preserving regional
   shape.
10. Select electorate preference gain from earlier outer folds only.
11. Apply the bounded direct mega-issue log-share shift when a pre-election
    intensity is above 1.0 and an explicit political-shock target exists.
12. Renormalize each regional contest to 100%.

The final transfer addition uses `withdrawn_candidate_transfer` with the fixed
active scale `0.15`. It is applied after Ridge and before contest
normalization. The 2002 Roh-Chung event experiment is not a learned predictor
in this active variant.

## Strict nested base plus bounded development postprocess

| Election | Regional weighted MAE | National point MAE |
|---|---:|---:|
| 2002 | 3.679%p | 2.485%p |
| 2007 | 12.430%p | 10.934%p |
| 2012 | 3.388%p | 2.681%p |
| 2017 | 6.789%p | 5.863%p |
| 2022 | 1.625%p | 0.881%p |
| Equal-election macro | **5.582%p** | **4.569%p** |

The Ridge fit, candidate assignment, and electorate-gain selection are strict
nested. The direct mega-issue gain `0.40` was promoted after through-2022
development comparison and was not selected inside each outer fold. Therefore
this table is a through-2022 development estimate with a strict nested base,
not an untouched holdout or a fully nested-selected postprocess estimate.

## Reproduction

```powershell
python scripts/build_preliminary_slot_assignments.py
python scripts/run_active_presidential_model.py
python presidential_issue_engine/audit_point_in_time.py
python -m pytest -q
```

Canonical outputs are under `outputs/active_presidential_nested/`.

The issue interpretation schema is `automatic_issue_interpretation_v2`.
Candidate direction is generated only by an explicit evaluated person, party,
or government target. Overall discussion tone and the speaker's party may
change evidence reliability, but cannot manufacture candidate polarity. The
implementation and audit record are in
`docs/ISSUE_INTERPRETATION_LAYER_V2_20260718.md`.

The direct mega-issue layer uses gain `0.40`, score cap `0.50`, log-shift cap
`0.20`, and an intensity gate strictly above `1.0`. In the current through-2022
inputs it activates only for `pres_2017 / B / regime_change`, producing a
log-share shift of `-0.101933`. It does not alter the other four elections.
This fixed gain was promoted after through-2022 development comparison, so the
new aggregate metric is not an untouched holdout estimate. See
`docs/DIRECT_MEGA_ISSUE_SHIFT_20260718.md`.
