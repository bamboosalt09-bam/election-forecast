# Final active presidential model V27

## Promotion

- Active version: V27
- Predecessor and rollback: frozen V26
- Historical runner: `python scripts/run_active_presidential_model_v27.py`
- Prospective runner: `python scripts/run_prospective_forecast_v27.py`
- Public audit: `python scripts/audit_public_active_presidential_model_v27.py`
- Post-2022 outcomes used: none

V27 retains V26's graded mega-issue ladder and event-class alignment and adds
one terminal regional-shape layer at gain 1.0 after the frozen V26 stack. No Ridge predictor, historical ballot,
candidate national level, winner rule, shock coefficient or third-candidate
rule changes.

## Added layer

The point-in-time `recent_bloc_base` supplies inherited party-regional width.
When the fitted candidate map is narrower, V27 restores the fraction of the
missing logit dispersion supported by measured concrete mass and direct-party
reliability:

    factor = 1 + core_mass * direct_party_reliability
                   * max(0, prior_sd / fitted_sd - 1)

The fitted regional ordering remains authoritative.  Candidate vote-weighted
national shares and each region's 100% composition are restored exactly after
the transform.  Gain 1 is structural, not the panel optimum: it means that the
evidenced concrete share retains its proportional part of inherited regional
dispersion.

## Development-panel result

| metric | V26 | V27 |
| --- | ---: | ---: |
| equal-election regional weighted MAE | 2.7122%p | **2.6139%p** |
| equal-election national candidate MAE | 0.7210%p | **0.7210%p** |
| winner accuracy | 4/5 | 4/5 |
| rows | 232 | 232 |

The transform materially improves 2007 and 2017, slightly improves 2022, and
leaves 2002 and 2012 effectively unchanged.  The historical panel remains a
development panel and does not establish future accuracy.

## Prospective boundary

The 2025 demonstration uses the D-1 cutoff and prior-2022 regional vote volume
for the conservation weights.  It does not read or score the realised 2025
outcome.  V27 preserves the V26 candidate national allocation exactly and
changes regional shape only.  Because the 2025 path was repaired after its
outcome was known, it remains a corrected demonstration rather than genuine
out-of-sample evidence.

## Selection disclosure

The regionalism defect and the tested mechanism were developed by inspecting
the same five historical residual sets used for evaluation.  Gain 3.0 was the
panel minimum and was rejected as outcome-selected.  Gain 1.0 was promoted
because it is the formula's parameter-free one-for-one interpretation.  Future
changes require a new version; V23 through V27 must not be edited in place.
