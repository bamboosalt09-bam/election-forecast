# Structural layer reactivation audit (2026-07-19)

## Scope

- Active project: open-source presidential forecast only.
- Scored elections: 2002, 2007, 2012, 2017, 2022.
- Warm-up presidential election: 1997.
- Every historical terrain estimator filters source elections to dates strictly before its target.
- No 2025 presidential outcome is loaded or compared.
- These comparisons use 2002-2022 outcomes for development decisions and are therefore not an untouched holdout, even though every outer Ridge fit excludes its target election.

## Corrected integration error

The early-fold selector previously returned a neutral configuration whenever fewer than two earlier scored presidential elections existed. That rule also disabled terrain already estimated from earlier National Assembly and local elections. It confused two different restrictions:

1. Target presidential outcomes must not tune their own fold.
2. Point-in-time-safe earlier party and candidate ballots may define structural terrain for that fold.

The active policy now retains the second source while continuing to enforce the first restriction.

## Terrain definition

`terrain_anchor_gain` is not an independent candidate correction. It controls how strongly the existing electorate decomposition enters the forecast:

- `core`: durable concrete support estimated mainly from direct party ballots.
- `critical`: recent support above the durable floor that may defect on issues.
- `swing`: the remaining middle and floating electorate.

The existing fixed issue templates remain ordered so that core reacts least, critical support reacts more, and swing support reacts most. An additional layer-separation multiplier was rejected because it changed national MAE from 3.8605%p to 3.8618-3.8656%p.

Terrain strength is outcome-free within a forecast:

```text
raw gain = min(0.25, 0.50 * mean positive direct-party reliability)
final gain = raw gain / max(1.0, pre-election mega-issue intensity)
```

This retains reliable party terrain but prevents it from overpowering an exceptional national shock such as the 2017 impeachment election.

## Factor decisions

| Layer | Tested setting | Decision | Reason |
|---|---:|---|---|
| Continuous preliminary share predictor | enabled from one earlier scored presidential election | Reject | Improved 2007 but materially worsened 2012 and 2022; coefficient is unstable at tiny candidate-level N. |
| Candidate conversion | 0.05 | Adopt | Bounded PIT candidate/party conversion evidence; improved aggregate diagnostics. |
| Candidate regionalism | 0.15 | Adopt | Small net improvement and preserves a candidate's documented regional base. |
| Within-bloc transfer | 0.50 | Adopt | Improved both 2002 and 2007; transfer remains region-zero-sum. |
| Stronghold component | 0.25 | Adopt | Bounded part of the within-bloc transfer signal. |
| Third-candidate gate | enabled globally | Reject | Raised 2007 C and worsened national MAE. |
| Third-character multiplier | enabled globally | Reject | No general improvement in the active hierarchy path. |
| Preference response floor | 0.04 | Adopt | Fixed core/critical/swing templates improved both early folds. |
| Direct mega at intensity 1.0 | inclusive threshold | Reject | Selected a positive `security_nk` shift for 2007 B and worsened 2007 and 2022. |
| Extra layer separation | 0.25-1.00 | Reject | Existing layer sensitivities already provide the intended ordering. |

## Active result

| Metric | Previous active | Reactivated active | Change |
|---|---:|---:|---:|
| Regional contest-vote weighted macro MAE | 5.5821%p | 4.9850%p | -0.5971%p |
| National candidate macro MAE | 4.5686%p | 3.8756%p | -0.6930%p |
| Winner accuracy | 2/5 | 4/5 | +2 elections |

National predictions:

| Election | Candidate predictions | National candidate MAE |
|---|---|---:|
| 2002 | Roh 47.81, Lee Hoi-chang 52.19 | 3.41%p |
| 2007 | Lee Myung-bak 42.88, Chung Dong-young 39.93, Lee Hoi-chang 17.19 | 7.51%p |
| 2012 | Park 50.71, Moon 49.29 | 1.07%p |
| 2017 | Moon 39.73, Hong 37.42, Ahn 22.84 | 6.43%p |
| 2022 | Yoon 51.34, Lee Jae-myung 48.66 | 0.96%p |

The remaining 2007 error is primarily the national A:B mass split. Structural terrain now restores the correct winner and nearly matches the C share, but the candidate/public-treatment layer still rates Chung too close to Lee Myung-bak.

## Reproduction

```powershell
python scripts/evaluate_structural_layer_reactivation.py
python scripts/run_active_presidential_model.py
python -m pytest -q -p no:cacheprovider
python presidential_issue_engine/audit_point_in_time.py
python presidential_issue_engine/audit_weight_selection_boundary.py
```

Artifacts:

- `outputs/structural_layer_reactivation/summary.csv`
- `outputs/structural_layer_reactivation/by_election.csv`
- `outputs/structural_layer_reactivation/national_predictions.csv`
- `outputs/active_presidential_nested/terrain_anchor_by_fold.csv`
