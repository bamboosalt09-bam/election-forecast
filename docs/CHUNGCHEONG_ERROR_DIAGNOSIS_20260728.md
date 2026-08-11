# Chungcheong Error Diagnosis (2026-07-28)

## Status

- Active model: `active_strict_nested_v15_chungcheong_identity`
- This document is a post-election diagnostic. Actual outcomes are used only
  for scoring and error decomposition, never as forecast inputs.
- The first two remediation experiments were rejected. The third, VIF-gated
  regional-offset experiment, was promoted as v14.
- 2025 outcomes are not used.

## Frozen baseline

| Metric | Active v13 |
|---|---:|
| Regional contest-vote weighted equal-election macro MAE | 3.573555%p |
| National candidate equal-election macro MAE | 2.078219%p |
| Winner accuracy | 80% (4/5) |
| Regional prediction rows | 199 |

Regional error is weighted by the target contest's valid votes inside each
election. The five election-level errors are then averaged equally. The
region table below instead pools contest-vote weights over the rows available
for each region, so it diagnoses geographic concentration rather than the
official headline metric.

## Regional performance table

| Region | Weighted MAE | Weighted RMSE |
|---|---:|---:|
| Ulsan | 5.8635%p | 6.5843%p |
| Jeju | 5.8477%p | 6.6078%p |
| Daejeon | 5.3424%p | 5.9071%p |
| Jeonbuk | 4.9992%p | 5.9396%p |
| Chungnam | 4.9978%p | 6.7611%p |
| Sejong | 4.9485%p | 6.1099%p |
| Jeonnam | 4.6250%p | 5.3769%p |
| Gwangju | 4.1627%p | 5.1032%p |
| Chungbuk | 4.1132%p | 5.5307%p |
| Incheon | 3.5923%p | 4.1285%p |
| Seoul | 3.4498%p | 4.3306%p |
| Gyeongnam | 3.3974%p | 3.8377%p |
| Gangwon | 3.1683%p | 3.4758%p |
| Daegu | 3.1136%p | 3.6796%p |
| Gyeonggi | 3.0787%p | 4.1355%p |
| Gyeongbuk | 3.0170%p | 3.9366%p |
| Busan | 2.7483%p | 3.3992%p |

The machine-readable source is
`outputs/chungcheong_error_audit_v14/regional_performance.csv`.

## Cross-election Chungcheong pattern

The model underpredicts the eventual national winner in 17 of 18 available
Chungcheong region-election rows. The mean unweighted bias over these rows is
`-5.2281%p`. This occurs for liberal and conservative winners alike.

| Election | Region | National winner | Error |
|---|---|---|---:|
| 2002 | Daejeon | Roh Moo-hyun | -9.8980%p |
| 2002 | Chungbuk | Roh Moo-hyun | -9.1989%p |
| 2002 | Chungnam | Roh Moo-hyun | -5.2025%p |
| 2007 | Daejeon | Lee Myung-bak | -1.4511%p |
| 2007 | Chungbuk | Lee Myung-bak | -4.0280%p |
| 2007 | Chungnam | Lee Myung-bak | -0.0167%p |
| 2012 | Daejeon | Park Geun-hye | -6.2067%p |
| 2012 | Sejong | Park Geun-hye | -10.8652%p |
| 2012 | Chungbuk | Park Geun-hye | -8.7607%p |
| 2012 | Chungnam | Park Geun-hye | -12.6882%p |
| 2017 | Daejeon | Moon Jae-in | -7.0833%p |
| 2017 | Sejong | Moon Jae-in | -9.6819%p |
| 2017 | Chungbuk | Moon Jae-in | -0.2973%p |
| 2017 | Chungnam | Moon Jae-in | -3.6517%p |
| 2022 | Daejeon | Yoon Suk-yeol | -2.8364%p |
| 2022 | Sejong | Yoon Suk-yeol | -1.8635%p |
| 2022 | Chungbuk | Yoon Suk-yeol | +0.1236%p |
| 2022 | Chungnam | Yoon Suk-yeol | -0.4990%p |

This table does not imply that a future model may use the eventual winner.
It identifies a retrospective symptom: the regional allocator does not expand
the forecast-leading camp enough in Chungcheong.

## 2012 mechanism

The largest local miss is 2012 Chungnam. The final forecast gives Park
Geun-hye `44.2850%` against an actual `56.9732%`. However, the strictly prior
direct-party terrain already allocates `56.1112%` of the two-major-camp base to
the conservative side. The information exists before the target election, but
the Ridge prediction and flexible-mass allocation reverse its direction.

| Region | Park prior direct-party split | Park final prediction | Park actual |
|---|---:|---:|---:|
| Daejeon | 52.8319% | 43.9194% | 50.1261% |
| Sejong | 42.7895% | 41.3096% | 52.1748% |
| Chungbuk | 55.2439% | 47.7546% | 56.5152% |
| Chungnam | 56.1112% | 44.2850% | 56.9732% |

Sejong remains a sparse-history exception: its prior direct-party split itself
points in the wrong direction. It therefore needs hierarchical borrowing from
the wider Chungcheong history rather than a stronger local anchor.

## Collinearity diagnosis

The six active predictors are manually fixed, but their effective basis is not
stable in the early rolling folds. For the 2012 fold, trained only through
2007, `issue_advantage` and `rif` are exact duplicates under the strict neutral
curated-input policy and therefore have infinite VIF. After recording that
basis defect separately, the largest finite VIF is `272.885`.

| Predictor | 2012-fold VIF |
|---|---:|
| issue_advantage | infinity |
| rif | infinity |
| partisan_prior | 1.0014 |
| landscape_bloc_alignment | 11.4366 |
| landscape_centrist | 272.8849 |
| landscape_inferred_prior | 1.0019 |

Ridge makes the inversion numerically solvable, but it does not make individual
coefficient interpretation stable. This explains part of the 2012 reversal;
it is not sufficient on its own to explain all Chungcheong errors.

## Remediation experiments

### Fold-local predictor orthogonalization

Two correlated pairs were residualized using training `X` only:

- `issue_advantage -> rif`
- `landscape_bloc_alignment -> landscape_centrist`

No response or target-election row entered the transform. Chungcheong improved,
but 2012 collapsed globally, so the experiment was rejected.

| Metric | v13 | Orthogonalized | Change |
|---|---:|---:|---:|
| Regional macro MAE | 3.5807%p | 5.1106%p | +1.5298%p |
| National macro MAE | 2.0791%p | 3.9001%p | +1.8210%p |
| Chungcheong macro MAE | 5.4002%p | 4.8888%p | -0.5114%p |

### Bounded direct-party center

The second experiment preserved minor and independent candidate shares, then
partly moved only the two eligible major-party candidates toward their prior
direct-party split. The gain was reliability-weighted and attenuated for mega
shocks. It improved 2007 and 2012 but harmed 2002 and 2022, so it was rejected.

| Metric | v13 | Direct-party center | Change |
|---|---:|---:|---:|
| Regional macro MAE | 3.5807%p | 3.7445%p | +0.1637%p |
| National macro MAE | 2.0791%p | 2.0984%p | +0.0193%p |
| Chungcheong macro MAE | 5.4002%p | 5.2762%p | -0.1240%p |
| Worst election regression | - | - | +0.6970%p |

These failures are useful constraints: neither unconditional basis rotation nor
a stronger static party anchor is a valid general solution.

### VIF-gated non-presidential regional offset

Rolling validation on Assembly PR, metropolitan-council PR, and local-council
PR elections selected a hierarchical regional logit offset over a full swing
slope. The offset MAE was `5.3848%p`, compared with `5.7290%p` for elasticity
and `16.2864%p` for a flat national share.

The presidential layer activates only after two prior scored elections and
only when the largest finite fold VIF exceeds `20`. Its maximum gain is `0.25`,
with shock attenuation and profile-reliability shrinkage. It activated only in
2012; 2002, 2007, 2017, and 2022 are exact no-ops.

| Metric | v13 | Promoted v14 | Change |
|---|---:|---:|---:|
| Regional macro MAE | 3.5807%p | 3.5736%p | -0.0072%p |
| National macro MAE | 2.0791%p | 2.0782%p | -0.0009%p |
| Chungcheong macro MAE | 5.4002%p | 5.2292%p | -0.1710%p |
| Worst election regression | - | 0.0000%p | none |

The improvement is deliberately modest. Its value is that it reduces the
confirmed Chungcheong error without moving any stable fold.

### Required all-fold stress test

The same offset was also forced onto every fold with an available prior
profile. This keeps PIT filtering and outcome exclusion but bypasses the VIF
gate. The test rejects universal application.

| Metric | Active VIF gate | All available folds | Change |
|---|---:|---:|---:|
| Regional macro MAE | 3.5736%p | 3.6407%p | +0.0671%p |
| National macro MAE | 2.0782%p | 2.1261%p | +0.0479%p |
| Chungcheong macro MAE | 5.2292%p | 5.1865%p | -0.0427%p |

| Election | Regional MAE change | National MAE change |
|---|---:|---:|
| 2002 | 0.0000%p | 0.0000%p |
| 2007 | +0.0661%p | +0.1034%p |
| 2012 | 0.0000%p | 0.0000%p |
| 2017 | -0.0068%p | +0.0308%p |
| 2022 | +0.2764%p | +0.1052%p |

The nominal 2002 gain is a no-op because no eligible prior direct-party profile
exists. Active v14 already applies the same correction to 2012. Broad
application slightly improves 2017 regional shape and 2007 Chungcheong, but it
double-corrects the already stable 2022 allocation. This confirms that the
offset is a fold-instability fallback, not a universal regional multiplier.

## Implemented solution and next limit

The promoted mechanism is a separately trained, point-in-time regional-offset
fallback:

1. Build event-level conservative/liberal two-camp shares from Assembly PR,
   metropolitan-council PR, and local-council PR elections.
2. For each region, estimate its stable two-camp log-odds offset from the
   national result. Partially pool it toward zero; sparse Sejong history borrows
   from the Chungcheong hierarchy.
3. Select shrinkage and type weights only by rolling holdout on those
   non-presidential direct-party elections. Presidential outcomes must not tune
   this layer.
4. At forecast time, use the presidential model's own national camp signal as
   input and use the frozen offset only to distribute that signal across
   regions. Preserve the modeled third-candidate pool exactly.
5. Require strict nested PIT filtering and reject promotion unless aggregate
   regional and national MAE improve, Chungcheong improves, and no election
   worsens materially.

The data did not support a distinct regional swing slope yet. That extension is
deferred until more direct-party events are available. The current fallback
does not encode a preferred party, a named candidate, the actual winner, or a
Chungcheong bonus.

## Correction: the missing layer is regional identity, not a camp offset

The all-fold stress test above answers only whether the conservative-versus-
liberal regional offset should be applied more broadly. It does **not** test
the more specific Chungcheong hypothesis. Yeongnam and Honam are represented
directly by durable conservative or liberal lineage mass. Chungcheong's
historical regional-party mass has no equivalent destination in the active
model.

Strictly prior direct-party ballots show a large Chungcheong-specific third-
bloc excess. For example, the regional third-bloc share minus the median among
all available regions was approximately:

| Election | Daejeon | Chungbuk | Chungnam |
|---|---:|---:|---:|
| 2004 Assembly PR | 13.3%p | 5.1%p | 22.6%p |
| 2006 local-council PR | 18.7%p | 2.3%p | 32.2%p |
| 2008 Assembly PR | 29.9%p | 9.2%p | 33.3%p |
| 2010 metro-council PR | 31.2%p | 4.8%p | 32.3%p |
| 2012 Assembly PR | 15.4%p | 2.9%p | 18.0%p |

This mass is currently reduced to either a generic `centrist` accent or a
weak `regionalist` speech-landscape match. Some source rows, including the
large 2008/2010 regional-party vote, are already normalized to `third_bloc`,
so the original regional-party lineage is no longer available to the accent
matcher. The explicit candidate-base table helps a documented candidate such
as Lee Hoi-chang in 2007, but it is a candidate concentration adjustment, not
a durable Chungcheong terrain layer.

The active Chungcheong weighted MAE pattern is consistent with this omission:

| Election | Chungcheong weighted row MAE |
|---|---:|
| 2002 | 7.8340%p |
| 2007 | 4.6301%p |
| 2012 | 9.5450%p |
| 2017 | 3.0285%p |
| 2022 | 1.1081%p |

The next experiment must therefore estimate a PIT-safe **regional-identity
reservoir** from prior direct-party elections. It must not classify that
reservoir as major-party concrete support. Candidate routing must be a separate
step based only on pre-election evidence: documented personal regional base,
dated party merger/endorsement or coalition lineage, and directional regional-
development issue ownership. If no recipient is supported by those inputs, the
reservoir remains critical/swing mass instead of being assigned mechanically.
The experiment must be evaluated on every 2002-2022 strict nested fold; no
single-election Chungcheong correction is eligible for promotion.

## V15 implementation result

The regional-identity design above was implemented and promoted as
`active_strict_nested_v15_chungcheong_identity`.

- Reservoir: strictly prior third/regional-party share above the cross-region
  median; direct-party ballots have full weight and presidential ballots have
  weight `0.35`.
- Stability: 12-year half-life, partial-pooling strength `1.5`, and volatility
  reliability.
- Routing: candidate regional base or dated pre-election alignment only.
- Transfer: gain `0.50`, at most `8%p` in one region, exact regional mass
  conservation.
- No evidence: no transfer; reservoir remains critical/swing.
- Outcome fields used by the layer: none.

| Metric | V14 | V15 | Change |
|---|---:|---:|---:|
| Regional macro MAE | 3.5736%p | 3.3953%p | -0.1783%p |
| National macro MAE | 2.0782%p | 1.8483%p | -0.2299%p |
| Chungcheong macro MAE | 5.2292%p | 3.5944%p | -1.6347%p |
| Winner accuracy | 80% | 80% | unchanged |

Regional MAE changes are `-0.2504%p` in 2002, `-0.1249%p` in 2007,
`-0.5162%p` in 2012, and exactly zero in 2017 and 2022. Gain sensitivity at
`0.25` and `0.75` also improves all aggregate metrics, so the result is not a
single-gain artifact. The promoted `0.50` is the originally tested middle
setting, not the numerically best retrospective setting.

The active artifact is `outputs/active_presidential_nested_v15/`; the complete
per-region reservoir, reliability, evidence, and transfer audit is
`chungcheong_identity_audit.csv`.

The promoted code, configuration, inputs, active output, experiment output,
diagnostic output, and sensitivity summaries are frozen under
`archives/experiments/chungcheong_identity_v15_20260728/`; its
`archive_manifest.csv` contains 60 SHA-256-checked files.

## Reproduction artifacts

- `scripts/audit_chungcheong_error.py`
- `outputs/chungcheong_error_audit_v14/summary.json`
- `outputs/chungcheong_error_audit_v14/regional_performance.csv`
- `outputs/chungcheong_error_audit_v14/chungcheong_national_winner_errors.csv`
- `outputs/chungcheong_error_audit_v14/pres_2012_chungcheong_stage.csv`
- `outputs/chungcheong_error_audit_v14/pres_2022_cancellation.csv`
- `outputs/chungcheong_error_audit_v14/vif_pooled.csv`
- `outputs/chungcheong_error_audit_v14/vif_by_fold.csv`
- `outputs/predictor_orthogonalization_v14_experiment/decision.json`
- `outputs/direct_party_center_v14_experiment/decision.json`
- `outputs/regional_swing_elasticity_nonpresidential/decision.json`
- `outputs/vif_gated_regional_offset_v14_experiment/decision.json`
- `outputs/all_fold_regional_offset_v14_experiment/decision.json`
