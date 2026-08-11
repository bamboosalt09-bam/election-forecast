# Active V22: Automatic Policy, Issue, and Third-Candidate Controls

## Decision

V22 promotes the smallest automatic-control bundle that improved V21 under the
same strict nested pipeline. The active components are:

1. factual regional policy commitments with automatically derived effect strength;
2. Assembly-derived mega-issue class, numeric taxonomy, and class intensity;
3. automatic third-candidate profile, withdrawn-candidate landscape, and source-lane pressure as one bundle;
4. automatic economic and housing responsibility input tables.

The current pointer is `data/config/current_presidential_model.json`. Rollback is
available at `backups/model_checkpoints/20260802_pre_automatic_policy_v22`.

## Policy-Pledge Review

The policy layer is deliberately narrow. The current registry contains one dated
factual commitment: Roh Moo-hyun's 2002 administrative-capital commitment to the
Chungcheong region. The registry does not contain a fitted strength.

The compiler derives:

```text
affinity = sqrt(candidate_issue_association * election_normalized_issue_importance)
confidence = sqrt(source_quality * issue_importance_confidence)
weighted_affinity = affinity * confidence
```

For the registered row, the derived values are approximately `0.758`, `0.798`,
and `0.605`. Policy-only ablation improves 2002 regional MAE from `4.084150%p`
to `3.956813%p` and leaves every other election unchanged.

This is not yet a general manifesto parser. Additional elections must enter
through dated official manifesto facts using the same schema and formula. No
candidate- or election-specific effect strength may be added to the registry.

## Strict Nested Performance

| Election | V20 regional MAE | V21 regional MAE | V22 regional MAE | V22-V21 |
|---|---:|---:|---:|---:|
| 2002 | 3.636101%p | 4.084150%p | 3.986168%p | -0.097982%p |
| 2007 | 4.820084%p | 5.093073%p | 4.728944%p | -0.364129%p |
| 2012 | 2.194842%p | 2.459895%p | 2.459895%p | 0.000000%p |
| 2017 | 3.991872%p | 3.964735%p | 4.065126%p | +0.100391%p |
| 2022 | 1.443361%p | 1.373917%p | 1.374495%p | +0.000578%p |

| Model | Regional macro MAE | National diagnostic MAE | Winner accuracy |
|---|---:|---:|---:|
| V20 | 3.217252%p | 1.481393%p | 80% |
| V21 | 3.395154%p | 1.765729%p | 80% |
| V22 | 3.322926%p | 1.677567%p | 80% |

V22 recovers part of the V20-to-V21 loss and materially improves 2007. It does
not restore V20's aggregate accuracy and slightly worsens 2017. The automation
bundle was selected after comparing 2002-2022 development outcomes, so these
numbers are not an untouched holdout performance claim.

## Automation Boundary

| Input | Status | Reason |
|---|---|---|
| Mega-issue intensity and numeric character | Active | Universal Assembly-evidence class thresholds |
| Economic/housing responsibility | Active input | Structural incumbency plus explicit government-target discourse; currently no numerical change in the selected six-predictor model |
| Third-candidate profile | Active | Prior elections, direct-party ballots, candidate history, Assembly role, and political axes |
| Withdrawn-candidate political landscape | Active | Low-rank axes derived from the automatic third profile |
| Third-candidate absorption pressure | Active | Must be activated with the profile and landscape; pressure-only ablation failed |
| Generation electorate composition | Shadow | Lagged NEC reports are sparse; regional MAE worsened by about 0.005%p |
| Party voter-retention rate | Shadow | Current transition rows lack usable before/after direct-party ballot pairs |
| Transfer rate and voter compliance | Semiautomatic | Scenario parameters, not point estimates learned from two heterogeneous withdrawals |

The automatic generation and behavioral-retention compilers remain implemented
under `outputs/automatic_controls_v22/`; shadow status means they are available
for evidence expansion but are not read by the active forecast.

## Validation

- canonical/ablation `nested_predictions.csv` SHA-256 match;
- strict deep point-in-time audit: PASS;
- outcome invariance rows: `215/215`;
- active input-manifest paths containing `2025`: `0`;
- automatic-control rows for `pres_2025`: `0`;
- test suite: `462 passed`;
- target-outcome fields used by new layers: none.

## Artifacts

- `scripts/run_active_presidential_model_v22.py`
- `data/config/active_presidential_model_v22.json`
- `presidential_issue_engine/automatic_controls_v22.py`
- `presidential_issue_engine/regional_policy_commitment.py`
- `data/raw/regional_policy_commitments.csv`
- `outputs/automatic_controls_v22/`
- `outputs/automatic_controls_v22_ablation/`
- `outputs/active_presidential_nested_v22/`

