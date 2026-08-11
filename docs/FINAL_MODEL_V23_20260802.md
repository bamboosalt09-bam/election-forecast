# Final Presidential Forecast Model V23

## Status

V23 is the frozen through-2022 presidential forecast baseline. It is the only
model invoked by `scripts/run_current_presidential_model.py` and the only model
identified by `data/config/current_presidential_model.json`.

- rolling warmup: 1997 presidential election;
- scored development elections: 2002, 2007, 2012, 2017, 2022;
- 2025 outcomes: prohibited from fitting, tuning, model selection, and
  pre-evaluation comparison;
- active evaluation: strict chronological nested pipeline;
- lifecycle: frozen before the 2025 evaluation.

The five scored elections are a development sample, not an untouched holdout.
Any future methodological change must be developed as V24 or later without
modifying V23 in place.

## Model Pipeline

1. Candidate identities, party roles, factual withdrawal events, and dated
   party lineage are resolved into canonical registries.
2. Preliminary A/B/C roles are estimated from strictly prior expected strength.
   Realized target-election rank is not used.
3. Regional partisan terrain is reconstructed from prior presidential,
   National Assembly, proportional, and local-election evidence. Direct party
   ballots receive greater reliability than candidate-personal ballots.
4. Assembly records produce unsigned issue salience, candidate/party links,
   explicitly attributed direction, issue character, and evidence confidence.
   Uncertain directional sentences do not directly move vote share.
5. Every outer fold fits Ridge after excluding the target election. The six
   active predictors are:
   - `issue_advantage`;
   - `rif`;
   - `partisan_prior`;
   - `landscape_bloc_alignment`;
   - `landscape_centrist`;
   - `landscape_inferred_prior`.
6. The predicted flexible vote is combined with conservative major-party core,
   critical-support, and swing masses. These layers have different issue and
   regime sensitivity.
7. Evidence-gated postprocessing applies regional identity, candidate
   conversion, third-candidate competition, mega-issue shock, incumbent burden,
   cumulative rejection, and within-bloc tactical transfer.
8. Withdrawn-candidate transfers are read from one registry. Consumer-specific
   fields prevent coalition, feature, and preliminary-transfer semantics from
   being counted twice.
9. Candidate shares are normalized compositionally within each region, so each
   modeled candidate set sums to 100%.

## Automation Boundary

| Component | Final status |
|---|---|
| Candidate viability and political profile | Automatic with evidence shrinkage |
| Third-candidate profile and pressure | Automatic |
| Regional and party terrain | Automatic from strictly prior election history |
| Issue salience, taxonomy, intensity, and attribution | Automatic from dated Assembly evidence |
| Economic and housing responsibility inputs | Automatic dated inputs |
| Generation composition | Latest strictly prior official report; neutral fallback when absent |
| Candidate withdrawal profile | Same canonical automatic candidate profile |
| Withdrawal event facts | Factual registration |
| Transfer rate and voter compliance | Universal low/medium/high scenarios |
| Legal party rename/merger continuity | Factual registration |
| Electoral support continuity across party transitions | Conservative prior/shrinkage; not fully learned |
| Model gains, caps, and Ridge policy | Frozen development-selected hyperparameters |

The remaining semiautomatic items are deliberate. Historical withdrawal events
are too sparse for stable point estimation. The scenario rules are common to
all elections and cannot be freely tuned by election.

## Candidate And Withdrawal Resolution

The canonical profile is
`outputs/automatic_controls_v23/candidate_political_profiles.csv`.

- 2022 Ahn Cheol-soo uses strictly prior 2017 same-person evidence and shrinks
  toward neutral: viability `0.521338`, centrist appeal `0.644248`, anti-major
  appeal `0.600900`, regional overlap `0.268980`, confidence `0.572269`.
- 2012 Ahn Cheol-soo uses pre-withdrawal Assembly target-mention breadth for
  stature only: viability `0.734506`, confidence `0.724091`. With no valid
  directional evidence, centrist and anti-major traits use the universal
  neutral fallback `0.5/0.5`.
- 2002 Chung Mong-joon uses strictly prior candidate election history.

The only active transfer input is
`outputs/automatic_controls_v23/withdrawal_transfer_registry.csv`. The legacy
transfer CSVs remain historical files and are absent from the active input
manifest.

## Performance

Primary strict nested diagnostics:

- regional contest-vote-weighted equal-election macro MAE: `3.367899%p`;
- national candidate equal-election macro MAE: `1.597845%p`;
- winner accuracy: `4/5` (`80%`);
- prediction rows: `199`.

| Election | Regional weighted MAE | National candidate MAE |
|---|---:|---:|
| 2002 | 3.981641%p | 3.322040%p |
| 2007 | 4.759601%p | 1.676099%p |
| 2012 | 2.666917%p | 0.162674%p |
| 2017 | 4.065777%p | 2.652642%p |
| 2022 | 1.365557%p | 0.175768%p |

National candidate diagnostics:

| Election | Candidate | Predicted | Actual |
|---|---|---:|---:|
| 2002 | Roh Moo-hyun | 47.8950% | 51.2170% |
| 2002 | Lee Hoi-chang | 52.1050% | 48.7830% |
| 2007 | Lee Myung-bak | 52.1139% | 54.1403% |
| 2007 | Chung Dong-young | 31.6029% | 29.0888% |
| 2007 | Lee Hoi-chang | 16.2832% | 16.7709% |
| 2012 | Park Geun-hye | 51.9360% | 51.7733% |
| 2012 | Moon Jae-in | 48.0640% | 48.2267% |
| 2017 | Moon Jae-in | 45.2863% | 47.4759% |
| 2017 | Hong Joon-pyo | 31.7521% | 27.7731% |
| 2017 | Ahn Cheol-soo | 22.9616% | 24.7510% |
| 2022 | Yoon Suk-yeol | 50.2039% | 50.3796% |
| 2022 | Lee Jae-myung | 49.7961% | 49.6204% |

The national diagnostic aggregates regional predictions using realized turnout
only for post-hoc evaluation. It is not itself a deployable pre-election turnout
forecast.

## Validation Record

- full test suite: `471 passed`;
- strict PIT audit: PASS;
- target-outcome mutation invariance: `215/215` rows;
- through-2022 weight-selection boundary audit: PASS;
- V22 exact reproduction: PASS;
- V23 promotion gate: PASS;
- active and gated-experiment prediction SHA-256: identical;
- active legacy transfer inputs: none;
- 2025 rows or target-outcome flags in V23 automatic outputs: none;
- active prediction SHA-256:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`.

## Reproduction

```powershell
cd C:\english_folder\poll_project
python scripts\run_current_presidential_model.py
python scripts\audit_active_presidential_model_v23.py
python presidential_issue_engine\audit_point_in_time.py --deep
python presidential_issue_engine\audit_weight_selection_boundary.py
python -m pytest -q
```

Do not run the long Assembly reprocessing pipeline for ordinary reproduction.
V23 consumes the completed extraction and its hashed derived inputs.

## Canonical Artifacts

- active pointer: `data/config/current_presidential_model.json`;
- active configuration: `data/config/active_presidential_model_v23.json`;
- runner: `scripts/run_active_presidential_model_v23.py`;
- automatic-control lineage: `outputs/automatic_controls_v23/lineage_manifest.json`;
- strict nested predictions: `outputs/active_presidential_nested_v23/nested_predictions.csv`;
- input hashes: `outputs/active_presidential_nested_v23/input_manifest.csv`;
- promotion record: `outputs/active_presidential_nested_v23/promotion_manifest.json`;
- finalization hashes and verification record:
  `outputs/active_presidential_nested_v23/finalization_manifest.json`;
- detailed V23 decision record:
  `docs/ACTIVE_V23_UNIFIED_WITHDRAWAL_GENERATION_20260802.md`;
- pre-V23 rollback checkpoint:
  `backups/model_checkpoints/20260802_pre_v23_unified_withdrawal_generation`;
- verified V23 checkpoint:
  `backups/model_checkpoints/20260802_active_v23_unified_withdrawal_generation`.

## Freeze Policy

V23 is closed for retrospective tuning. New data importers, alternative transfer
models, electoral-continuity estimators, or hyperparameter changes must write to
new versioned outputs and pass the same strict nested, PIT, manifest, and
rollback requirements before the active pointer can change.
