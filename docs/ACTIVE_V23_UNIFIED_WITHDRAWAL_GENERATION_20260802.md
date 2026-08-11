# Active V23 Unified Withdrawal And Generation Controls

## Decision

V23 promotes the strict nested candidate-profile, withdrawal-registry, and
generation-composition bundle evaluated in
`outputs/automatic_controls_v23_ablation_v3`. The active runner is
`scripts/run_active_presidential_model_v23.py`; the canonical output is
`outputs/active_presidential_nested_v23`.

The scored development elections remain 2002, 2007, 2012, 2017, and 2022, with
1997 used only as rolling warmup. No 2025 outcome, row, comparison, or tuning
signal is used.

## Automatic Candidate Profile

`outputs/automatic_controls_v23/candidate_political_profiles.csv` is the one
canonical profile table for final third candidates and withdrawn preliminary
candidates.

- 2022 Ahn Cheol-soo is derived from the strictly prior 2017 same-person
  profile. Evidence decay and confidence shrink the traits toward neutral:
  viability `0.521338`, centrist appeal `0.644248`, anti-major-party appeal
  `0.600900`, regional overlap `0.268980`, confidence `0.572269`.
- 2012 Ahn Cheol-soo has no prior election profile. Pre-withdrawal Assembly
  target-mention breadth supplies stature only: viability `0.734506` and
  confidence `0.724091`. Centrist and anti-major traits remain the common
  neutral fallback `0.5/0.5`; regional overlap remains `0`.
- 2002 Chung Mong-joon uses strictly prior Assembly election history with
  evidence shrinkage.

Target-mention breadth is not sentiment, support, or issue direction. The
compiler is prohibited from using it for those fields.

## Unified Withdrawal Registry

`outputs/automatic_controls_v23/withdrawal_transfer_registry.csv` is the only
active prediction registry. Candidate traits come from the canonical profile;
`data/raw/withdrawal_events.csv` contains event facts only.

The registry has explicit consumer switches and consumer-specific fields for:

- coalition issue-feature transfer;
- withdrawn-candidate feature transfer;
- preliminary latent-candidate redistribution.

Transfer and compliance are common low/medium/high scenarios selected from
event timing and formal endorsement. This remains semiautomatic because the
historical event sample is too small for stable point estimation. The same
scenario rules apply to every election; election-specific free point tuning is
not allowed.

The legacy files below are isolated and absent from the active input manifest:

- `data/raw/withdrawn_candidate_transfers.csv`
- `data/raw/withdrawal_event_profiles.csv`
- `presidential_issue_engine/fixed_dataset/coalition_events.csv`

An audit found that the preliminary ballot assembler previously read a legacy
withdrawn-transfer file while temporarily disabling coalition events. The read
did not change the validated predictions, but it violated single-lineage
reproducibility. V23 now supplies explicit empty temporary coalition and
withdrawal inputs during that assembly step and restores all engine paths
afterward.

## Generation Composition

`outputs/automatic_controls_v23/election_generation_weights.csv` uses the most
recent official age-turnout composition published strictly before each target
election. Weights sum to one. Sparse early elections fall back to the common
uninformed prior.

For 2022, the active values are the 2017 report values: young `0.173`, middle
`0.368`, senior `0.459`. The 2022 report is post-election evidence for that fold
and is not used.

## Strict Nested Ablation

| Variant | Regional weighted macro MAE | National point macro MAE | Winner accuracy |
|---|---:|---:|---:|
| Active V22 | 3.322926%p | 1.677567%p | 80% |
| V22 exact reproduction | 3.322926%p | 1.677567%p | 80% |
| Automatic profile + legacy transfer + generation | 3.327787%p | 1.689892%p | 80% |
| Unified profile + registry + generation | 3.367899%p | 1.597845%p | 80% |

The final bundle passes the predeclared conservative gate:

- regional degradation `+0.044973%p`, cap `0.10%p`;
- national change `-0.079723%p`, cap `+0.10%p`;
- maximum one-election regression `+0.207022%p`, cap `0.25%p`;
- no winner-accuracy regression.

## Election Diagnostics

| Election | Regional weighted MAE | National candidate MAE | Regional change vs V22 |
|---|---:|---:|---:|
| 2002 | 3.981641%p | 3.322040%p | -0.004527%p |
| 2007 | 4.759601%p | 1.676099%p | +0.030657%p |
| 2012 | 2.666917%p | 0.162674%p | +0.207022%p |
| 2017 | 4.065777%p | 2.652642%p | +0.000651%p |
| 2022 | 1.365557%p | 0.175768%p | -0.008937%p |

V23 is therefore a methodology and reproducibility promotion, not a claim that
every diagnostic improved. The principal cost is the 2012 regional regression;
the aggregate national diagnostic improves.

## Validation

- `python -m pytest -q`: `471 passed`.
- `python presidential_issue_engine/audit_point_in_time.py --deep`: PASS,
  target-outcome invariance `215/215`.
- `python presidential_issue_engine/audit_weight_selection_boundary.py`: PASS.
- `python scripts/audit_active_presidential_model_v23.py`: PASS; active input
  hashes, cutoff dates, no 2025 rows, no target-outcome flags, strict generation
  lag, legacy-transfer isolation, active/experiment SHA equality, and V23
  outcome invariance are checked.
- Active prediction SHA-256:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`.

## Limitations

The 2002-2022 elections are the development sample. The promotion decision is
outcome-aware and is not an untouched historical holdout. Withdrawal scenario
levels remain semiautomatic, and early generation composition relies on a
neutral fallback where no strictly prior official report exists. These limits
must remain visible in any external performance claim.
