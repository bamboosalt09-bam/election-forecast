# Automatic mega-issue intensity v5

## Objective

Replace the five manually assigned election-level values in
`mega_issue_intensity.csv` with a deterministic pre-election signal. The
experiment uses no presidential outcome field and no post-2022 row.

## Speech evidence

The compiler reads 195,758 dated rows from the archived through-2022 Assembly
speaker-issue match table. For the `regime_change` axis it measures:

- share of all weighted issue evidence;
- phrase-weight density per matched speech row;
- share of distinct speakers discussing the axis;
- simultaneous `corruption_integrity` activation.

The saturation references are universal: 15% issue share, phrase density 1.0,
and 75% speaker coverage. Their geometric joint evidence is converted to an
intensity in `[0.5, 2.0]` with a conservative quadratic response.

| Election | Speech-only intensity |
|---|---:|
| 2002 | 0.645005 |
| 2007 | 1.044210 |
| 2012 | 0.849162 |
| 2017 | 1.571218 |
| 2022 | 1.200186 |

The issue weights are lexicon-assisted. This removes the manual election
scalar, but it does not make the upstream political phrase vocabulary fully
automatic.

## Discontinuous-gate defect

The first strict run exposed a structural defect in
`compile_direct_mega_scores`: any intensity just above `1.0` activated the full
direct mega-issue shift. Values `1.044` in 2007 and `1.200` in 2022 therefore
caused very large discontinuous changes.

The original hard-gate result was:

| Variant | Regional MAE | National MAE |
|---|---:|---:|
| Manual candidate-v2 baseline | 3.334606%p | 1.781094%p |
| Automatic speech, hard gate | 5.933751%p | 4.960639%p |

Those 49 artifacts are preserved under
`archives/experiments/automatic_mega_intensity_v5_hard_gate_20260729`.

The consumer now multiplies the existing intensity-scaled direct score by
`clip(intensity - minimum_intensity, 0, 1)`. With the default minimum of 1.0,
the effect ramps continuously from zero at 1.0 to its previous full value at
2.0. The active manual input has only one row above the gate, 2017 at 2.0, so
the active output is unchanged byte for byte.

## Strict nested results after smoothing

| Variant | Regional MAE | National MAE | Winner accuracy |
|---|---:|---:|---:|
| Manual candidate-v2 baseline | 3.334606%p | 1.781094%p | 80% |
| Neutral intensity | 4.437678%p | 3.169054%p | 60% |
| Automatic speech only | 3.778872%p | 2.606802%p | 80% |

The smooth consumer fixes the cliff, but speech evidence alone understates the
2017 institutional crisis and still overactivates 2007 and 2022.

## Dated event-class gate

A second predeclared variant reads only `election_id`, `shock_type`, and
`available_date` from `mega_issue_taxonomy.csv`. It explicitly does not read
manual `severity`, `national_scope`, `persistence`, `polarization`,
`target_specificity`, or `confidence` values. Universal semantic levels gate
the speech evidence: institutional crisis, state capture, accountability or
coalition realignment, and candidate or policy events.

This produces intensities `0.687`, `1.042`, `0.934`, `1.768`, and `1.115` for
2002-2022. Its strict nested result is `3.456403%p` regional and `2.016320%p`
national MAE with 80% winner accuracy. It is materially better than the
speech-only version but remains worse than the manual baseline, especially in
2017.

## Decision

Do not promote either automatic intensity file. Retain the compiler and dated
event-class gate as experimental components. Retain the continuous activation
fix because it removes a genuine discontinuity and reproduces the current
active v16 prediction file exactly: regional `3.381670%p`, national
`1.841654%p`, winner accuracy `80%`, and identical SHA-256
`51B8BE3ABA23E4C25F6C884AAC280604BC2D47C5618613986AE59B81F98DD8A3`.

Further adjustment of the event levels or formula from these five outcomes is
not allowed. Replacement requires external validation or a future untouched
election.

Authoritative artifacts:

- `outputs/speech_derived_mega_intensity_v5/lineage_manifest.json`
- `outputs/automatic_mega_issue_intensity_v5_ablation/summary.csv`
- `outputs/automatic_mega_issue_intensity_v5_ablation/by_election.csv`
- `outputs/automatic_mega_issue_intensity_v5b_event_gate/summary.csv`
- `outputs/automatic_mega_issue_intensity_v5b_event_gate/by_election.csv`
- `outputs/active_presidential_nested_v16/pre_smooth_gate_snapshot.json`
