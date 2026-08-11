# Speech-Derived Candidate Issue Profile V1

Date: 2026-07-28

Status: implemented and validated as an isolated strict experiment; not
promoted to the active v16 model.

## Objective

Replace retrospective candidate-specific values such as manually assigned
`association_strength=0.95` with deterministic values derived from
pre-election National Assembly issue evidence.

The compiler keeps three concepts separate:

1. Association is unsigned. Repeated issue discussion can connect a candidate
   or party lane to an issue but cannot create a vote direction.
2. Direction requires an explicit person, party, or government target.
3. Confidence is determined by evidence coverage, target-attribution quality,
   and directional consistency rather than by a human-entered score.

## Inputs

- `data/candidate_issue_link.csv`: candidate-lane issue emphasis from the full
  Assembly processing pass.
- `data/issue_salience_assembly.csv`: election-window issue salience.
- `data/raw/assembly_issue_character_overlay.csv`: conservative 5,000-row
  sentence-context overlay with explicit target attribution.
- `archives/experiments/manual_seed_lineage_v17_rejected_20260728/artifacts/assembly_speaker_issue_matches_15_22.csv`:
  speaker and issue posture for Assembly terms 15-22. The file contains only
  presidential scopes 2002-2022.
- The standardized result table is read only for candidate identity columns:
  `election_id`, `slot`, `candidate_name`, `party_name`, and
  `is_active_slot`. Vote and vote-share columns are not read by the profile
  builder.

The 5.55 GB `assembly_stance_rows_15_22.csv` was audited but is not a model
input. It contains a 2025 scope and every row has
`meeting_date_proxy_not_model_eligible`, so it cannot supply strict signed
candidate evidence without rebuilding its availability provenance.

## Formula

For each election, candidate slot, and issue:

```text
unsigned_association
  = geometric_mean(
      candidate_within_slot_emphasis_percentile,
      election_issue_salience_percentile,
      election_issue_evidence_coverage_percentile
    )

association_strength
  = 1 - (1 - unsigned_association)
        * (1 - explicit_target_attribution_confidence)

direction
  = target_signed_evidence / target_absolute_evidence
```

`direction=0` unless the source types include an explicit `person`, `party`,
or `government` target. Speaker party and general issue tone cannot create or
reverse direction.

For explicit directional rows:

```text
confidence
  = target_attribution_confidence * abs(direction)
```

For unsigned rows, confidence is the geometric mean of issue-quality and
evidence-coverage percentiles. This confidence may support issue salience and
association but cannot create a signed candidate effect while direction is
zero.

All formula components are universal across elections. There are no named
candidate, election-specific strength, or target vote-share constants.

## Generated artifacts

`outputs/speech_derived_issue_context_v1/` contains:

- `auto_issue_seed/candidate_issue_profile.csv`: 247 rows
- `auto_issue_seed/mega_issue_axis.csv`: 15 rows
- `auto_issue_seed/mega_issue_attribution.csv`: 9 rows
- `candidate_party_speech_context.csv`: 13 rows
- `candidate_party_tone_gap.csv`: 13 rows
- `candidate_public_treatment.csv`: 13 rows
- `candidate_vote_conversion_context.csv`: 13 rows
- `lineage_manifest.json`
- `active_run/`: isolated strict nested result
- `decision.json`

The candidate profile has 28 explicitly directional rows. Every other
candidate-issue direction is exactly zero.

## Lineage audit

The builder records the SHA-256 of every CSV read. It also rejects the run if
either of these files appears in the transitive read set:

- `data/raw/candidate_issue_profile.csv`
- `data/raw/mega_issue_attribution.csv`

Verified result:

- complete CSV inputs: 12
- forbidden manual issue seed reads: 0
- generated future-dated rows: 0
- generated elections: 2002, 2007, 2012, 2017, 2022 only
- post-2022 outcomes used: no

`third_candidate_profile.csv` remains a separate downstream candidate-regime
input and appears in the complete manifest. It is not used to calculate the
new issue association or direction, but it means the whole presidential model
is not yet free of every retrospective candidate prior.

## Strict nested result

| Model | Regional weighted macro MAE | National point MAE | Winner accuracy |
|---|---:|---:|---:|
| Active v16, manual issue ancestry | 3.3817%p | 1.8417%p | 80% |
| Manual ancestry removed, no replacement | 4.3489%p | 3.3247%p | 60% |
| Speech-derived profile v1 | **4.2790%p** | **3.1167%p** | **80%** |

| Election | Active v16 regional | Speech v1 regional | Active v16 national | Speech v1 national |
|---|---:|---:|---:|---:|
| 2002 | 3.7484 | 3.7745 | 3.1286 | 3.1674 |
| 2007 | 4.9237 | 5.0716 | 2.5487 | 2.8381 |
| 2012 | 2.1992 | 2.6787 | 0.2564 | 1.1554 |
| 2017 | 4.5431 | 8.4159 | 3.2492 | 8.1388 |
| 2022 | 1.4939 | **1.4546** | 0.0254 | 0.2840 |

The automatic profile improves materially over deleting the manual ancestry
and restores winner accuracy. It does not reproduce the active v16 score.

## 2017 limitation

The strict overlay finds explicit negative target evidence for the conservative
incumbent-continuity slot, including `regime_change`, but no comparable
explicit positive target evidence for Moon Jae-in. Renormalization therefore
releases support to both Moon and Ahn rather than identifying the principal
opposition beneficiary. The resulting national prediction is Moon 35.27%,
Hong 33.99%, and Ahn 30.74%.

This must not be fixed by restoring a manual Moon `+0.95` value. A defensible
next step is to expand target-attributed sentence coverage with source dates,
then combine negative incumbent responsibility with strictly prior candidate
viability when allocating the opposition-side beneficiary. That allocation
must be a universal rule and selected without reading the 2017 result.

## Verification

- New focused tests: 3 passed.
- Full suite: 401 passed.
- Repeated profile hash:
  `c01b76d69918a0c0de57379ebdc1366e0c74e889f70c4b2f704096d37a71a2c0`.
- Active v16 configuration and active output were not changed.

## Reproduction

```powershell
python scripts/build_speech_derived_issue_context.py
python scripts/evaluate_speech_derived_issue_context.py
pytest -q -p no:cacheprovider
```

