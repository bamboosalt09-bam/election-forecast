# pres_2025 Intermediate-chain Correction

## Scope

This correction addresses missing or discarded pre-election inputs in the
forecast-only 2025 path. It does not change a V23 coefficient, gain, threshold,
formula, active pointer, or historical prediction. The 2025 outcome remains
forbidden from fitting, tuning, comparison, and scoring.

## Root causes

### 1. Speaker-to-party mapping

The D-1 campaign slice contains 147 unique speakers. The earlier profile used
historical member metadata that did not adequately cover the 22nd Assembly,
so only 9 speakers received a party bloc. The official Assembly member API was
queried only for speakers already present in the bounded slice. Term-aligned
22nd-Assembly party, constituency, and mandate fields were retained; mutable
biographical fields were discarded.

Result: 87 speakers and 1,533 of 2,446 issue-match rows are now roster-mapped.
The roster input and its source/hash manifest are committed next to the 2025
context.

### 2. Directional evidence representation

`candidate_target_context_weekly.csv` already carried signed and absolute
directional evidence, but `build_speech_derived_issue_context.py` expected an
issue-character overlay and therefore replaced the absent overlay with neutral
direction. The new bridge converts the existing target evidence into that
contract and reuses the established reliability formula. It adds no forecast
weight or threshold.

Result: 20 of 57 candidate-issue rows have non-zero direction and four mega
attribution rows are emitted.

### 3. Generated target artifacts omitted at runtime

The 2025 political landscape, automatic third-candidate profile, candidate
issue profile, mega axis, and mega attribution were generated but not patched
into both engine instances used by the strict prospective runner. The runner
now combines each target artifact with its through-2022 automatic history in a
temporary directory and invokes the existing V23 pipeline.

The final input manifest lists explicit active files rather than every CSV in
the candidate-context directory.

## Reproduction

```powershell
python scripts\collect_pres_2025_assembly_roster.py

python -m presidential_issue_engine.build_assembly_speaker_influence `
  --matches data\raw\official_sources\assembly_pres_2025_context\pres_2025_speaker_issue_matches.csv `
  --roster data\assembly_roster.csv `
  --roster-extra data\raw\official_sources\assembly_pres_2025_context\assembly22_speaker_roster.csv `
  --member-history data\raw\assembly_member_history.csv `
  --speaker-out data\raw\official_sources\assembly_pres_2025_context\assembly_speaker_influence_pres_2025.csv `
  --issue-out outputs\pres_2025_speaker_profile\assembly_issue_speaker_weighted.csv `
  --scope-out outputs\pres_2025_speaker_profile\issue_scope_weights_speaker.csv `
  --conversion-out outputs\pres_2025_speaker_profile\issue_vote_conversion_speaker.csv `
  --diagnostics-out data\raw\official_sources\assembly_pres_2025_context\assembly_speaker_influence_pres_2025_diagnostics.csv

python scripts\build_speech_derived_candidate_context_v2.py `
  --output-dir data\raw\official_sources\assembly_pres_2025_context\candidate_context_v2 `
  --assembly-matches data\raw\official_sources\assembly_pres_2025_context\pres_2025_speaker_issue_matches.csv `
  --candidates data\raw\official_sources\pres_2025_candidate_registry.csv `
  --speaker-profile data\raw\official_sources\assembly_pres_2025_context\assembly_speaker_influence_pres_2025.csv `
  --preserve-history-dir data\raw

python scripts\run_prospective_forecast.py --version v23
```

The 22nd-Assembly workbook validation may be repeated with
`extract_assembly_speaker_issue_matches.py --assemblies 22`. The archive ends
before the campaign window, so the official minutes supplement remains the
only eligible 2025 source.

## Corrected prospective output

| Candidate | Predicted share |
|---|---:|
| Kim Moon-soo | 40.0938% |
| Lee Jae-myung | 37.0492% |
| Lee Jun-seok | 22.8571% |

These are unscored forecast outputs. They must not be described as error,
accuracy, improvement, or deterioration relative to the realized election.

## Integrity checks

- forecast cutoff: `2025-06-02`
- training latest election: `pres_2022`
- outcome columns used: none
- performance metrics computed: false
- historical candidate-context prefix: exact
- frozen V23 `nested_predictions.csv` SHA-256:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`

## Residual limitation

After all known target inputs are connected, the preliminary slot-free Ridge
still places the three candidates near equal shares before candidate-strength
postprocessing. The remaining large third-candidate output is therefore a V23
structural calibration limitation, not another absent 2025 CSV. Any adjustment
requires an independently audited V24 ablation and human promotion decision.
