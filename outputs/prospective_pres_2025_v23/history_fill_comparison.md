# pres_2025 Intermediate-chain Correction Record

This record compares prospective runs only. It does not read or compare
against the 2025 election result, and none of the stages below is a scored
evaluation.

## Recalculation stages

| Stage | Kim Moon-soo | Lee Jae-myung | Lee Jun-seok |
|---|---:|---:|---:|
| Original deployment adapter | 39.1202% | 37.3354% | 23.5444% |
| Direct speech context before correction | 38.5672% | 37.2815% | 24.1513% |
| 22nd-Assembly roster alignment | 38.8083% | 36.8983% | 24.2935% |
| Explicit target direction restored | 39.2867% | 37.2224% | 23.4909% |
| Target landscape and third-candidate profile connected | 40.1018% | 37.0419% | 22.8563% |
| Automatic issue seeds connected, final rerun | **40.0938%** | **37.0492%** | **22.8571%** |

The final regional sample standard deviations are 20.029%p for Kim Moon-soo,
18.151%p for Lee Jae-myung, and 2.856%p for Lee Jun-seok. National aggregation
uses 2022 valid-vote volume, not a 2025 outcome field.

## Confirmed intermediate errors

1. The 2025 speaker profile initially lacked a term-aligned 22nd-Assembly
   roster. Only 9 of 147 speakers received a party bloc. The official term
   roster raises this to 87 of 147 speakers and 1,533 of 2,446 match rows.
2. `candidate_target_context_weekly.csv` contained signed person/party target
   evidence, but the forecast issue builder did not transform it into the
   issue-character overlay contract. Candidate issue directions therefore
   collapsed to zero. The corrected build retains non-zero direction in 20 of
   57 candidate-issue rows and emits four explicit mega attributions.
3. The target political landscape, automatic third-candidate profile, and
   automatic issue seed files were generated but omitted from the final V23
   runtime patch. The prospective runner now composes each target file with
   its historical counterpart before invoking the unchanged strict pipeline.
4. The old input manifest glob also listed inactive experimental CSVs as if
   they were runtime inputs. The manifest now enumerates only the active
   candidate-context and automatic-seed files.

## Extraction boundary validation

The supplied 16th-22nd Assembly workbook archive ends on 2024-12-31. A
resume-safe validation of all six 22nd-Assembly workbook classes found no rows
inside the 2025 D-90 campaign window. The 2,446-row 2025 slice is therefore not
a truncated workbook extraction: it is the complete eligible slice produced
from the separately collected official minutes supplement under the existing
campaign-window and D-1 availability rules.

The four historical candidate-context CSVs retain their frozen 13-row content
as an exact byte prefix and append only three `pres_2025` rows.

## Remaining model limitation

The corrected chain still gives the third candidate a large share. Diagnostics
show that this is no longer caused by a missing 2025 input: the slot-free Ridge
starts the three target candidates close to one third each, while candidate
conversion and third-candidate controls do not provide a strong enough
absolute major-party-versus-third-party scale. Correcting that behavior would
change model formulas or weights and must be evaluated as a new V24 candidate,
not patched into the frozen V23 demonstration.

## Outcome boundary

- `outcome_columns_used: []`
- `performance_metrics_computed: false`
- `pres_2025_outcome_present: false`
- forecast cutoff: `2025-06-02`
- V23 frozen prediction hash: unchanged
