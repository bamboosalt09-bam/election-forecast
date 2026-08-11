# Active V20: Automatic Third-Candidate Character Subset

Date: 2026-08-01

## Decision

V20 is the active V10-lineage model. It retains V18 automatic viability and
replaces two additional third-candidate character fields with strictly prior
election- and Assembly-derived values:

- `anti_major_party_appeal`;
- `regional_base_overlap`.

`centrist_appeal` remains manual. Third-candidate pressure remains manual.
V18 is preserved at `backups/model_checkpoints/20260801_active_v18/`.

## Factorial ablation

Each field was replaced alone before testing the passing pair together.

| Field | Regional MAE | Change | National MAE | Change | Max election regression | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| centrist appeal | 3.234416%p | +0.017867 | 1.465640%p | -0.013742 | +0.089345 | fail |
| anti-major-party appeal | 3.216857%p | +0.000307 | 1.479226%p | -0.000155 | +0.001263 | pass |
| regional-base overlap | 3.216921%p | +0.000372 | 1.481548%p | +0.002166 | +0.000951 | pass |

The equivalence gate was fixed as:

- regional degradation no greater than 0.01%p;
- national degradation no greater than 0.01%p;
- maximum single-election regression no greater than 0.05%p.

The combined V20 result is:

| Metric | V18 | V20 | Change |
| --- | ---: | ---: | ---: |
| Regional equal-election macro MAE | 3.216549%p | 3.217252%p | +0.000703%p |
| National equal-election macro MAE | 1.479382%p | 1.481393%p | +0.002011%p |
| Maximum election regression | - | +0.002174%p | - |
| Winner accuracy | 0.80 | 0.80 | 0.00 |

V20 is an automation-equivalence promotion, not a performance-improvement
claim. The two fields were selected after through-2022 ablation; there is no
untouched historical holdout for this choice. This limitation must remain in
performance reporting.

## Derivation

The automatic source is the V15 election-derived profile. It combines:

- direct-party vote competitiveness;
- district organization evidence;
- prior presidential stature;
- won-office history;
- candidate and party Assembly political vectors;
- candidate regional footprint and party concentration.

The target election outcome is not read. Only 2002, 2007, and 2017 active
third-candidate rows have sufficient evidence for replacement. The 2012 and
2022 factual contest rows are preserved.

## Reproducibility

The canonical V20 run regenerates V14, V14b, V15, and V20b profiles before
forecast execution. Its `nested_predictions.csv` SHA-256 is identical to the
V20b confirmation ablation.

The first direct invocation audit found that
`scripts/run_current_presidential_model.py` did not add the repository root to
`sys.path`, so file execution could not import the `scripts` package. The
wrapper was corrected and the actual canonical command was rerun with explicit
exit-code checking. It completed successfully and reproduced the same
prediction SHA-256.

Canonical artifacts:

- `scripts/run_current_presidential_model.py`
- `scripts/run_active_presidential_model_v20.py`
- `data/config/active_presidential_model_v20.json`
- `outputs/automatic_third_character_v20b/`
- `outputs/automatic_third_character_v20b_ablation/`
- `outputs/active_presidential_nested_v20/`
- `backups/model_checkpoints/20260801_active_v20_verified/`
