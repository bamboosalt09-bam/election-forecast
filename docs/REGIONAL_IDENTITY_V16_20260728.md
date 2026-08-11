# Regional Identity V16

## Decision

Promoted as `active_strict_nested_v16_regional_identity` with gain `0.10` and
regional transfer cap `0.04`. The v15 Chungcheong layer remains unchanged.

## Mechanism

1. Build regional party distributions from elections strictly before each
   target date.
2. Give Assembly PR, metropolitan-council PR, and local-council PR elections
   full weight; give presidential elections weight `0.35`.
3. Measure regional distinctiveness as total-variation distance from the
   cross-region median distribution.
4. Apply time decay, effective-sample shrinkage, and volatility reliability.
5. Route only to a candidate with a dated row in
   `data/raw/candidate_regional_base.csv` for that region and election.
6. Finance the transfer first from camps least compatible with the existing
   strictly prior regional camp profile. Preserve each region's vote-share sum.
7. Exclude Daejeon, Sejong, Chungbuk, and Chungnam because v15 already models
   their separate third-bloc identity reservoir.

The layer does not use target outcomes, does not infer new candidate links, and
is an exact no-op where dated evidence is absent.

## Sensitivity

| Gain | Regional MAE | National MAE | Worst election change | Winner accuracy |
|---:|---:|---:|---:|---:|
| 0.10 | **3.3817%p** | **1.8417%p** | 0.0000%p | 80% |
| 0.25 | 3.3662%p | 1.8336%p | 0.0000%p | 80% |
| 0.50 | 3.3471%p | 1.8342%p | 0.0000%p | 80% |

Although `0.25` and `0.50` improve the retrospective metrics more, selecting
either would use the five scored presidential outcomes to optimize the gain.
The promotion rule therefore keeps the smallest passing value, `0.10`.

## Active Result

| Election | V15 regional MAE | V16 regional MAE | Change |
|---|---:|---:|---:|
| 2002 | 3.7659%p | 3.7484%p | -0.0175%p |
| 2007 | 4.9237%p | 4.9237%p | 0.0000%p |
| 2012 | 2.1992%p | 2.1992%p | 0.0000%p |
| 2017 | 4.5663%p | 4.5431%p | -0.0233%p |
| 2022 | 1.5211%p | 1.4939%p | -0.0272%p |

Aggregate regional MAE changes from `3.3953` to `3.3817%p`; national candidate
MAE changes from `1.8483` to `1.8417%p`. Chungcheong predictions are bitwise
unchanged at CSV precision, regional sums remain one within `2.22e-16`, and the
winner result remains `4/5`.

## Limits

This is still a through-2022 development comparison, not an untouched external
holdout. Candidate-region links are sparse and historically curated. Regions
without a dated link deliberately receive no new transfer; their identity is
already represented by the general prior-party terrain and regional-accent
layers. The equal-share deviation slope changes from `0.9011` to `0.9006`, so
v16 does not solve the model's remaining central-regression problem.

## Verification

- full tests: `398 passed`;
- strict PIT deep audit: PASS, outcome invariance `215/215`;
- through-2022 weight-selection boundary audit: PASS;
- slot leakage audit: active old slot predictors absent;
- input manifest: 43 records, no 2025 path;
- maximum Chungcheong prediction change: `0`.

## Artifacts

- `presidential_issue_engine/regional_identity.py`
- `scripts/evaluate_general_regional_identity_v16.py`
- `tests/test_regional_identity.py`
- `outputs/regional_identity_v16_camp_donor_experiment/decision.json`
- `outputs/regional_identity_v16_camp_donor_experiment/sensitivity.csv`
- `outputs/active_presidential_nested_v16/regional_identity_audit.csv`
- `outputs/active_presidential_nested_v16/input_manifest.csv`
- `archives/experiments/regional_identity_v16_20260728/archive_manifest.csv`
