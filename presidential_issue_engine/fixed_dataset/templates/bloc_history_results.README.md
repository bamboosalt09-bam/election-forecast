# bloc_history_results.csv

This optional input lets `presidential_issue_engine/issue_vote_engine.py` estimate
regional partisan structure from repeated historical results instead of
hand-coding regions such as Honam, TK, PK, or Gangwon.

Create this file when proportional-election data is ready:

```text
presidential_issue_engine/fixed_dataset/bloc_history_results.csv
```

Schema:

```csv
election_id,election_type,region_id,bloc,vote_share,data_quality_weight
```

Default election-type weights:

```text
presidential      0.8
assembly_pr       1.45
assembly_district 0.18
metro_council_pr  0.55
local_council_pr  0.11
metro_council_district 0.18
local_council_district 0.028
metro_governor    0.004
local_governor    0.003
education_superintendent 0.001
education_council 0.001
```

Recommended source mix:

- `presidential`: regional presidential vote shares, useful but down-weighted
  because candidate effects are part of the signal.
- `assembly_pr`: National Assembly proportional vote shares by region.
- `assembly_district`: National Assembly district vote shares by region. This
  should be treated as a low-weight directional prior because candidate and
  constituency effects are much stronger than in proportional elections.
- `metro_council_pr`, `local_council_pr`: local-election proportional council
  vote shares by region. These are party-list votes and are weighted above
  candidate-centered local races.
- `metro_council_district`, `local_council_district`: local-election district
  council vote shares by region, used at lower weight because constituency and
  candidate effects are stronger.
- `metro_governor`, `local_governor`: mayor/governor/head election vote shares,
  used at low weight because candidate effects are strong.
- `education_superintendent`, `education_council`: non-party education races,
  included only as very low-weight regional context.

The engine uses only elections earlier than the target election, applies time
decay, subtracts each bloc's national share from its regional share, and attaches
the resulting feature as `partisan_prior`.

If `presidential_issue_engine/fixed_dataset/bloc_history_results.csv` does not exist,
the engine falls back to `presidential_results_standardized.csv` so the workflow
remains runnable.
