# District-first candidate regional base v6

## Correction

Candidate regional evidence must be reconstructed at the National Assembly
constituency level before it is rolled up to the 17 province-level forecast
regions. The previous official-history v5 compiler collapsed candidate office
history directly to a province and could not compare a candidate with the
party/bloc terrain of the exact constituency.

## Data

`data/raw/official_sources/nec_assembly_district_history.csv`

- NEC Assembly constituency count API;
- elections from 1992 through 2020;
- 69 cached API pages;
- 1,911 election-constituency units;
- 8,484 real candidate rows after removing zero-filled empty candidate slots;
- no post-2022 query.

The collection exposed and fixed an old parser defect: NEC fills unused
`dugsuNN` fields with the string `0`. The old parser treated these as candidate
rows. Aggregated history later dropped their empty party names, hiding the bug,
but district-level storage exposed it. `count_item_candidate_rows` now excludes
empty zero-filled slots.

## Reconstruction

For a candidate's prior Assembly constituency:

```text
personal_excess
  = candidate constituency vote share
  - same-election province mean share of the candidate's bloc

personal_constituency_signal
  = (0.15 * candidate share + 0.85 * positive personal_excess)
  * recency decay
```

Metropolitan/provincial and municipal executive history remains a separate
component. For non-major-party candidates, the latest strictly prior Assembly
election supplies party organization from:

```text
regional share excess
+ candidate-fielding coverage excess
+ constituency seat-share excess
```

Major-party organization is not added again because it is already present in
the active partisan terrain. The components are combined only after scoring at
the constituency level and then rolled up to province.

## Automatic bases recovered

The output automatically recovers, among others:

- Roh Moo-hyun: Busan personal constituency excess;
- Chung Dong-young: North Jeolla constituency history;
- Moon Jae-in: Busan constituency history;
- Lee Myung-bak: Seoul executive and constituency history;
- Ahn Cheol-soo: Honam organization from 2016 constituency-level third-bloc
  vote, candidate coverage, and seats;
- Lee Jae-myung: Gyeonggi executive history.

## Strict nested result

> Superseded comparison note: this original evaluator also enabled role-aware
> slot assignment and the speech-derived v4 candidate context. The numbers
> below are valid for that full variant, but they are not an isolated
> candidate-base-only ablation. The corrected comparison is documented in
> `docs/DISTRICT_CANDIDATE_BASE_CLEAN_V8_20260729.md`.

| Variant | Regional weighted MAE | National candidate MAE | Winner accuracy |
|---|---:|---:|---:|
| Active manual v16 | 3.3817%p | 1.8417%p | 0.80 |
| District-first v6 | 3.5014%p | 2.0565%p | 0.60 |

Election-level change:

| Election | Active regional MAE | District-first regional MAE | Change |
|---|---:|---:|---:|
| 2002 | 3.7484 | 3.9312 | +0.1828 |
| 2007 | 4.9237 | 4.7379 | -0.1857 |
| 2012 | 2.1992 | 2.6363 | +0.4370 |
| 2017 | 4.5431 | 4.5566 | +0.0135 |
| 2022 | 1.4939 | 1.6450 | +0.1510 |

## Decision

The district-first reconstruction is methodologically correct and retained,
but not promoted. It improves 2007 and reconstructs the intended factual
components, while a single constituency can still be spread too broadly over a
whole province. The next compiler must weight personal constituency evidence by
its electorate footprint or model at constituency level before aggregation.

Active v16 and its manual regional base remain unchanged.
