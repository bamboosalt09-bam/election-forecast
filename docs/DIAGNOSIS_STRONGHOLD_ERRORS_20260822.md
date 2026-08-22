# The 10 %p cells: two distinct defects, not inherent shrinkage

## Status

- Date: 2026-08-22
- Status: diagnosis only; no change made
- Supersedes the mechanism proposed in `DIAGNOSIS_REGIONAL_SHRINKAGE_20260822.md`,
  whose measurement stands but whose explanation was too coarse

A single candidate-region cell missing by more than 10 %p is a large error, and
the macro figures hide how many there are and where.

## The tail

Across 232 candidate-region cells: median 2.73 %p, 90th percentile 6.84,
95th 9.35, maximum 15.69.

| threshold | cells | share |
| --- | ---: | ---: |
| > 5 %p | 51 | 22.0 % |
| > 10 %p | 11 | 4.7 % |
| > 15 %p | 1 | 0.4 % |

Of the eleven cells above 10 %p, **seven are in 2007**, two in 2012 and two in
2017. 2002 and 2022 have none.

| election | region | candidate | actual | predicted | error |
| --- | --- | --- | ---: | ---: | ---: |
| 2007 | 광주 | 정동영 | 86.92 | 71.23 | **-15.69** |
| 2007 | 제주 | 정동영 | 37.84 | 23.03 | -14.81 |
| 2007 | 전남 | 정동영 | 85.97 | 71.32 | -14.65 |
| 2007 | 전북 | 정동영 | 86.55 | 72.03 | -14.52 |
| 2007 | 제주 | 이명박 | 44.76 | 58.96 | +14.19 |
| 2007 | 충남 | 이회창 | 37.52 | 23.79 | -13.72 |
| 2007 | 대전 | 이회창 | 32.57 | 21.86 | -10.71 |
| 2017 | 대구 | 홍준표 | 55.25 | 42.74 | -12.52 |
| 2017 | 경북 | 홍준표 | 57.02 | 44.92 | -12.10 |
| 2012 | 세종 | 박근혜 | 52.17 | 41.77 | -10.41 |
| 2012 | 세종 | 문재인 | 47.83 | 58.23 | +10.41 |

Every one is a regional stronghold - 호남 for the liberal candidate, TK for the
conservative, 충청 for 이회창 - except 세종, which is the region that first
appears in 2012 and has no prior of its own.

## Extremity is not the cause

Cell-level regression toward the mean is present: sorting cells by how far the
realised share sits from that candidate's national level, the model
over-predicts below the level and under-predicts above it, with an overall
deviation-reproduction slope of 0.895.

But that slope is not uniform, and the elections that reproduce deviations
badly are not the ones with the most extreme regions.

| election | slope | cells with abs deviation > 20 | largest deviation | slope on those cells |
| --- | ---: | ---: | ---: | ---: |
| 2002 | **1.007** | 13 | 46.2 | **1.012** |
| 2007 | **0.742** | 11 | **57.8** | 0.780 |
| 2012 | **1.028** | 10 | 44.0 | 1.031 |
| 2017 | **0.715** | 8 | 29.2 | 0.685 |
| 2022 | **0.956** | 10 | 38.7 | 0.948 |

Every election has eight to thirteen extreme cells with deviations of 29 to 58
points. **2002 reproduces a 46-point deviation at slope 1.012.** The model can
represent strongholds; in 2007 and 2017 it does not. Inherent regression toward
the mean does not explain a model that is exact on 2002 and compressed on 2007.

## The stronghold error is the other candidates' excess

| 광주 2007 | actual | predicted | error |
| --- | ---: | ---: | ---: |
| 정동영 | 86.92 | 71.23 | **-15.69** |
| 이명박 | 9.37 | 17.07 | **+7.70** |
| 이회창 | 3.71 | 11.70 | **+7.99** |

-15.69 = +7.70 + 7.99, exactly. The model is not failing to raise 정동영; it is
failing to let the other two fall. The same holds in 전남. This is why the
effect appears only where a third candidate is substantial: with two candidates
there is nobody to hold a floor.

The declared floors are 0.01, far below the 11.70 the model gives 이회창 in
광주, so this floor is emergent in the fitted base rather than imposed by a
constant.

## Two different causes

### 2007: a party-less candidate loses his regional prior

| | bloc | 광주 `recent_bloc_base` | 광주 actual |
| --- | --- | ---: | ---: |
| 이회창 2002 | 국민의힘 | 0.0851 | 0.0359 |
| **이회창 2007** | **무소속** | **0.0000** | 0.0371 |

The same person, with the same conservative regional profile the panel has seen
twice, runs without a party in 2007 and his regional base becomes **zero**. With
no prior to anchor him the fitted base falls back to something generic, which
puts him at 11.70 in 광주 where he took 3.71.

`third_candidate_lineage.csv` already records `origin_lane = conservative` for
him, and V24 already routes inherited regional identity through dated party
rename and merger paths. The gap is that neither reaches a candidate whose bloc
is 무소속.

### 2017: shrinkage away from a prior that was right

| 2017 대구 | bloc | `recent_bloc_base` | predicted | actual |
| --- | --- | ---: | ---: | ---: |
| 홍준표 | 국민의힘 | **0.7007** | 0.4274 | 0.5525 |
| 문재인 | 더불어민주당 | 0.1594 | 0.3579 | 0.2651 |
| 안철수 | 제3지대 | 0.0875 | 0.2147 | 0.1824 |

Here the prior is present and good. 홍준표's TK base is 0.7007 and he took
0.5525; the model predicts 0.4274, undershooting past the outcome rather than
stopping at it. 문재인 is pushed from a 0.1594 prior to 0.3579 against an actual
0.2651.

Shrinking away from the 2016 bloc base is correct - the party had split and the
president had been removed - but the model shrinks past the answer. That is a
regime-response calibration question, and it sits with the mega-issue machinery
rather than with regional representation.

## What this changes

The earlier note framed this as one phenomenon scaling with third-candidate
size. The correlation is real - r = -0.975 - but it is a symptom of two
unrelated defects that both happen to occur in the two elections with a large
third candidate. A single dispersion correction cannot address both, which is
consistent with the swept rescale helping 2007 and 2017 while hurting 2002 and
2022.

Neither defect is a tuning question. The 2007 one is a representation gap with
a documented pre-election fact already sitting unused in the lineage table. The
2017 one is about how far the regime response is allowed to move a candidate
from a reliable prior.

Both were found by reading scored residuals, and that has to be declared in any
writeup. But the 2007 gap in particular is the kind that can be argued from
structure rather than from fit: a candidate does not lose his regional history
by filing as an independent.
