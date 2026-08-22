# Regionalism: party regionalism is measured and has no corrective path

## Status

- Date: 2026-08-22
- Status: diagnosis only; no change made
- This is the upstream cause the veto and dispersion work kept running into

## Two kinds of regionalism, one of them represented

The model carries a regional identity layer. What it actually holds is
**candidate footprint** - all 24 rows of it:

    노무현 부산·서울   권영길 울산·경남·대전·서울   이명박 서울
    정동영 전북        박근혜 대구                문재인 부산
    홍준표 경남·서울   안철수 광주·전남·전북 외    이재명 경기

Every entry is somewhere the candidate personally ran or governed. That is a
real signal and it is thinly covered - 24 rows against 232 candidate-region
cells, and 이회창 has none at all.

It is also the wrong signal for the failures. **홍준표's TK strength is not
personal footprint** - he is from 경남 창녕, and 경남 0.850 is what the table
records for him. His 대구 and 경북 strength is 국민의힘's party regionalism, and
there is no row for it. 정동영 has 전북 0.126 and nothing for 광주 or 전남.

## Party regionalism is measured, and it is good

`recent_bloc_base` carries it correctly:

| candidate | region | bloc base | predicted | actual | predicted − base |
| --- | --- | ---: | ---: | ---: | ---: |
| 정동영 2007 | 광주 | 0.8072 | 0.7123 | **0.8692** | -0.0950 |
| 정동영 2007 | 전남 | 0.8143 | 0.7132 | **0.8597** | -0.1011 |
| 정동영 2007 | 전북 | 0.7848 | 0.7203 | **0.8655** | -0.0645 |
| 홍준표 2017 | 대구 | 0.7007 | 0.4274 | 0.5525 | -0.2733 |
| 홍준표 2017 | 경북 | 0.7069 | 0.4492 | 0.5702 | -0.2577 |

In 호남 2007 the realised share sits **above** the bloc base and the model
predicts below it. Using the raw bloc base in 광주 would have missed by 6.2 %p
where the model misses by 15.7 - the untouched prior is two and a half times
more accurate there than the model that processes it.

TK 2017 is different: the base overshoots by 14.8 points, the model undershoots
by 12.5, and the truth lies between them.

## The gap

There is no layer that carries party regionalism as a **corrective**. The chain
is:

1. `recent_bloc_base` states the regional prior correctly
2. the fitted base stage moves away from it - correctly in direction, since 2017
   had a party split and a removed president
3. the amount of that movement is not checked against anything
4. the regional identity layer would be the check, but it carries candidate
   footprint and has no row for these cells

`general_regional_identity` was superseded and its audit file is 5 bytes - it
never fires. `unified_exact_lineage_identity` replaced it using footprint and a
genealogy graph, neither of which reaches a candidate's own bloc depth. The
applied shift in 대구, 경북, 광주 and 전남 is **exactly 0.0000** against a
needed 0.19 to 0.26.

## What this is not

**Not "trust the prior more".** Anchoring predictions toward the normalised bloc
prior was measured and degrades every election monotonically
(`EXPERIMENT_DISPERSION_ALTERNATIVES_20260822.md`). The prior is stale in
aggregate and the model is right to move away from it.

The defect is narrower: the **size** of the move is unconstrained in the
strongholds. 2007 호남 should not have moved at all; 2017 TK should have moved
about half as far.

## Why this ranks above the other findings

Every downstream repair attempted so far runs into it.

The layered veto cannot help, because blocking the whole transfer still leaves
홍준표 7 points short - he is already below where he should be when the veto
starts. The absolute-erosion mode has the same ceiling. The dispersion
expansion helps precisely because it is putting back, after the fact, the
stronghold spread that regionalism failed to supply in front - which is why it
works on 2007 and 2017 and does nothing on the three elections whose
strongholds came through intact.

Repair order follows: regionalism first, concrete erosion second, dispersion
third and probably smaller once the first is done.

## Not yet established

Why the move away from the prior is sized as it is, and whether it is one
mechanism or several. That question sits in the fitted base stage, which no work
in this session has instrumented.
