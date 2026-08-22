# Two structural gaps, built and measured

## Status

- Date: 2026-08-22
- Status: both implemented, both measured, **neither adopted**
- Defaults unchanged; V26 prediction hash unchanged
- Post-2022 outcomes used: none

`DIAGNOSIS_PERSON_HISTORY_AND_MOVABLE_MASS_20260822.md` named two gaps that are
provable as defects without an outcome. Both were then built and run.

## Gap 1: personal history

### What was built

`presidential_results_standardized.csv` carries `candidate_name` by region, and
이회창 appears in both 2002 and 2007. His own record is available; the model
just never looks at it.

The regularity that makes it usable is in the data. Between 2002 and 2007 his
national share fell from 46.6 % to 15.1 %, and:

| his 2002 regional share | mean absolute change to 2007 |
| --- | ---: |
| below 15 % (3 regions) | **1.25 %p** |
| 15 % or above (13 regions) | 33.81 %p |

광주 3.58 → 3.40, 전남 4.63 → 3.61, 전북 6.19 → 3.64, while 대구 fell 77.75 →
18.06. Hostile-territory floors survive a collapse in national standing;
favourable-territory shares scale with it.

Three variants were measured, capping a candidate at their own prior share in
regions where that prior sat below their own prior national level.

| variant | regional macro | national macro | worst cell | winners |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.7122 | **0.7210** | 15.69 | 4/5 |
| cap, lost history only | 2.6290 | 0.8048 | **14.81** | 4/5 |
| cap, every repeat candidate | **2.6047** | 1.0886 | 14.81 | 4/5 |
| cap with level held | 2.6224 | 0.7750 | 15.01 | 4/5 |

### Why not

**Every variant improves regional error and degrades national error.**

Scoping to candidates whose modelled history is actually empty is the right
restriction and was applied: 문재인 kept his party between 2012 and 2017, his
hostile-region support *rose* - 대구 19.5 → 21.8 - and capping him at his
earlier level is simply wrong. Restricting to the real defect cuts the capped
cells from 13 to 3, all 이회창 2007, and halves the national damage.

It does not remove it, and the reason is instructive. **이회창's national
prediction was already right**: 16.26 against a realised 16.77. His error was
entirely regional shape. Capping his 호남 cells fixes the shape and lowers a
level that needed no correction.

Holding his level - redistributing the released mass to his own other regions
rather than to rivals - recovers part of it, 0.8048 to 0.7750, but not all,
because renormalising each region hands some of it back.

The deeper obstacle is 정동영. He is **+0.99 over nationally and -15.69 under in
광주**, so he is substantially over everywhere else. Raising him in 호남 without
lowering him elsewhere pushes his national further wrong. The compression has
two ends and cannot be fixed at one of them.

## Gap 2: shock magnitude setting movable mass

### What was built

`strong_incumbent_veto` erodes the burdened candidate's core floor with

    eroded_floor = base_runner_floor * (1.0 - rupture_floor_activation)

where the activation is election-level - exactly 0.866 across all seventeen
regions in 2017 - and the floor is regional. A `floor_erosion_mode` of
`absolute` was added, subtracting one nationwide erosion in vote terms instead:

    eroded_floor = base_runner_floor - rupture_floor_activation * mean_core_floor

so the shock sizes the push and each region resists out of its own depth.

### It does exactly what it was designed to do

| 2017 홍준표 | actual | proportional | absolute | gain |
| --- | ---: | ---: | ---: | ---: |
| 경북 | 57.02 | -12.10 | **-8.71** | 3.39 |
| 대구 | 55.25 | -12.52 | **-9.39** | 3.13 |
| 경남 | 42.63 | -2.61 | **+0.04** | 2.57 |
| 부산 | 36.54 | -3.30 | -1.35 | 1.95 |

His mean absolute error falls from 3.780 to 3.253, and the gain is largest in
exactly the deepest-core regions. The mechanism is confirmed.

### Why not

The panel gets worse: regional 2.7122 to 2.7470, national 0.7210 to 0.8796.

| 2017 | regional MAE | national error |
| --- | --- | --- |
| 홍준표 | 3.932 → 3.863 | **+0.03 → +1.49** |
| 문재인 | 2.163 → 2.753 | **+0.27 → -1.19** |

The proportional erosion was landing the national level almost exactly - +0.03
and +0.27. Redistributing the erosion regionally changes the total transferred
and breaks that. **The same rate is doing two jobs**: setting the regional shape
of the transfer and setting its total size. Fixing one requires refitting the
other.

And that refit would have one observation. `rupture_floor_activation` is 0.0 in
2007, so the rupture erosion is exercised by **2017 alone** across the scored
panel.

## What both results have in common

Each gap is real, each fix does what it was designed to do in the cells it
targets, and each degrades the panel because it disturbs a calibration that the
existing form was implicitly carrying.

That is the identifiability problem stated concretely: in both cases one
quantity is setting both a shape and a level, and the panel offers one or two
observations to re-separate them. Neither result argues the diagnosis was wrong
- the mechanisms are confirmed in the cells they predict - only that this panel
cannot support the repair.

Both modes remain available and default to the shipped behaviour:
`third_candidate_lineage_constraint.recipient_weight_mode`,
`strong_incumbent_veto.floor_erosion_mode`.
