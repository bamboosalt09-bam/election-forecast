# V25 intensity ladder: filling the gap between an inert and a saturated shock

## Status

- Date: 2026-08-22
- Status: measured candidate, **not promoted**; the active V25 pointer is unchanged
- V23 `nested_predictions.csv` SHA-256: `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b` (unchanged)
- V25 `nested_predictions.csv` SHA-256: `218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b` (unchanged)
- Post-2022 outcomes used: none
- New constants introduced: none
- Reproduce with `python scripts/evaluate_v25_intensity_ladder.py`

The candidate is two coupled changes. Both are measured here; neither is merged.
The comparison is a development table over the same five scored elections that
would decide promotion, so it is not an untouched holdout.

## The defect

`SHOCK_CLASS_INTENSITY` maps a shock class to one of `{0.50, 0.75, 1.00, 2.00}`
and `compile_direct_mega_scores` ramps attribution with

    intensity_activation = (mega_issue_intensity - 1.0).clip(0.0, 1.0)

The ramp is continuous, but the reachable inputs are not: the four class values
put activation at exactly `{0, 0, 0, 1}`. A direct shock is therefore either
inert or fully saturated, and no intermediate regime exists. That is what makes
the 2025 vocabulary registration move the forecast by 21 %p in a single step -
crossing the class boundary crosses every downstream threshold at once.

## Why the gap cannot be filled from above

The obvious construction scales intensity by how far the evidence sits above
the crisis gate. It is not available, and the reason is worth recording.

| | 2002 | 2007 | 2012 | **2017** | 2022 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_regime_evidence` | 0.080 | 0.384 | 0.288 | **0.677** | 0.374 | 0.744 |
| `accountability_component` | 0.773 | 0.843 | 0.603 | **0.896** | 0.860 | 0.806 |
| `source_rows` | 50,294 | 58,040 | 66,580 | **15,838** | 5,006 | 10,478 |
| margin above the crisis gate | — | — | — | **0.077** | — | 0.223 |

2017 clears `CRISIS_MIN_REGIME_EVIDENCE` by 0.027 of its 0.35 range. The
archetypal impeachment election sits essentially on the threshold, because a
snap election leaves the Assembly record a short window: 15,838 source rows
against 50,294 for 2002. Any margin-proportional intensity collapses 2017 to
1.077 and with it the one scored calibration point this path has. The existing
design comment - that raw mention volume must not shrink an institutional
crisis back toward an ordinary campaign issue - is empirically correct.

## The construction, filled from below

    proximity = clip(min_regime_evidence / CRISIS_MIN_REGIME_EVIDENCE, 0, 1)
              * clip(accountability_component / CRISIS_ACCOUNTABILITY, 0, 1)
    intensity = floor + (2.00 - floor) * proximity

The ceiling is the existing crisis level, each floor is that election's existing
intensity, and both gates are the classifier's own thresholds, now named in
`automatic_controls_v22` rather than inlined in `np.select` so that this reuse
is checkable. No new number enters. `proximity` is exactly 1.0 for any election
at or above both gates, so an election the classifier already calls a crisis
cannot move:

| election | intensity | activation |
| --- | --- | ---: |
| pres_2002 | 0.50 → 0.6837 | 0.0000 |
| pres_2007 | 1.00 → 1.5901 | 0.5901 |
| pres_2012 | 0.75 → 1.1958 | 0.1958 |
| **pres_2017** | **2.00 → 2.0000** | **1.0000** |
| pres_2022 | 1.00 → 1.5753 | 0.5753 |
| pres_2025 | 2.00 → 2.0000 | 1.0000 |

## Results

Two-by-two, so the ladder and the event-class alignment can be read apart.

| variant | regional macro | national macro | winners | worst burdened |
| --- | ---: | ---: | ---: | ---: |
| baseline | 3.440304 | 0.989620 | 4/5 | 3.512 |
| ladder | 5.003686 | 3.227753 | 4/5 | 11.410 |
| alignment | 3.440304 | 0.989620 | 4/5 | 3.512 |
| **ladder + alignment** | **3.421403** | **0.720994** | 4/5 | 3.512 |

Burdened-candidate error by election:

| election | burdened | baseline | ladder | alignment | ladder + alignment |
| --- | --- | ---: | ---: | ---: | ---: |
| pres_2002 | 노무현 | -3.512 | -3.512 | -3.512 | -3.512 |
| pres_2007 | 정동영 | 1.608 | 8.793 | 1.608 | **0.992** |
| pres_2012 | 박근혜 | 0.641 | 0.127 | 0.641 | **0.127** |
| pres_2017 | 홍준표 | 0.030 | 0.029 | 0.030 | **0.029** |
| pres_2022 | 이재명 | -0.251 | -11.410 | -0.251 | **-0.190** |

Three readings the factorial gives that a one-variable test would not.

**Alignment alone is bit-identical to baseline**, to every printed digit. Its
own docstring justifies it as a guard, and on the scored panel it currently
guards nothing: the elections whose winning issue is off-class sit at intensity
1.00 where activation is exactly zero, and the one election above the gate
already selects an on-class issue.

**The ladder alone is catastrophic**, and the cause is structural rather than a
tuning failure. 2007 and 2022 both have `security_nk` winning the winner-take-all
issue race in a class that does not contain it, and the retrospective applies no
alignment - `align_profile_to_event_class` is reached from the prospective
forecast only. **The intensity gate at 1.00 has been doing the alignment's job
on the panel.** Raising the floors exposes the unfiltered race immediately.

**Only the interaction helps.** Every election holds or improves, 2017 is
preserved by construction, and national macro falls 27 %.

## Why this is not promoted here

Three reasons, all of which should be cleared before the active pointer moves.

1. It is two coupled changes, and the second alters a scored path that is
   currently bit-identical. Promotion means the retrospective and the forecast
   start sharing a filter they do not share today.
2. The combination was chosen after comparing panel metrics. The 2025 outcome
   was never read, but the five scored outcomes selected the variant, so this is
   selection on the development sample in the same sense the rest of the V24 and
   V25 record is.
3. 2002 does not move at all and remains the worst election in the panel at
   -3.512 %p. The candidate improves the elections that were already close and
   leaves the outlier untouched, which is the pattern to be suspicious of.

## What it does not do

Nothing for the 2025 composition. 2025 sits above both gates, so its intensity
stays at 2.00 and the forecast is unchanged at 이재명 55.81 · 김문수 35.52 ·
이준석 8.66. The ladder adds rungs below the crisis level; it does not create a
regime between 2025 and a milder one. Concerns that the 2025 forecast suppresses
the conservative slot too hard are not addressed by this candidate.

## Rejected alongside

Gating `direct_mega_score` by the measured `target_specificity` axis. The five
taxonomy quality axes are computed and then unused by the attribution formula,
so gating by the axis that most directly expresses measurement quality is a
natural repair. It moves the only scored election that reaches the path: 2017's
burdened candidate goes from +0.030 to +0.441 %p and national macro from 0.990
to 1.013, against a 0.007 regional gain that is redistribution noise.
