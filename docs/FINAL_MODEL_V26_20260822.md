# Final model V26: graded mega-issue intensity with event-class alignment

## Status

- Date: 2026-08-22
- Active pointer: `v26`, predecessor `v25`
- Rollback: V25, prediction SHA-256
  `218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b`
- V23 frozen reference `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`
  and V24 `edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a`
  are unchanged
- Post-2022 outcomes used: none
- Ridge model, predictors, alpha, ballot panel and the three structural
  postprocesses: identical to V25
- New constants introduced: none
- Runner: `python scripts/run_active_presidential_model_v26.py`
- Audit: `python scripts/audit_public_active_presidential_model_v26.py`

V26 changes exactly two things and nothing else.

## The defect V26 addresses

`SHOCK_CLASS_INTENSITY` maps a shock class to one of `{0.50, 0.75, 1.00, 2.00}`,
and `compile_direct_mega_scores` ramps attribution with

    intensity_activation = (mega_issue_intensity - 1.0).clip(0.0, 1.0)

The ramp is continuous; its reachable inputs are not. The four class values put
activation on `{0, 0, 0, 1}`, so a direct political shock is either inert or
fully saturated and no intermediate regime is representable. A single change of
class therefore crosses every downstream threshold at once.

## Change 1: graded intensity from classifier-gate proximity

    proximity = clip(min_regime_evidence / CRISIS_MIN_REGIME_EVIDENCE, 0, 1)
              * clip(accountability_component / CRISIS_ACCOUNTABILITY, 0, 1)
    intensity = floor + (2.00 - floor) * proximity

The ceiling is the existing institutional-crisis level, each floor is the
election's existing V23 intensity, and both gates are the classifier's own
thresholds — named in `automatic_controls_v22` rather than inlined in
`np.select` so the reuse is checkable. Nothing is fitted.

The rule is one-sided by construction: `proximity` is exactly 1.0 at or above
both gates, so an election the classifier already calls a crisis cannot move,
and an election absent from the diagnostics keeps its floor.

| election | V23/V25 intensity | V26 intensity | activation |
| --- | ---: | ---: | ---: |
| pres_2002 | 0.50 | 0.683736 | 0.0000 |
| pres_2007 | 1.00 | 1.590052 | 0.5901 |
| pres_2012 | 0.75 | 1.195804 | 0.1958 |
| **pres_2017** | **2.00** | **2.000000** | **1.0000** |
| pres_2022 | 1.00 | 1.575327 | 0.5753 |
| pres_2025 | 2.00 | 2.000000 | 1.0000 |

The table is written to `outputs/automatic_controls_v26/mega_issue_intensity.csv`
by `scripts/build_automatic_controls_v26.py`, so the promoted model reads the
same kind of control input every previous version read and the difference from
V23 is a diff rather than a runtime patch.

**Filling from above was rejected.** Scaling intensity by the margin above the
gate collapses 2017 to 1.077: it clears the regime gate by 0.027 of its 0.35
range, because a snap election leaves the Assembly record a short window —
15,838 source rows against 50,294 for 2002. The existing design note, that raw
mention volume must not shrink an institutional crisis back toward an ordinary
campaign issue, is empirically correct.

## Change 2: event-class alignment on the scored path

`align_profile_to_event_class` keeps only issues declared compatible with the
election's shock class. It was reachable from the prospective forecast only;
the retrospective ran `compile_direct_mega_scores` on the raw profile. V26
applies it on both.

The two changes are inseparable, and the two-by-two shows why.

| variant | regional macro | national macro |
| --- | ---: | ---: |
| V25 baseline | 3.440304 | 0.989620 |
| graded intensity alone | 5.003686 | 3.227753 |
| alignment alone | 3.440304 | 0.989620 |
| **both (V26)** | **3.421403** | **0.720994** |

(These are the unweighted regional means the experiment harness reports; the
headline figures below are the contest-vote weighted metric the pointer records.)

Alignment alone is bit-identical to V25 because the elections whose leading
political-shock issue is off-class sit at intensity 1.00, where activation is
exactly zero. Grading alone is catastrophic because raising those floors
exposes the winner-take-all issue race on 2007 and 2022, where `security_nk`
leads a class that does not contain it. **The intensity gate at 1.00 had been
doing the alignment's job on the scored panel.**

## Performance

| metric | V25 | V26 |
| --- | ---: | ---: |
| regional contest-vote weighted, equal-election macro MAE | 2.773943 %p | **2.712233 %p** |
| national candidate equal-election macro MAE | 0.989620 %p | **0.720994 %p** |
| winner accuracy | 80 % (4/5) | 80 % (4/5) |
| prediction rows | 232 | 232 |

By election, national candidate MAE and the burdened-candidate error:

| election | burdened | V25 national | V26 national | V25 error | V26 error |
| --- | --- | ---: | ---: | ---: | ---: |
| pres_2002 | 노무현 | 2.342 | 2.342 | -3.512 | -3.512 |
| pres_2007 | 정동영 | 1.305 | **0.661** | 1.608 | **0.992** |
| pres_2012 | 박근혜 | 0.641 | **0.127** | 0.641 | **0.127** |
| pres_2017 | 홍준표 | 0.201 | 0.201 | 0.030 | **0.029** |
| pres_2022 | 이재명 | 0.459 | **0.274** | -0.251 | **-0.190** |

No election degrades. 2017, the one election that already reached full
saturation, is preserved by construction.

## What this does not fix

**2002 does not move at all** and remains the worst election in the panel at
-3.512 %p. Its intensity is 0.6837, still below the activation gate, so the
direct mega path stays inert there. V26 improves the elections that were
already close and leaves the outlier exactly where it was. That is the pattern
to remain suspicious of, and it is the first place to look next.

**The 2025 forecast is unchanged.** 2025 sits above both gates, so its
intensity stays at 2.00 and the composition remains 이재명 55.81 · 김문수 35.52
· 이준석 8.66. The ladder adds rungs below the crisis level; it does not create
a regime between 2025 and a milder one, and it does not address concerns that
the conservative slot is suppressed too hard.

## Selection disclosure

The pairing was chosen by comparing the same five scored outcomes that measure
it. No 2025 outcome was read at any point, and no constant was introduced — the
ceiling, the floors and both gates all existed before. But the five scored
elections are a development sample, not an untouched holdout, and this promotion
does not change that. The alignment in particular moves a scored path that was
bit-identical under V25, so V26's retrospective and its forecast now share a
filter they did not share before.

Full measurement record: `docs/EXPERIMENT_V25_INTENSITY_LADDER_20260822.md`.

## Rejected alongside

- **Margin-proportional intensity from above** — collapses 2017 to 1.077.
- **Graded intensity without the alignment** — national macro 0.990 → 3.228.
- **Gating `direct_mega_score` by `target_specificity`** — the five measured
  taxonomy quality axes are computed and then unused by the attribution
  formula, so gating by the axis expressing measurement quality is a natural
  repair; it moves 2017's burdened candidate from +0.030 to +0.441 %p.
