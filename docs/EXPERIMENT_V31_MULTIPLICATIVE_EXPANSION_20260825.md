<!-- active-model-version: v31 -->
# V31: an expansion that cannot reach zero

## Status

- Date: 2026-08-25
- Status: **promoted**; predecessor V30 frozen and unchanged
- Post-2022 outcomes used: none
- 2025 outcome used: none

## What was wrong

V29 expands each candidate's regional deviations around that candidate's own
weighted national level:

```
scaled = level + (1 + gain × predicted_third_share) × (pred − level)
```

This is linear in the deviation, so it has no lower bound. A candidate far
below their own national level is pushed toward zero and, for a large enough
factor, through it. V29 handled that with a per-election feasibility cap: stop
at the largest factor the election admits without a negative share.

The cap does something that was not noticed when it was adopted. It is
*defined* as the factor at which some region reaches zero. So the region that
sets the cap lands on **exactly zero**, every time the cap binds — not as an
estimate, but as the point where the arithmetic ran out.

| where | into the transform | published | realised |
| --- | ---: | ---: | ---: |
| 2017 홍준표 광주 | 3.549% | **0.000%** | 1.680% |
| 2017 홍준표 전남 | 4.289% | 0.847% | 2.636% |
| 2025 김문수 광주 | 2.670% | **0.000%** | not used |

The core model never produced these values. Its lowest output anywhere on the
panel is 2.09%; the zeros are made by the terminal transform. And because each
region is renormalised to 100%, the displaced mass does not vanish — in the
published 2025 artifact 광주 reads 이재명 87.688 / 김문수 0.000 / 이준석 12.312,
so one candidate's floor error inflates the other two.

## The change

```
scaled = level × (pred / level) ** (1 + gain × predicted_third_share)
```

The quantity being scaled is a ratio rather than a difference, so a positive
input stays positive for any factor. The constraint the cap enforced now holds
by the form, and **the cap is removed rather than adjusted**.

Everything else is V30's: the Ridge stack, the predictors, the shock structure,
the V28 external-model boundary, V27's regional transform, the forecast-time
weighting, and the gain.

### What it costs, and how it is paid back

The additive form conserves each candidate's weighted national level exactly —
the property V29 was promoted on. The multiplicative form does not: scaling
ratios preserves a geometric mean, not the arithmetic one the level is.
Measured, candidate levels move by up to `0.465%p`, which would make a
dispersion transform quietly change the national forecast.

Rescaling each candidate back would break the regional sums; renormalising the
regions would move the levels again. The two constraints are satisfied together
by alternating them to convergence. That introduces no constant — the targets
are the input levels and one — and it converges in 1–17 rounds on the panel.
Non-convergence raises rather than shipping a nearly-normalised artifact.

| form | regional macro | national macro | worst level shift | cap binds |
| --- | ---: | ---: | ---: | :-: |
| additive + zero cap (V30) | 2.566445 | **0.720437** | 1.1e-14 | 2017 |
| multiplicative alone | 2.516501 | 0.770450 | **0.465 %p** | never |
| **multiplicative + reconciliation (V31)** | **2.500701** | 0.724291 | 2.3e-13 | never |

## Result

| metric | V30 | V31 |
| --- | ---: | ---: |
| regional equal-election macro MAE | 2.5664447526782004 %p | **2.5007010072077227 %p** |
| national equal-election macro MAE | **0.7204374174124484 %p** | 0.7242913678028117 %p |
| winner accuracy | 0.8 | 0.8 |
| prediction rows | 232 | 232 |
| lowest predicted share | 0.0001 % | **1.9688 %** |

**The national figure is worse and the change was made anyway.** A prediction of
exactly zero for a major-party candidate in a metropolitan region is wrong in
kind, not in degree; `+0.0039%p` on a diagnostic aggregate does not weigh
against it. Both figures were measured before the decision, and the regional
improvement is recorded as an outcome, not as the reason.

The two rows that motivated the change:

| | V30 | V31 | realised |
| --- | ---: | ---: | ---: |
| 2017 홍준표 광주 | 0.0001% | **1.9688%** | 1.6800% |
| 2017 홍준표 전남 | 0.8470% | **2.5013%** | 2.6360% |

## The 2025 demonstration

Regenerated. The winner and the ranking are unchanged, and the national levels
move by `4.4e-14 %p` — the correction is entirely within regions.

| 광주 | into the transform | V30 published | V31 |
| --- | ---: | ---: | ---: |
| 이재명 | 85.292 | 87.688 | 85.931 |
| 김문수 | 2.670 | **0.000** | **2.053** |
| 이준석 | 12.038 | 12.312 | 12.016 |

The 2025 result was not consulted. It is not needed to see the defect: the
transform's own input said 2.67%, and no conservative candidate has ever
recorded below 1.68% in 광주 anywhere in the scored panel.

## Two conditions this version refuses

V30's shared weights module falls back to equal regions when the 1997 warmup
table is absent, and picks one value when an election-region's turnout
disagrees across its candidate rows. Both are fail-open, and both contradict
that module's own docstring. It is frozen into V30 by hash, so neither can be
fixed there; the V31 runner states the preconditions instead and stops rather
than running a different model quietly.

## Two errors in V30's frozen record, noted not rewritten

- `summary.json` publishes `"rule": "previous scored election's regional
  volumes; equal regions for the first"`. V30's code stopped doing that once
  1997 was sourced. The V31 runner publishes the rule its code follows.
- `audit_public_active_presidential_model_v30.py` requires the active version to
  score no worse than V29 on both macros. That would have rejected V30 itself
  had its recorded `+0.0119%p` projected cost materialised, and it would reject
  V31. The V31 audit carries no comparison against a predecessor's score; it
  checks that the published figures are the artifact's, that no share is zero,
  and that no cap reappeared.

Both are left in place because V30's artifact is frozen evidence. Rewriting a
frozen record to make it correct is worse than recording that it was wrong.

## The alternative that was measured and not taken

Letting the multiplicative form move the national levels — dropping the
reconciliation — is a coherent option and was measured rather than dismissed.

| | regional macro | national macro | worst level shift | 2025 national forecast |
| --- | ---: | ---: | ---: | --- |
| multiplicative alone | 2.516501 | 0.770450 | **0.465 %p** | moves |
| **multiplicative + reconciliation** | **2.500701** | 0.724291 | 2.3e-13 | unchanged to 4e-14 %p |

Both remove the zero equally; the difference is whether the side effect is
kept. It was not, for one reason: that 0.465%p is not a judgement about the
national level, it is the residue of a form that preserves a geometric mean
being asked to preserve an arithmetic one. Accepting it would mean a dispersion
transform silently changing the national forecast, and would make the V30→V31
difference unattributable — two things would have moved at once. If the national
levels should be adjusted, that belongs to a component with its own argument.

## Regional dispersion is still short, and not evenly

V31 removes the floor artifact. It does not resolve the wider question of
whether regional variation is under-reflected, so that is measured here rather
than left as an impression.

Regressing realised regional share on predicted, per candidate-election — slope
above 1 means the prediction is too flat:

| election | mean slope | realised / predicted regional SD |
| --- | ---: | ---: |
| pres_2002 | 0.740 | 0.925 |
| **pres_2007** | **1.265** | **1.361** |
| pres_2012 | 0.949 | 0.963 |
| pres_2017 | 0.882 | 1.018 |
| pres_2022 | 0.669 | 0.898 |

Across the panel the median slope is `0.970` and 6 of 14 candidate-elections are
under-dispersed, so **the model is not uniformly too flat** — in 2002, 2012 and
2022 it is if anything too spread. The shortfall is concentrated in the
elections with a substantial third candidate, and inside those it is
concentrated further:

| candidate-election | realised / predicted SD, V30 | V31 |
| --- | ---: | ---: |
| 이회창 2007 | 2.055 | **1.944** |
| 홍준표 2017 | 1.090 | 1.091 |
| 정동영 2007 | 1.077 | 1.127 |

이회창's realised regional spread is nearly twice the predicted one, and V31
narrows that only slightly. The expansion factor is `1 + predicted_third_share`,
which is `1.163` in 2007 against a shortfall of roughly `1.36` — the
parameter-free gain under-corrects there by construction, which is the price of
having no constant to select.

That is a real gap and it is left open. Closing it means either a gain that is
not 1, which the panel would be selecting, or a different index for the
correction — a modelling change with its own argument, not a floor fix.

## What this does not change

The scoring panel is still defined by which candidates cleared roughly 1% of the
actual vote (`DIAGNOSIS_SCORING_SCOPE_20260824.md`). The headline metric still
weights by `contest_votes`, which is the definition of national vote share
rather than a leak (`METRIC_WEIGHTING_20260825.md`). 2002–2022 remain a
development panel and 2025 remains a corrected demonstration.
