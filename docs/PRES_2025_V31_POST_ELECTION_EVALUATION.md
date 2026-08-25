<!-- active-model-version: v31 -->
# V31 post-election evaluation for the 2025 presidential election

## Boundary

This scores the already-frozen 2025-06-02 D-1 **V31** forecast against the
later official count. The realised outcome is not added to the model inputs,
training panel, stage selection, thresholds, or parameters. It is read only by
`scripts/evaluate_pres_2025_active.py`.

The 2025 development path was outcome-informed and is not a genuine untouched
out-of-sample forecast. Publishing this score does not remove that limitation.

**It is also not a selection criterion.** No promotion in this repository has
cited a 2025 score; V28 through V31 were each justified structurally, and each
justification was written before this number existed. It is published because
leaving V27's score beside a V31 model states something no longer true.

Reproduce: `python scripts/evaluate_pres_2025_active.py`. The official count is
the same transcription the V27 evaluation used; it is a property of the
election, not of a model version.

## Result

| 2025 post-hoc point error | V27 | **V31** | change |
| --- | ---: | ---: | ---: |
| regional, actual A/B/C vote weighted MAE | 4.6281 %p | **4.6345 %p** | +0.0064 |
| regional, equal-region MAE | 4.6968 %p | **4.6853 %p** | −0.0115 |
| frozen national forecast MAE | 4.0539 %p | **4.0540 %p** | +0.0001 |

Four versions of model work moved the 2025 score by hundredths of a percentage
point, in both directions. That is the honest headline.

| candidate | frozen national forecast | actual, A/B/C normalised | error |
| --- | ---: | ---: | ---: |
| 이재명 | 55.81 % | 49.96 % | +5.85 %p |
| 김문수 | 35.52 % | 41.61 % | −6.08 %p |
| 이준석 | 8.66 % | 8.43 % | +0.23 %p |

## What V31 fixed, and what it did not

V30 published 김문수's 광주 at exactly `0.000%` — the feasibility cap landing on
the region that set it. V31 cannot emit a zero and puts him at `2.053%`.

The realised figure is `8.10%`.

| 광주, 김문수 | value |
| --- | ---: |
| V30 published | 0.000 % |
| into V31's transform | 2.670 % |
| **V31 published** | **2.053 %** |
| V27 published | 2.694 % |
| **realised (A/B/C normalised)** | **8.104 %** |

So V31 is a large improvement on V30 in this row and still misses by six
points, and the expansion moves it *away* from the result: the stage feeding
the transform said 2.670% and the transform lowered it. The core model, not the
terminal transform, is what predicts 2.67% where the outcome was 8.10%.

## The expansion is not aimed at 2025

Across the 51 forecast rows, the terminal expansion moved **21 toward** the
result and **30 away** from it, for a net regional MAE change of 4.6949 →
4.6853. It is close to a coin flip, which is what a transform justified on
structure rather than on this outcome should look like.

The largest remaining errors are all the conservative candidate in regions
where he was under-predicted:

| region | into the transform | published | realised | error |
| --- | ---: | ---: | ---: | ---: |
| 전라남도 | 31.88 % | 31.32 % | 41.05 % | 9.74 %p |
| 광주광역시 | 26.04 % | 25.11 % | 33.64 % | 8.53 %p |
| 충청남도 | 35.82 % | 35.58 % | 43.72 % | 8.15 %p |
| 전북특별자치도 | 34.83 % | 34.56 % | 42.13 % | 7.58 %p |
| 대구광역시 | 59.03 % | 60.79 % | 68.21 % | 7.42 %p |

Note the shape: the misses are not confined to the candidate's weak regions.
대구, his strongest, is under-predicted by 7.4 points as well. The 2025 error is
dominated by a national level that was 6 points low for him, not by the regional
distribution around it.

## What this says about the open question

The standing open question is whether regionalism is under-reflected for
candidates the model has little history on. 김문수 was a first-time
presidential candidate, and the pattern here is consistent with that: the model
placed him too low almost everywhere, most severely in 호남, and the terminal
dispersion machinery — which operates *around* a candidate's national level —
cannot repair a level that is wrong.

That is recorded, not acted on. Acting on it means changing how a
shallow-history candidate's level and regional profile are formed, which is a
modelling change needing its own argument, and it must not be argued from this
page.
