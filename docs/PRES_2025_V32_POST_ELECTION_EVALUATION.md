<!-- active-model-version: v32 -->
# V32 post-election evaluation for the 2025 presidential election

## Boundary

**V32 was frozen before this evaluation; 2025 outcomes were not used for V32
model selection, parameterization, or promotion.**

That sentence is the whole condition under which this number may be read. The
order was fixed deliberately and is recoverable from the repository rather than
from this claim:

| step | evidence |
| --- | --- |
| V32 promoted and sealed | commit `6f15664`, pointer `active_version: v32` |
| forecast artifact frozen | `outputs/prospective_pres_2025_v32/prospective_predictions.csv`, SHA-256 `d39329363d49ac9751857fc9efd2866554f08fe3909260e2258eaab744ef8fb7` |
| evaluation run afterwards | at commit `8afbedf`, recorded in the evaluation's own `scored_forecast.repository` block |
| forecast unchanged since | all 9 artifact files byte-identical before and after |

The realised outcome is not added to the model inputs, training panel, stage
selection, thresholds, or parameters. It is read only by
`scripts/evaluate_pres_2025_active.py`, from
`evaluations/pres_2025_v27/official_results.csv` — the same transcription every
evaluation uses, because the count is a property of the election rather than of
any model version.

The 2025 development path was outcome-informed and is not a genuine untouched
out-of-sample forecast. Publishing this score does not remove that limitation.

**V32 is not modified in response to this result**, and the result is worse than
its predecessor's. Committing to that in advance is the only thing that makes
the number worth publishing.

Reproduce: `python scripts/evaluate_pres_2025_active.py --version v32`.

## Result

A/B/C contest-normalised, against the official count.

| metric (%p) | V31 | **V32** | change |
| --- | ---: | ---: | ---: |
| regional, contest-vote weighted | 4.6345 | **4.9096** | **+0.2751** |
| regional, equal region | 4.6853 | **4.9801** | **+0.2948** |
| national, frozen forecast | 4.0540 | **4.3308** | **+0.2768** |

**V32 scores worse on every metric.** 45 of 51 region-candidate rows got worse
and 6 got better.

National levels:

| slot | candidate | actual | V31 | V32 |
| --- | --- | ---: | ---: | ---: |
| A | 이재명 | 49.96% | 55.81% (`+5.85`) | 56.23% (`+6.27`) |
| B | 김문수 | 41.61% | 35.52% (`−6.08`) | 35.11% (`−6.50`) |
| C | 이준석 | 8.43% | 8.66% (`+0.23`) | 8.66% (`+0.23`) |

## What actually moved

The direction is the informative part. Both versions already over-predicted
이재명 and under-predicted 김문수 by about six points. V32 did not introduce that
error — **it widened it**, by roughly 0.4 points in each direction.

The worsening concentrates where the conservative candidate was already most
under-predicted:

| region | slot | \|error\| V31 → V32 |
| --- | --- | --- |
| 경상북도 | B 김문수 | 4.42 → 5.66 (`+1.24`) |
| 경상북도 | A 이재명 | 5.17 → 6.41 (`+1.23`) |
| 대구광역시 | B 김문수 | 7.42 → 8.60 (`+1.18`) |
| 대구광역시 | A 이재명 | 8.65 → 9.82 (`+1.17`) |
| 세종특별자치시 | A 이재명 | 5.67 → 6.54 (`+0.87`) |

TK and 세종 — the regions where an active regional accent layer has the most to
say. Under V31 that layer contributed exactly nothing to the 2025 forecast,
because 27 columns were zero-filled. V32 made it contribute, and what it
contributed pushed these rows further from the count.

## How to read this, and how not to

**This is not evidence that V32's change was wrong.** V32 fixed a contract: the
prospective assembly could satisfy a missing feature by writing a zero, and five
model-active families were silently dead in the published forecast. That defect
existed whether or not repairing it improved a score. A version that restores a
feature layer to the value the estimator actually produces is more correct than
one that runs it at zero by accident, and it would still be more correct if the
score had improved instead.

**Nor is it evidence that V32's change was right.** The honest reading is
narrower: with the accent layer active, the model's existing bias against the
conservative candidate in TK gets amplified rather than corrected. That is
information about the *core model*, not about the contract — the 2.67% versus
8.10% gap in 광주 recorded in V31's evaluation made the same point from the
other side. The terminal layers are not where this error lives.

What would be wrong is to now tune the accent gain, or the contract, until this
number improves. That is fitting to the 2025 outcome through a slower route, and
it is rejected for the same reason a direct fit is.

## Relationship to V31's evaluation

`evaluations/pres_2025_v31/` is preserved byte-for-byte and is **not**
superseded. It scores V31's forecast, which is a different artifact; V32's
forecast moved by up to `1.2381%p` regionally. Two evaluations of two forecasts,
neither replacing the other.

## A defect this evaluation exposed

`evaluations/pres_2025_v32/` already existed in the repository, tracked, and
scored a forecast that had since been regenerated: it recorded
`forecast_sha256: e19ef33d…` while the committed artifact hashed to
`d3932936…`. The field was written and never read back, so a published score
that belonged to no surviving forecast sat in the tree unnoticed.

The evaluator now also records the commit and the working-tree state
(`scored_forecast.repository`), and `tests/test_evaluation_matches_its_forecast.py`
fails when any tracked evaluation's recorded hash stops matching the artifact it
names. A provenance field nothing checks is how the stale one survived.

## Related

- `PRES_2025_V31_POST_ELECTION_EVALUATION.md` — the predecessor's score
- `FINAL_MODEL_V32_20260826.md` — what V32 changed and why
- `EXPERIMENT_V32_PROSPECTIVE_FEATURE_CONTRACT_20260826.md` — the five dead
  families and the evidence for each
