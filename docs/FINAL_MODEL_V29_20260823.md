# Final presidential model V29 — third-share regional dispersion expansion

V29 is V28 with one added transform. Each candidate's regional deviations are
expanded around that candidate's own vote-weighted national level, by a factor
of `1 + predicted_third_share`, after which each region is renormalised. The
V28 external-model boundary is unchanged and is re-audited here.

The transform addresses a specific, located defect: predicted regional spread
matches the realised spread in 2002, 2012 and 2022 and falls short of it in
2007 and 2017, the two scored elections with a substantial third candidate. The
slope of realised on predicted exceeds 1 in exactly those two — near 2 for
이회창 2007 — while sitting at 1 elsewhere. That is a calibration gap, not the
shrinkage a regularised predictor is supposed to have.

## Frozen development metrics

- regional equal-election macro MAE: `2.5736074405126663%p`
- national equal-election macro MAE: `0.7262497116354087%p`
- winner accuracy: `0.8`
- scored rows: `232`
- post-2022 outcomes used: `false`

The regional figure improves on V28's `2.638410502170951%p`. The national figure
is V28's `0.726249711635409%p` to nine decimals, and that is a property of the
transform rather than a result: candidate levels sum to one in every region, so
a uniform expansion around each candidate's own level leaves every region
summing to one and the renormalisation is a no-op. The measured largest
candidate-level shift is `5.6e-15` percentage points.

By election, regional weighted MAE:

| election | V28 | V29 |
| --- | ---: | ---: |
| pres_2002 | 2.795 | 2.957 |
| pres_2007 | 4.044 | 3.730 |
| pres_2012 | 2.387 | **2.387** |
| pres_2017 | 2.796 | 2.607 |
| pres_2022 | 1.171 | 1.187 |

2012 has no third candidate, so its expansion factor is exactly 1.0000 and it is
untouched. The transform scopes itself by the same quantity that diagnoses the
gap; no separate index decides where it acts.

## The gain is not fitted

At a gain of exactly 1 the expansion factor is the predicted third share itself,
so there is no constant for the scored panel to select. A swept gain of 0.50
gives a better regional macro — `2.555129%p` against `2.573607%p` — and is
**not** adopted, because it is a constant chosen on the same five outcomes it is
then scored against, and its advantage is smaller than the selection freedom
used to find it. The national macro is flat across the entire sweep, so this
choice costs nothing there.

The functional form remains a choice, and the decision to build a dispersion
correction at all was made by reading these five outcomes' residuals. That part
is in-sample and is not repaired by leaving the magnitude unfitted.

## Feasibility cap

2017 is capped below its nominal factor of 1.2461, at 1.1474. At the nominal
factor the expansion drives 홍준표's 광주 and 전남 predictions below zero, and a
negative share is not a dispersion. The cap is applied per election rather than
per candidate: differing factors would break the regional sums, making the
renormalisation non-neutral and moving the levels — the very failure the cap
exists to prevent. Measured both ways, per-candidate capping leaves a `0.124%p`
level shift in 2017 and uniform capping leaves `5.6e-15`.

## AI boundary

Unchanged from V28 and re-audited:

- hosted inference API: none
- downloaded model weights at runtime: none
- external neural encoder at runtime: none
- external-model-derived active input: one compact frozen candidate-issue aggregate
- fitted component: scikit-learn Ridge plus deterministic project transforms

The expansion adds no dependency and reads no outcome field. Its only input is
the model's own predicted regional shares.

## Evidence boundary

2002–2022 remain a development panel, not an untouched holdout. The 2025
artifact remains a corrected D-1 demonstration, not genuine prospective
validation. Future-election prospective validation is still absent.

2002 worsens under this transform, and it is the fold least able to afford it:
one training election, predictor VIF `1.0000`, and the largest single share of
the national macro. Expanding its deviations expands what is a level error. The
transform cannot distinguish compressed dispersion from a mis-levelled fold, and
2002 is the second case.
