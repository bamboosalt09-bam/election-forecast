# V24 regional-shape and interval calibration experiment

## Status

- Date: 2026-08-02
- Status: shadow experiment, not promoted
- Active production model remains V23.
- V23 prediction input SHA-256: `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`
- Elections used: 2002, 2007, 2012, 2017, 2022 only
- Post-2022 outcomes used: no
- Full regression validation after the structural extension: `476 passed`

The experiment reads the frozen V23 `nested_predictions.csv` and never changes
V23 configuration, code, or output files. The active V23 point-prediction column
is `layer_pred`. The generic `pred` column is an intermediate prediction and
does not reproduce the frozen V23 metrics.

## Regional-shape experiment

The new layer applies a small candidate-region log-share tilt from:

`regional_accent_signal_scaled * regional_accent_reliability * (1 - core_voting_mass_effective)`

After the tilt, iterative proportional fitting enforces both constraints:

1. Candidate shares sum to 100% in every region.
2. Candidate national totals under forecast-time region weights remain exactly
   equal to the V23 baseline totals.

The target election's actual turnout or vote volume is not used by the
correction. Region weights come from the latest prior presidential election.
Because no earlier vote-volume artifact exists for the first scored fold, 2002
uses an explicit equal-region fallback.

The wide gain ablation was `0.0` through `0.3`. It showed that large gains can
overfit the earliest folds. The strict sequential candidate therefore limits
the selectable grid to `0.0, 0.025, 0.05`; each target fold chooses its gain
using only earlier scored elections. The selected gain is 0 for 2002 and 0.05
for 2007-2022.

| Election | V23 regional MAE | V24 regional MAE | Change | V23 national MAE | V24 national MAE |
|---|---:|---:|---:|---:|---:|
| 2002 | 3.9816%p | 3.9816%p | 0.0000%p | 3.3220%p | 3.3220%p |
| 2007 | 4.7596%p | 4.6699%p | -0.0897%p | 1.6761%p | 1.6704%p |
| 2012 | 2.6669%p | 2.7147%p | +0.0478%p | 0.1627%p | 0.1714%p |
| 2017 | 4.0658%p | 3.9849%p | -0.0809%p | 2.6526%p | 2.6453%p |
| 2022 | 1.3656%p | 1.3697%p | +0.0042%p | 0.1758%p | 0.1767%p |
| Macro | **3.3679%p** | **3.3442%p** | **-0.0237%p** | **1.5978%p** | **1.5972%p** |

- Winner accuracy remains 4/5 (80%).
- Maximum forecast-weighted national-total drift is `1.17e-12`.
- National diagnostic values can move by a very small amount because the layer
  preserves totals under prior forecast weights while the post-hoc national MAE
  uses the target election's realized vote-volume weights.

Decision: keep as a shadow candidate. The macro gain is only 0.024%p, 2012 and
2022 regress, and there is no untouched historical holdout. The result supports
the mechanism but is not strong enough for V23 replacement.

## Predictive-interval experiment

For each target from 2007 onward, residual components are estimated only from
earlier scored elections. The first fold, 2002, is excluded because no strictly
prior scored residual panel is available. Residuals are represented in
compositional log-share space and split into:

1. candidate-common election shock,
2. regional component,
3. local row component.

Normal and empirical draws, local shrinkage, and residual scales from 0.5 to
2.5 were compared. The narrowest development-sample configurations meeting
each nominal weighted coverage level were:

| Nominal level | Structure | Scale | Observed coverage | Mean width |
|---|---|---:|---:|---:|
| 90% | normal common only | 2.0 | 91.56% | 17.44%p |
| 95% | normal common + regional | 2.0 | 95.58% | 25.34%p |
| 99% | normal + 50% local component | 2.5 | 99.06% | 50.14%p |

These rows are a development frontier, not deployable selected settings. They
use realized historical coverage to select a row, so the selection itself is
outcome-aware. They also select a different residual structure at each nominal
level, whereas a production distribution should generate nested intervals from
one coherent draw process.

These are regional row prediction intervals, not national candidate vote-share
intervals. Their width must not be presented as the uncertainty of the national
forecast.

## National candidate predictive intervals

The same compositional draws were aggregated with forecast-time region weights.
No target-election vote volume enters the forecast aggregation. Official target
vote volumes are used only to calculate the realized national result for the
coverage audit.

The coherent shadow distribution uses one setting for every interval level:

- empirical candidate-common + regional + local residual hierarchy,
- residual scale `0.75`,
- region weights from the latest prior presidential election,
- uncertainty in regional vote-volume composition estimated from strictly
  earlier election-to-election transitions.

| Nominal level | Equal-election coverage | Mean total width |
|---|---:|---:|
| 90% | 91.67% | 8.42%p |
| 95% | 100.00% | 9.79%p |
| 99% | 100.00% | 12.48%p |

Thus the 95% national predictive interval is about `+/-4.90%p` on average,
while the 95% regional row interval remains much wider. This reduction is due
to valid aggregation of partially offsetting regional residuals, not deletion
of residual variance.

Only four elections and ten candidate outcomes are evaluable because 2002 has
no strictly prior scored residual panel. Coverage therefore has very coarse
resolution and the `0.75` scale is development-outcome-aware. The coherent
national distribution remains shadow rather than production.

## Region-weight structural uncertainty

The national aggregation now draws regional vote-volume weights as well as vote
shares. The log-weight sigma is derived only from transitions ending before the
target election and shrunk when transition count is small:

| Target | Prior transitions | Log-weight sigma |
|---|---:|---:|
| 2007 | 0 | 0.0000 |
| 2012 | 1 | 0.0382 |
| 2017 | 2 | 0.0475 |
| 2022 | 3 | 0.0988 |

This increases the 95% national mean width only from `9.76%p` to `9.79%p`, so
regional turnout-composition uncertainty is real but not the main source of the
current national interval width.

Decision:

- reject the regional row interval for production because honest row coverage
  remains too wide;
- retain the coherent national candidate interval as a shadow candidate;
- do not claim a classical confidence interval. These are predictive intervals
  that include historical out-of-sample residual variation;
- no separate Ridge coefficient-covariance draw is added. Historical nested
  residuals implicitly contain estimation error, but explicit decomposition of
  coefficient and residual uncertainty remains unfinished.

## Reproduction

```powershell
python scripts\evaluate_v24_regional_shape_and_intervals.py --n-sim 10000 --seed 2402
python -m pytest -q tests\test_v24_calibration.py tests\test_regional_offset.py tests\test_predictor_interval_audit.py
```

Primary outputs:

- `outputs/experiments/v24_regional_shape_intervals/summary.json`
- `outputs/experiments/v24_regional_shape_intervals/nested_performance_by_election.csv`
- `outputs/experiments/v24_regional_shape_intervals/fixed_gain_ablation.csv`
- `outputs/experiments/v24_regional_shape_intervals/hierarchical_interval_scale_summary.csv`
- `outputs/experiments/v24_regional_shape_intervals/hierarchical_interval_frontier.csv`
- `outputs/experiments/v24_regional_shape_intervals/coherent_national_candidate_intervals.csv`
- `outputs/experiments/v24_regional_shape_intervals/coherent_national_candidate_interval_summary.csv`
- `outputs/experiments/v24_regional_shape_intervals/coherent_national_candidate_interval_by_election.csv`

## Next valid step

Do not increase the regional gain using these five elections. A valid promotion
test needs a genuinely later untouched election or more non-presidential
regional contests mapped into the same pre-election feature space. For
intervals, explicit coefficient draws and the residual hierarchy must be
generated in the same strict nested fold before production calibration is
selected. Withdrawal-event scenario spread and issue-measurement uncertainty
also remain separate structural components; they must not be represented by
arbitrarily inflating one global residual sigma.

## 2026-08-03 addendum

The national interval scale was refined with five seeds and 10,000 draws per
seed. The 95%-only level-calibrated scale is `0.72`, with `100%` minimum seed
coverage and `9.42%p` mean total width. The coherent one-distribution reference
remains scale `0.75`. See
`docs/EXPERIMENT_V24_INTERVAL_SCALE_REFINEMENT_20260803.md`.
