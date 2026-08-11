# V24 national interval scale refinement

## Status

- Date: 2026-08-03
- Status: shadow development calibration, not promoted
- Active production model: frozen V23
- Elections used: 2002-2022 only; 2002 supplies residual history but has no
  interval of its own
- Post-2022 outcomes used: no

## Motivation

The coherent V24 national candidate predictive distribution used one residual
scale, `0.75`, for all nominal levels. With 10,000 draws its equal-election
coverage and total widths were:

| Level | Coverage | Mean total width |
|---|---:|---:|
| 90% | 91.67% | 8.42%p |
| 95% | 100.00% | 9.79%p |
| 99% | 100.00% | 12.48%p |

The 95% result was conservative enough to justify a small refinement.

## Boundary audit

An initial single-seed run suggested scale `0.71`, but a different Monte Carlo
seed reduced 95% coverage to 91.67%. That value was rejected. The final audit
used five seeds (`2402, 3402, 4402, 5402, 6402`), 10,000 draws per seed, and
scales from `0.55` to `0.85`.

The smallest scale satisfying every seed at each nominal level was:

| Level | Scale | Minimum seed coverage | Mean width | Worst-seed mean width |
|---|---:|---:|---:|---:|
| 90% | 0.74 | 91.67% | 8.29%p | 8.31%p |
| 95% | 0.72 | 100.00% | 9.42%p | 9.45%p |
| 99% | 0.60 | 100.00% | 10.10%p | 10.15%p |

Raw level-specific intervals had 15 nesting violations across 50
seed-election-candidate combinations. The final output enforces:

1. the 95% interval contains the 90% interval;
2. the 99% interval contains the adjusted 95% interval.

After nesting enforcement, the 99% mean width is `10.12%p`; 90% and 95% widths
do not change.

## Result

The refined 95% national candidate interval has:

- mean total width: `9.42%p`, approximately `+/-4.71%p`;
- reduction from coherent single-scale baseline: `0.37%p` total width;
- minimum coverage across five seeds: `100%` on the development elections;
- region vote-volume uncertainty included;
- target election vote volumes used only for post-hoc coverage evaluation.

This is a level-calibrated predictive interval, not a classical coefficient
confidence interval and not one Monte Carlo distribution shared by all nominal
levels. It remains outcome-aware because historical coverage selected each
scale. Four evaluable elections and ten candidate outcomes are too few for
production promotion.

## Reproduction

```powershell
python scripts\evaluate_v24_national_interval_scale_refinement.py `
  --n-sim 10000 `
  --seeds 2402,3402,4402,5402,6402
```

Outputs:

- `outputs/experiments/v24_national_interval_scale_refinement/summary.json`
- `outputs/experiments/v24_national_interval_scale_refinement/scale_summary.csv`
- `outputs/experiments/v24_national_interval_scale_refinement/by_seed.csv`
- `outputs/experiments/v24_national_interval_scale_refinement/selected_nested_intervals_canonical_seed.csv`
- `outputs/experiments/v24_national_interval_scale_refinement/selected_nested_interval_summary.csv`

## Decision

Retain scale `0.72` for the 95%-only national shadow interval. Keep the coherent
single-distribution scale `0.75` as the reference distribution. Do not reduce
the 95% scale below `0.72`; both `0.70` and the seed-sensitive `0.71` failed the
stability requirement.
