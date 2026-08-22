# V27 post-election evaluation for the 2025 presidential election

## Boundary

This is a post-election evaluation of the already-frozen 2025-06-02 D-1 V27
forecast. The realized outcome is not added to the model inputs, training
panel, stage selection, thresholds, or parameters. It is read only by
`scripts/evaluate_pres_2025_v27.py`.

The earlier 2025 development path was outcome-informed and is not a genuine
untouched out-of-sample forecast. Publishing this score does not remove that
limitation.

## Official result source

- Publisher: National Election Commission of the Republic of Korea
- Record: `제21대 대통령선거 투표구별 개표결과`
- Published: 2025-06-13
- Page: `https://www.nec.go.kr/site/lvt/ex/bbs/View.do?bcIdx=293099&cbIdx=1129`
- Attached workbook SHA-256:
  `b146e76cc2a3536eb970138f16185da0d5bc0a3acdc367d06feea80bb3e0e936`
- Repository evaluation extract:
  `evaluations/pres_2025_v27/official_results.csv`
- Extract SHA-256:
  `87d3370f5a38d57a50e542edc40158786ba98766f70c6ab483d752cdf0ac35b9`

The 17 regional totals and national totals for all five candidates in the NEC
workbook were checked against the evaluation extract. National valid votes are
34,980,616. The three modeled candidates account for 34,600,675 votes.

## Comparable composition

V27 predicts only slots A, B, and C and normalizes them to one within every
region. Therefore the headline actual share is also normalized within those
three candidates:

`actual_contest_share[c,r] = votes[c,r] / sum(votes[A:B:C,r])`

Comparing the three-candidate prediction directly with raw all-candidate vote
shares would compare different compositions. Raw shares remain in
`national_scored.csv` as a descriptive field, not the headline score.

## Metrics

| metric | value |
|---|---:|
| regional A/B/C contest-vote-weighted point MAE | **4.628096%p** |
| regional equal-region point MAE | **4.696797%p** |
| frozen national-forecast candidate point MAE | **4.053941%p** |

The regional weighted metric averages 51 candidate-region absolute errors,
using each region's realized A/B/C vote volume. The equal-region metric gives
each of the 51 rows equal weight.

The national headline uses the actually frozen `national_summary.csv`, whose
weights were the predeclared 2022 valid-vote volumes. It does not replace that
forecast with a post-election reaggregation using 2025 regional vote volumes.

| candidate | frozen prediction | actual A/B/C share | signed error |
|---|---:|---:|---:|
| 이재명 | 55.8143% | 49.9629% | +5.8514%p |
| 김문수 | 35.5242% | 41.6051% | -6.0809%p |
| 이준석 | 8.6615% | 8.4320% | +0.2295%p |

## Reproduction

```bash
python scripts/evaluate_pres_2025_v27.py
```

Outputs:

- `evaluations/pres_2025_v27/summary.json`
- `evaluations/pres_2025_v27/regional_scored.csv`
- `evaluations/pres_2025_v27/national_scored.csv`
