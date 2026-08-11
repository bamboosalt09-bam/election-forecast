# Universal Evidence-Gated Pipeline V8

Date: 2026-07-27

## Decision

Promote `structural_mega_shock_regime` as the single active historical and
future deployment pipeline. Remove target-by-target candidate-stage selection
from active execution. Retain all five candidate stages only as diagnostics.

## Reason

V7 fell back to `strict_base` for 2002 and 2007 because it required two prior
scored elections before selecting a stage. This disabled dated direct-party
terrain and bounded regime evidence in 2007 even though those inputs were
available before the election. The fallback caused the model to predict Chung
Dong-young narrowly above Lee Myung-bak.

V8 applies one policy to every target. It is not a `pres_2007` branch. Layers
activate only from their ordinary evidence gates:

1. strict rolling Ridge fit;
2. candidate conversion, regionalism, and within-bloc transfer;
3. core, critical-support, and swing terrain response;
4. direct mega-issue attribution when intensity exceeds its gate;
5. governing-camp shock response from explicit target evidence;
6. contest-regime and cumulative-rejection response.

## Result

| Metric | V7 | V8 |
|---|---:|---:|
| Regional weighted macro MAE | 4.8571%p | 3.9499%p |
| National candidate macro MAE | 3.5059%p | 2.6298%p |
| Winner accuracy | 60% | 80% |
| 2007 national candidate MAE | 10.2024%p | 4.8983%p |

2007 Lee Myung-bak changes from `38.837%` to `47.636%`; the actual
three-candidate share is `54.140%`. The 2002 national MAE increases by
`0.9236%p`. The later three elections are unchanged because v7 already selected
the full pipeline for them.

## Leakage boundary

- Target rows remain excluded from every Ridge fit.
- No realized A/B slot predictor is active.
- No historical target outcome selects a stage in v8 because the deployment
  stage is fixed for all targets.
- Numeric parameters remain historically development-selected through 2022.
- Observed contest-vote aggregation remains a post-election diagnostic.
- The indirect manual-seed lineage found in the accompanying leakage audit is
  not changed by this promotion and remains follow-up work.

## Backup and recovery

Pre-change snapshot:

`archives/experiments/pre_universal_pipeline_v8_20260727/`

Completed v8 snapshot:

`archives/experiments/universal_pipeline_v8_20260727/`

V7 outputs remain available under:

`outputs/active_presidential_nested_v7/`
